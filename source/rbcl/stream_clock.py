"""Task-ID-free replay driven by a fixed current-sample clock."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from avalanche.training.plugins import SupervisedPlugin


class StreamClockReplayPlugin(SupervisedPlugin):
    """Online reservoir replay whose state updates every ``clock_samples``.

    The plugin never reads ``strategy.experience`` or task labels to decide
    when memory changes.  It stores each current-stream sample only during
    the first epoch of an experience, so the clock advances once per observed
    stream item rather than once per optimization epoch.  Replay is injected
    before the forward pass and therefore newly committed memory is available
    to subsequent batches in the same nominal experience.
    """

    def __init__(
        self,
        *,
        mem_size: int,
        clock_samples: int,
        seed: int,
        min_replay_age_samples: int = 0,
        coverage_floor: int = 0,
        coverage_strength: float = 0.0,
        replay_loss_scale: float = 1.0,
        current_uncertainty_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        if mem_size <= 0:
            raise ValueError("mem_size must be positive")
        if clock_samples <= 0:
            raise ValueError("clock_samples must be positive")
        if min_replay_age_samples < 0:
            raise ValueError("min_replay_age_samples must be non-negative")
        if coverage_floor < 0:
            raise ValueError("coverage_floor must be non-negative")
        if not 0.0 <= coverage_strength <= 1.0:
            raise ValueError("coverage_strength must be in [0, 1]")
        if replay_loss_scale <= 0:
            raise ValueError("replay_loss_scale must be positive")
        if current_uncertainty_lambda < 0:
            raise ValueError("current_uncertainty_lambda must be non-negative")
        self.mem_size = int(mem_size)
        self.clock_samples = int(clock_samples)
        self.min_replay_age_samples = int(min_replay_age_samples)
        self.coverage_floor = int(coverage_floor)
        self.coverage_strength = float(coverage_strength)
        self.replay_loss_scale = float(replay_loss_scale)
        self.current_uncertainty_lambda = float(current_uncertainty_lambda)
        self._generator = torch.Generator().manual_seed(int(seed))
        self._memory_x: List[torch.Tensor] = []
        self._memory_y: List[int] = []
        self._memory_arrival: List[int] = []
        self._pending_x: List[torch.Tensor] = []
        self._pending_y: List[int] = []
        self._seen_samples = 0
        self._epoch_index = 0
        self._cached_current_x: Optional[torch.Tensor] = None
        self._cached_current_y: Optional[torch.Tensor] = None
        self._update_positions: List[int] = []
        self._replay_examples = 0

    def before_training_exp(self, strategy, **kwargs) -> None:
        self._epoch_index = 0

    def after_training_epoch(self, strategy, **kwargs) -> None:
        self._epoch_index += 1

    def before_forward(self, strategy, **kwargs) -> None:
        # At this point the base dataloader contains only current-stream data.
        current_n = int(strategy.mb_x.shape[0])
        strategy.rbcl_current_batch_size = current_n
        strategy.rbcl_replay_loss_scale = self.replay_loss_scale
        strategy.rbcl_current_uncertainty_lambda = self.current_uncertainty_lambda
        if self._epoch_index == 0:
            self._cached_current_x = strategy.mb_x[:current_n].detach().cpu().clone()
            self._cached_current_y = strategy.mb_y[:current_n].detach().cpu().clone()
        else:
            self._cached_current_x = None
            self._cached_current_y = None

        eligible = [
            index
            for index, arrival in enumerate(self._memory_arrival)
            if self._seen_samples - arrival >= self.min_replay_age_samples
        ]
        if not eligible:
            return
        replay_n = current_n
        if self.coverage_floor > 0:
            eligible_label_counts: Dict[int, int] = {}
            for index in eligible:
                label = self._memory_y[index]
                eligible_label_counts[label] = eligible_label_counts.get(label, 0) + 1
            weights = torch.tensor(
                [1.0 / eligible_label_counts[self._memory_y[index]] for index in eligible]
            )
            selected = torch.multinomial(
                weights, replay_n, replacement=True, generator=self._generator
            ).tolist()
        elif self.coverage_strength > 0:
            eligible_label_counts: Dict[int, int] = {}
            for index in eligible:
                label = self._memory_y[index]
                eligible_label_counts[label] = eligible_label_counts.get(label, 0) + 1
            # A bounded inverse-frequency tilt: 0 is uniform sample replay and
            # 1 is class-uniform replay. This keeps within-class diversity.
            weights = torch.tensor(
                [
                    float(eligible_label_counts[self._memory_y[index]])
                    ** (-self.coverage_strength)
                    for index in eligible
                ]
            )
            selected = torch.multinomial(
                weights, replay_n, replacement=True, generator=self._generator
            ).tolist()
        else:
            selected = torch.randint(
                len(eligible), (replay_n,), generator=self._generator
            ).tolist()
        sample_indices = [eligible[index] for index in selected]
        replay_x = torch.stack([self._memory_x[index] for index in sample_indices]).to(
            strategy.device
        )
        replay_y = torch.tensor(
            [self._memory_y[index] for index in sample_indices],
            dtype=strategy.mb_y.dtype,
            device=strategy.device,
        )
        strategy.mbatch[0] = torch.cat([strategy.mb_x, replay_x], dim=0)
        strategy.mbatch[1] = torch.cat([strategy.mb_y, replay_y], dim=0)
        if len(strategy.mbatch) >= 3:
            replay_task = torch.zeros(
                replay_n,
                dtype=strategy.mbatch[-1].dtype,
                device=strategy.device,
            )
            strategy.mbatch[-1] = torch.cat([strategy.mbatch[-1], replay_task], dim=0)
        self._replay_examples += replay_n

    def after_training_iteration(self, strategy, **kwargs) -> None:
        if self._cached_current_x is None or self._cached_current_y is None:
            return
        self._pending_x.extend(self._cached_current_x.unbind(0))
        self._pending_y.extend(int(label) for label in self._cached_current_y.tolist())
        self._cached_current_x = None
        self._cached_current_y = None
        while len(self._pending_x) >= self.clock_samples:
            chunk_x = self._pending_x[: self.clock_samples]
            chunk_y = self._pending_y[: self.clock_samples]
            del self._pending_x[: self.clock_samples]
            del self._pending_y[: self.clock_samples]
            self._commit_chunk(chunk_x, chunk_y)

    def _commit_chunk(self, chunk_x: List[torch.Tensor], chunk_y: List[int]) -> None:
        for sample_x, sample_y in zip(chunk_x, chunk_y):
            self._seen_samples += 1
            if len(self._memory_x) < self.mem_size:
                self._memory_x.append(sample_x)
                self._memory_y.append(sample_y)
                self._memory_arrival.append(self._seen_samples)
                continue
            replacement = int(
                torch.randint(
                    self._seen_samples, (1,), generator=self._generator
                ).item()
            )
            if replacement < self.mem_size:
                slot = replacement
                if self.coverage_floor > 0:
                    label_counts: Dict[int, int] = {}
                    for label in self._memory_y:
                        label_counts[label] = label_counts.get(label, 0) + 1
                    replaceable = [
                        index
                        for index, label in enumerate(self._memory_y)
                        if label_counts[label] > self.coverage_floor
                    ]
                    if replaceable:
                        # Preserve the minimum per-label coverage whenever the
                        # fixed memory has surplus copies. The slot choice stays
                        # random and no task or boundary information is read.
                        slot = replaceable[replacement % len(replaceable)]
                elif self.coverage_strength > 0:
                    label_counts: Dict[int, int] = {}
                    for label in self._memory_y:
                        label_counts[label] = label_counts.get(label, 0) + 1
                    incoming_count = label_counts.get(sample_y, 0)
                    softer_slots = [
                        index
                        for index, label in enumerate(self._memory_y)
                        if label_counts[label] > incoming_count
                    ]
                    if softer_slots and bool(
                        torch.rand((), generator=self._generator).item()
                        < self.coverage_strength
                    ):
                        # Unlike a floor, this is only a probabilistic redirect;
                        # any class may still be replaced by Reservoir sampling.
                        pick = int(
                            torch.randint(
                                len(softer_slots), (1,), generator=self._generator
                            ).item()
                        )
                        slot = softer_slots[pick]
                self._memory_x[slot] = sample_x
                self._memory_y[slot] = sample_y
                self._memory_arrival[slot] = self._seen_samples
        self._update_positions.append(self._seen_samples)

    def summary(self) -> Dict[str, object]:
        return {
            "mem_size": self.mem_size,
            "clock_samples": self.clock_samples,
            "min_replay_age_samples": self.min_replay_age_samples,
            "coverage_floor": self.coverage_floor,
            "coverage_strength": self.coverage_strength,
            "replay_loss_scale": self.replay_loss_scale,
            "current_uncertainty_lambda": self.current_uncertainty_lambda,
            "seen_current_samples": self._seen_samples,
            "memory_size": len(self._memory_x),
            "pending_samples": len(self._pending_x),
            "update_positions": self._update_positions,
            "replay_examples_injected": self._replay_examples,
            "eligible_memory_size": sum(
                self._seen_samples - arrival >= self.min_replay_age_samples
                for arrival in self._memory_arrival
            ),
            "uses_task_id": False,
        }
