"""Consequence-aware replay retention policies for RBCL experiments."""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, List, Optional

import torch

from avalanche.training.storage_policy import (
    BalancedExemplarsBuffer,
    ReservoirSamplingBuffer,
)


ConsequenceProvider = Callable[[], Optional[torch.Tensor]]


class ConsequenceAwareReservoirSamplingBuffer(ReservoirSamplingBuffer):
    """Reservoir buffer with bounded class-consequence retention priorities.

    Every candidate keeps its original uniform random key.  When the buffer is
    reduced, the key is transformed as ``u ** (1 / w_y)`` where ``w_y`` is in
    ``[1, 1 + strength]``.  Larger class consequence therefore increases the
    chance that an already observed sample survives, without assigning any
    sample an infinite or deterministic priority.  With ``strength=0`` (or in
    ``uniform`` mode), this is exactly ordinary reservoir selection.
    """

    def __init__(
        self,
        max_size: int,
        *,
        consequence_provider: ConsequenceProvider,
        mode: str = "c_aware",
        strength: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__(max_size)
        if mode not in {"uniform", "c_aware"}:
            raise ValueError("mode must be one of: uniform, c_aware")
        if strength < 0:
            raise ValueError("strength must be non-negative")
        self.consequence_provider = consequence_provider
        self.mode = mode
        self.strength = float(strength)
        self.eps = float(eps)

    def _selection_scores(self, data, base_keys: torch.Tensor) -> torch.Tensor:
        if self.mode == "uniform" or self.strength == 0 or len(data) == 0:
            return base_keys

        consequence = self.consequence_provider()
        targets = getattr(data, "targets", None)
        if consequence is None or targets is None:
            return base_keys

        consequence = torch.as_tensor(consequence, dtype=torch.float32).detach().cpu()
        labels = torch.as_tensor(list(targets), dtype=torch.long)
        if consequence.numel() == 0 or labels.numel() != base_keys.numel():
            return base_keys

        safe_labels = labels.clamp(min=0, max=consequence.numel() - 1)
        values = consequence[safe_labels]
        spread = values.max() - values.min()
        if float(spread) <= self.eps:
            return base_keys

        normalized = (values - values.min()) / (spread + self.eps)
        class_weights = 1.0 + self.strength * normalized
        return base_keys.clamp_min(self.eps).pow(class_weights.reciprocal())

    def update_from_dataset(self, new_data):
        new_keys = torch.rand(len(new_data))
        all_keys = torch.cat([new_keys, self._buffer_weights])
        all_data = new_data.concat(self.buffer)
        scores = self._selection_scores(all_data, all_keys)
        _, sorted_idxs = scores.sort(descending=True)
        selected = sorted_idxs[: self.max_size]
        self.buffer = all_data.subset(selected)
        # Keep the immutable uniform keys, not the transformed scores.  This
        # permits later consequence changes to re-rank the surviving samples.
        self._buffer_weights = all_keys[selected]

    def resize(self, strategy, new_size: int):
        self.max_size = int(new_size)
        if len(self.buffer) <= self.max_size:
            return
        scores = self._selection_scores(self.buffer, self._buffer_weights)
        _, sorted_idxs = scores.sort(descending=True)
        selected = sorted_idxs[: self.max_size]
        self.buffer = self.buffer.subset(selected)
        self._buffer_weights = self._buffer_weights[selected]


class ConsequenceAwareExperienceBalancedBuffer(
    BalancedExemplarsBuffer[ConsequenceAwareReservoirSamplingBuffer]
):
    """Experience-balanced memory with C-aware within-experience retention.

    Capacity remains equally divided across experiences, matching Avalanche's
    default ReplayPlugin policy.  Only the selection of samples that survive a
    group's later shrink is changed, which limits the experimental confound.
    """

    def __init__(
        self,
        max_size: int,
        *,
        consequence_provider: ConsequenceProvider,
        mode: str = "c_aware",
        strength: float = 1.0,
    ):
        super().__init__(max_size=max_size, adaptive_size=True)
        self.consequence_provider = consequence_provider
        self.mode = mode
        self.strength = float(strength)
        self._num_exps = 0
        self.history: List[Dict[str, object]] = []

    def _new_group(self, size: int) -> ConsequenceAwareReservoirSamplingBuffer:
        return ConsequenceAwareReservoirSamplingBuffer(
            size,
            consequence_provider=self.consequence_provider,
            mode=self.mode,
            strength=self.strength,
        )

    def post_adapt(self, agent, exp):
        self._num_exps += 1
        lengths = self.get_group_lengths(self._num_exps)

        new_buffer = self._new_group(lengths[-1])
        new_buffer.update_from_dataset(exp.dataset)
        self.buffer_groups[self._num_exps - 1] = new_buffer

        for length, group in zip(lengths, self.buffer_groups.values()):
            group.resize(agent, length)
        self._record_state()

    def _record_state(self) -> None:
        group_lengths: Dict[str, int] = {}
        class_counts: Dict[str, int] = {}
        for group_id, group in self.buffer_groups.items():
            group_lengths[str(group_id)] = len(group.buffer)
            targets = getattr(group.buffer, "targets", [])
            for class_id, count in Counter(int(x) for x in targets).items():
                key = str(class_id)
                class_counts[key] = class_counts.get(key, 0) + int(count)

        consequence = self.consequence_provider()
        priority_snapshot = None
        if consequence is not None:
            priority_snapshot = (
                torch.as_tensor(consequence).detach().cpu().float().tolist()
            )
        self.history.append(
            {
                "experience": self._num_exps - 1,
                "size": len(self.buffer),
                "max_size": self.max_size,
                "group_lengths": group_lengths,
                "class_counts": class_counts,
                "consequence_before_update": priority_snapshot,
            }
        )

    def summary(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "strength": self.strength,
            "max_size": self.max_size,
            "history": self.history,
        }

