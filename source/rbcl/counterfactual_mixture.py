"""Strict-causal counterfactual mixture replay for taskification robustness."""

from __future__ import annotations

import random
from collections import Counter

import torch
import torch.nn.functional as F

from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import get_last_fc_layer

from .causal_er_ace import CausalERACE


class CausalCounterfactualMixtureERACE(CausalERACE):
    """ER-ACE with a counterfactually gated dual-memory replay mixture.

    The total memory is split equally between a progressive class-balanced
    store and a global Reservoir. At a fixed update cadence, classifier-head
    copies estimate current progress and memory gain for three candidate
    replay mixtures. The safest candidate with the largest memory gain is
    used until the next controller event. Controller copies never write back
    to the training model, and all samples enter both stores only after their
    causal first-epoch update.
    """

    def __init__(
        self,
        *,
        controller_every: int = 500,
        controller_rho: float = 0.90,
        mixture_candidates: tuple[float, ...] = (0.25, 0.50, 0.75),
        seed: int = 0,
        **kwargs,
    ) -> None:
        if controller_every <= 0:
            raise ValueError("controller_every must be positive")
        if not 0.0 < controller_rho <= 1.0:
            raise ValueError("controller_rho must be in (0, 1]")
        candidates = tuple(float(value) for value in mixture_candidates)
        if not candidates or any(not 0.0 <= value <= 1.0 for value in candidates):
            raise ValueError("mixture candidates must lie in [0, 1]")
        super().__init__(
            seed=seed,
            memory_policy="hybrid",
            **kwargs,
        )
        self.controller_every = int(controller_every)
        self.controller_rho = float(controller_rho)
        self.mixture_candidates = candidates
        self.coverage_fraction = 0.50
        self._controller_rng = random.Random(int(seed) + 2_000_033)
        self._controller_history: list[dict] = []

    def _coverage_items(self) -> list[tuple[torch.Tensor, int, int]]:
        return [
            (sample_x, label, task_id)
            for label in sorted(self._class_memory)
            for sample_x, task_id in self._class_memory[label]
        ]

    def _reservoir_items(self) -> list[tuple[torch.Tensor, int, int]]:
        return list(
            zip(
                self._hybrid_memory_x,
                self._hybrid_memory_y,
                self._hybrid_memory_tid,
            )
        )

    @staticmethod
    def _draw_indices(
        rng: random.Random, population: int, count: int
    ) -> list[int]:
        if count <= 0 or population <= 0:
            return []
        return rng.sample(range(population), min(population, count))

    def _mixture_items(
        self,
        coverage_fraction: float,
        count: int,
        rng: random.Random,
    ) -> list[tuple[torch.Tensor, int, int]]:
        coverage = self._coverage_items()
        reservoir = self._reservoir_items()
        total_available = len(coverage) + len(reservoir)
        if total_available == 0 or count <= 0:
            return []
        count = min(count, total_available)
        desired_coverage = int(round(count * coverage_fraction))
        coverage_count = min(desired_coverage, len(coverage))
        reservoir_count = min(count - coverage_count, len(reservoir))

        remaining = count - coverage_count - reservoir_count
        if remaining > 0:
            extra_coverage = min(remaining, len(coverage) - coverage_count)
            coverage_count += extra_coverage
            remaining -= extra_coverage
        if remaining > 0:
            reservoir_count += min(
                remaining, len(reservoir) - reservoir_count
            )

        coverage_indices = self._draw_indices(
            rng, len(coverage), coverage_count
        )
        reservoir_indices = self._draw_indices(
            rng, len(reservoir), reservoir_count
        )
        return [coverage[index] for index in coverage_indices] + [
            reservoir[index] for index in reservoir_indices
        ]

    def _items_to_batch(
        self, items: list[tuple[torch.Tensor, int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not items:
            return None
        x = torch.stack([item[0] for item in items]).to(self.device)
        y = torch.tensor(
            [item[1] for item in items],
            dtype=torch.long,
            device=self.device,
        )
        task_ids = torch.tensor(
            [item[2] for item in items],
            dtype=torch.long,
            device=self.device,
        )
        return x, y, task_ids

    def _sample_memory(self):
        items = self._mixture_items(
            self.coverage_fraction,
            min(self.batch_size_mem, len(self._memory_x)),
            self._rng,
        )
        batch = self._items_to_batch(items)
        if batch is not None:
            self._replay_examples += len(items)
        return batch

    def _controller_probe(
        self, count: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        # A 50/50 probe evaluates retention over both complementary stores.
        return self._items_to_batch(
            self._mixture_items(0.50, count, self._controller_rng)
        )

    def _before_replay_selection(
        self,
        current_x: torch.Tensor,
        current_y: torch.Tensor,
        current_tid: torch.Tensor,
    ) -> None:
        iteration = int(self.clock.train_iterations) + 1
        if iteration % self.controller_every != 0:
            return
        if len(self._coverage_items()) < 8 or len(self._reservoir_items()) < 8:
            return
        if int(current_y.numel()) < 8:
            return

        current_indices = list(range(int(current_y.numel())))
        self._controller_rng.shuffle(current_indices)
        cut = max(1, len(current_indices) // 2)
        train_indices = current_indices[:cut]
        probe_indices = current_indices[cut:]
        if not probe_indices:
            return

        current_train_x = current_x[train_indices].to(self.device)
        current_train_y = current_y[train_indices].to(self.device)
        current_train_tid = current_tid[train_indices].to(self.device)
        current_probe_x = current_x[probe_indices].to(self.device)
        current_probe_y = current_y[probe_indices].to(self.device)
        current_probe_tid = current_tid[probe_indices].to(self.device)
        memory_probe = self._controller_probe(
            min(32, len(self._memory_x))
        )
        if memory_probe is None:
            return
        memory_probe_x, memory_probe_y, memory_probe_tid = memory_probe

        was_training = self.model.training
        self.model.eval()
        try:
            _, head = get_last_fc_layer(self.model)
            parameters = tuple(
                parameter
                for parameter in head.parameters()
                if parameter.requires_grad
            )
            current_logits = avalanche_forward(
                self.model, current_train_x, current_train_tid
            )
            current_loss = self._ace_current_loss(
                current_logits, current_train_y
            )
            current_gradient = self._gradient_vector(
                current_loss, parameters
            )
            with torch.no_grad():
                current_before = F.cross_entropy(
                    avalanche_forward(
                        self.model, current_probe_x, current_probe_tid
                    ),
                    current_probe_y,
                )
            current_only_model = self._head_step_copy(current_gradient)
            with torch.no_grad():
                current_only_ce = F.cross_entropy(
                    avalanche_forward(
                        current_only_model,
                        current_probe_x,
                        current_probe_tid,
                    ),
                    current_probe_y,
                )
                current_only_memory_ce = F.cross_entropy(
                    avalanche_forward(
                        current_only_model,
                        memory_probe_x,
                        memory_probe_tid,
                    ),
                    memory_probe_y,
                )
            current_only_progress = float(
                (current_before - current_only_ce).cpu()
            )

            rows = []
            candidate_count = min(self.batch_size_mem, len(self._memory_x))
            for fraction in self.mixture_candidates:
                candidate = self._items_to_batch(
                    self._mixture_items(
                        fraction,
                        candidate_count,
                        self._controller_rng,
                    )
                )
                if candidate is None:
                    continue
                replay_x, replay_y, replay_tid = candidate
                replay_logits = avalanche_forward(
                    self.model, replay_x, replay_tid
                )
                replay_gradient = self._gradient_vector(
                    F.cross_entropy(replay_logits, replay_y), parameters
                )
                policy_model = self._head_step_copy(
                    (current_gradient + replay_gradient) / 2.0
                )
                with torch.no_grad():
                    policy_current_ce = F.cross_entropy(
                        avalanche_forward(
                            policy_model,
                            current_probe_x,
                            current_probe_tid,
                        ),
                        current_probe_y,
                    )
                    policy_memory_ce = F.cross_entropy(
                        avalanche_forward(
                            policy_model,
                            memory_probe_x,
                            memory_probe_tid,
                        ),
                        memory_probe_y,
                    )
                progress = float(
                    (current_before - policy_current_ce).cpu()
                )
                memory_gain = float(
                    (current_only_memory_ce - policy_memory_ce).cpu()
                )
                positive_reference = current_only_progress > 0.0
                safe = (
                    progress >= 0.0
                    and memory_gain >= 0.0
                    and (
                        not positive_reference
                        or progress
                        >= self.controller_rho * current_only_progress
                    )
                )
                rows.append(
                    {
                        "coverage_fraction": fraction,
                        "current_progress": progress,
                        "memory_gain": memory_gain,
                        "safe": safe,
                        "unique_replay_classes": len(
                            set(int(value) for value in replay_y.tolist())
                        ),
                    }
                )
                del policy_model

            safe_rows = [row for row in rows if row["safe"]]
            previous = self.coverage_fraction
            if safe_rows:
                selected = max(
                    safe_rows,
                    key=lambda row: (
                        row["memory_gain"],
                        row["current_progress"],
                        -abs(row["coverage_fraction"] - 0.50),
                    ),
                )
                self.coverage_fraction = float(
                    selected["coverage_fraction"]
                )
            self._controller_history.append(
                {
                    "iteration": iteration,
                    "previous_coverage_fraction": previous,
                    "selected_coverage_fraction": self.coverage_fraction,
                    "current_only_progress": current_only_progress,
                    "candidate_rows": rows,
                    "safe_candidate_count": len(safe_rows),
                }
            )
            del current_only_model
        finally:
            self.model.train(was_training)

    def rbcl_summary(self) -> dict:
        report = super().rbcl_summary()
        selections = [
            row["selected_coverage_fraction"]
            for row in self._controller_history
        ]
        report["counterfactual_mixture_controller"] = {
            "enabled": True,
            "every": self.controller_every,
            "rho": self.controller_rho,
            "candidates": list(self.mixture_candidates),
            "event_count": len(self._controller_history),
            "final_coverage_fraction": self.coverage_fraction,
            "selection_counts": dict(Counter(selections)),
            "history": self._controller_history,
        }
        return report


__all__ = ["CausalCounterfactualMixtureERACE"]
