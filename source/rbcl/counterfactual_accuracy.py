"""Non-invasive counterfactual accuracy-correction audit for fixed Hybrid."""

from __future__ import annotations

import copy
import random

import torch
import torch.nn.functional as F

from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import get_last_fc_layer

from .causal_er_ace import CausalERACE


class CausalHybridAccuracyCorrectionAudit(CausalERACE):
    """Audit a head-only plasticity correction without changing training.

    Actual storage, replay sampling, and optimizer updates remain exactly the
    fixed 50/50 Hybrid ER-ACE baseline. At a fixed cadence, independent model
    copies compare the anchor Hybrid head update with the same update followed
    by one current-only classifier-head correction. Copies never write back.
    """

    def __init__(
        self,
        *,
        audit_every: int = 500,
        apply_safe_correction: bool = False,
        relative_gain_budget: float | None = None,
        seed: int = 0,
        **kwargs,
    ) -> None:
        if audit_every <= 0:
            raise ValueError("audit_every must be positive")
        if relative_gain_budget is not None and not (
            0.0 < relative_gain_budget <= 1.0
        ):
            raise ValueError("relative_gain_budget must be in (0, 1]")
        super().__init__(
            seed=seed,
            memory_policy="hybrid",
            **kwargs,
        )
        self.accuracy_audit_every = int(audit_every)
        self.apply_safe_correction = bool(apply_safe_correction)
        self.relative_gain_budget = relative_gain_budget
        self._accuracy_audit_rng = random.Random(int(seed) + 4_000_037)
        self._accuracy_audit_history: list[dict[str, float | bool]] = []
        self._pending_correction = None
        self._applied_correction_count = 0

    def _head_step_from(
        self,
        model,
        gradient: torch.Tensor,
        *,
        step_scale: float = 1.0,
    ):
        candidate = copy.deepcopy(model).to(self.device)
        _, head = get_last_fc_layer(candidate)
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        offset = 0
        with torch.no_grad():
            for parameter in head.parameters():
                if not parameter.requires_grad:
                    continue
                width = parameter.numel()
                parameter.add_(
                    gradient[offset : offset + width].reshape_as(parameter),
                    alpha=-learning_rate * float(step_scale),
                )
                offset += width
        candidate.eval()
        return candidate

    @staticmethod
    def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
        return float((logits.argmax(dim=1) == labels).float().mean().cpu())

    def _memory_batch(self, indices: list[int]):
        x = torch.stack([self._memory_x[index] for index in indices]).to(
            self.device
        )
        y = torch.tensor(
            [self._memory_y[index] for index in indices],
            dtype=torch.long,
            device=self.device,
        )
        task_ids = torch.tensor(
            [self._memory_tid[index] for index in indices],
            dtype=torch.long,
            device=self.device,
        )
        return x, y, task_ids

    def _before_replay_selection(
        self,
        current_x: torch.Tensor,
        current_y: torch.Tensor,
        current_tid: torch.Tensor,
    ) -> None:
        self._pending_correction = None
        iteration = int(self.clock.train_iterations) + 1
        if iteration % self.accuracy_audit_every != 0:
            return
        if len(self._memory_x) < 24 or int(current_y.numel()) < 8:
            return

        current_indices = list(range(int(current_y.numel())))
        self._accuracy_audit_rng.shuffle(current_indices)
        current_cut = max(1, len(current_indices) // 2)
        current_train_indices = current_indices[:current_cut]
        current_probe_indices = current_indices[current_cut:]
        if not current_probe_indices:
            return

        memory_indices = list(range(len(self._memory_x)))
        self._accuracy_audit_rng.shuffle(memory_indices)
        memory_train_count = min(
            self.batch_size_mem, len(memory_indices) - 8
        )
        memory_train_indices = memory_indices[:memory_train_count]
        memory_probe_indices = memory_indices[memory_train_count:]
        if not memory_train_indices or not memory_probe_indices:
            return

        current_train_x = current_x[current_train_indices].to(self.device)
        current_train_y = current_y[current_train_indices].to(self.device)
        current_train_tid = current_tid[current_train_indices].to(self.device)
        current_probe_x = current_x[current_probe_indices].to(self.device)
        current_probe_y = current_y[current_probe_indices].to(self.device)
        current_probe_tid = current_tid[current_probe_indices].to(self.device)
        memory_train_x, memory_train_y, memory_train_tid = self._memory_batch(
            memory_train_indices
        )
        memory_probe_x, memory_probe_y, memory_probe_tid = self._memory_batch(
            memory_probe_indices
        )

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
            memory_logits = avalanche_forward(
                self.model, memory_train_x, memory_train_tid
            )
            anchor_gradient = self._gradient_vector(
                (
                    self._ace_current_loss(current_logits, current_train_y)
                    + F.cross_entropy(memory_logits, memory_train_y)
                )
                / 2.0,
                parameters,
            )
            anchor_model = self._head_step_from(self.model, anchor_gradient)

            _, anchor_head = get_last_fc_layer(anchor_model)
            anchor_parameters = tuple(
                parameter
                for parameter in anchor_head.parameters()
                if parameter.requires_grad
            )
            correction_logits = avalanche_forward(
                anchor_model, current_train_x, current_train_tid
            )
            correction_gradient = self._gradient_vector(
                self._ace_current_loss(
                    correction_logits, current_train_y
                ),
                anchor_parameters,
            )
            corrected_model = self._head_step_from(
                anchor_model, correction_gradient
            )

            with torch.no_grad():
                anchor_current_logits = avalanche_forward(
                    anchor_model, current_probe_x, current_probe_tid
                )
                corrected_current_logits = avalanche_forward(
                    corrected_model, current_probe_x, current_probe_tid
                )
                anchor_memory_logits = avalanche_forward(
                    anchor_model, memory_probe_x, memory_probe_tid
                )
                corrected_memory_logits = avalanche_forward(
                    corrected_model, memory_probe_x, memory_probe_tid
                )
                anchor_current_ce = float(
                    F.cross_entropy(
                        anchor_current_logits, current_probe_y
                    ).cpu()
                )
                corrected_current_ce = float(
                    F.cross_entropy(
                        corrected_current_logits, current_probe_y
                    ).cpu()
                )
                anchor_memory_ce = float(
                    F.cross_entropy(
                        anchor_memory_logits, memory_probe_y
                    ).cpu()
                )
                corrected_memory_ce = float(
                    F.cross_entropy(
                        corrected_memory_logits, memory_probe_y
                    ).cpu()
                )
                anchor_current_accuracy = self._accuracy(
                    anchor_current_logits, current_probe_y
                )
                corrected_current_accuracy = self._accuracy(
                    corrected_current_logits, current_probe_y
                )
                anchor_memory_accuracy = self._accuracy(
                    anchor_memory_logits, memory_probe_y
                )
                corrected_memory_accuracy = self._accuracy(
                    corrected_memory_logits, memory_probe_y
                )

            full_current_ce_gain = anchor_current_ce - corrected_current_ce
            target_current_ce_gain = None
            correction_scale = 1.0
            if self.relative_gain_budget is not None:
                target_current_ce_gain = (
                    self.relative_gain_budget * anchor_current_ce
                )
                if full_current_ce_gain <= 0.0:
                    correction_scale = 0.0
                else:
                    correction_scale = min(
                        1.0,
                        target_current_ce_gain / full_current_ce_gain,
                    )

                if correction_scale < 1.0:
                    del corrected_model
                    corrected_model = self._head_step_from(
                        anchor_model,
                        correction_gradient,
                        step_scale=correction_scale,
                    )
                    with torch.no_grad():
                        corrected_current_logits = avalanche_forward(
                            corrected_model,
                            current_probe_x,
                            current_probe_tid,
                        )
                        corrected_memory_logits = avalanche_forward(
                            corrected_model,
                            memory_probe_x,
                            memory_probe_tid,
                        )
                        corrected_current_ce = float(
                            F.cross_entropy(
                                corrected_current_logits,
                                current_probe_y,
                            ).cpu()
                        )
                        corrected_memory_ce = float(
                            F.cross_entropy(
                                corrected_memory_logits,
                                memory_probe_y,
                            ).cpu()
                        )
                        corrected_current_accuracy = self._accuracy(
                            corrected_current_logits,
                            current_probe_y,
                        )
                        corrected_memory_accuracy = self._accuracy(
                            corrected_memory_logits,
                            memory_probe_y,
                        )

            current_ce_gain = anchor_current_ce - corrected_current_ce
            memory_ce_damage = corrected_memory_ce - anchor_memory_ce
            current_accuracy_gain = (
                corrected_current_accuracy - anchor_current_accuracy
            )
            memory_accuracy_change = (
                corrected_memory_accuracy - anchor_memory_accuracy
            )
            self._accuracy_audit_history.append(
                {
                    "iteration": float(iteration),
                    "anchor_current_ce": anchor_current_ce,
                    "full_current_ce_gain": full_current_ce_gain,
                    "target_current_ce_gain": target_current_ce_gain,
                    "correction_scale": correction_scale,
                    "current_ce_gain": current_ce_gain,
                    "memory_ce_damage": memory_ce_damage,
                    "current_accuracy_gain": current_accuracy_gain,
                    "memory_accuracy_change": memory_accuracy_change,
                    "current_ce_improves": current_ce_gain > 0.0,
                    "memory_ce_not_worse": memory_ce_damage <= 0.0,
                    "safe_correction": (
                        current_ce_gain > 0.0 and memory_ce_damage <= 0.0
                    ),
                }
            )
            if (
                self.apply_safe_correction
                and current_ce_gain > 0.0
                and memory_ce_damage <= 0.0
            ):
                self._pending_correction = (
                    current_train_x.detach(),
                    current_train_y.detach(),
                    current_train_tid.detach(),
                    correction_scale,
                )
            del anchor_model, corrected_model
        finally:
            self.model.train(was_training)

    def _after_anchor_update(
        self,
        current_x: torch.Tensor,
        current_y: torch.Tensor,
        current_tid: torch.Tensor,
    ) -> None:
        if not self.apply_safe_correction or self._pending_correction is None:
            return
        train_x, train_y, train_tid, correction_scale = (
            self._pending_correction
        )
        was_training = self.model.training
        self.model.eval()
        try:
            _, head = get_last_fc_layer(self.model)
            parameters = tuple(
                parameter
                for parameter in head.parameters()
                if parameter.requires_grad
            )
            logits = avalanche_forward(self.model, train_x, train_tid)
            gradient = self._gradient_vector(
                self._ace_current_loss(logits, train_y), parameters
            )
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            offset = 0
            with torch.no_grad():
                for parameter in parameters:
                    width = parameter.numel()
                    parameter.add_(
                        gradient[offset : offset + width].reshape_as(parameter),
                        alpha=-learning_rate * float(correction_scale),
                    )
                    offset += width
            self._applied_correction_count += 1
            self._pending_correction = None
        finally:
            self.model.train(was_training)

    def rbcl_summary(self) -> dict:
        report = super().rbcl_summary()
        history = self._accuracy_audit_history

        def rate(key: str) -> float:
            if not history:
                return 0.0
            return sum(bool(row[key]) for row in history) / len(history)

        report["counterfactual_accuracy_correction_audit"] = {
            "enabled": True,
            "non_invasive": not self.apply_safe_correction,
            "actual_training_is_fixed_hybrid": (
                not self.apply_safe_correction
            ),
            "apply_safe_correction": self.apply_safe_correction,
            "relative_gain_budget": self.relative_gain_budget,
            "gain_budgeted": self.relative_gain_budget is not None,
            "applied_correction_count": self._applied_correction_count,
            "every": self.accuracy_audit_every,
            "event_count": len(history),
            "current_ce_improvement_rate": rate("current_ce_improves"),
            "memory_ce_safe_rate": rate("memory_ce_not_worse"),
            "safe_correction_rate": rate("safe_correction"),
            "safe_correction_count": sum(
                bool(row["safe_correction"]) for row in history
            ),
            "history": history,
        }
        return report


__all__ = ["CausalHybridAccuracyCorrectionAudit"]
