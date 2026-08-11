"""Current-risk constrained child of the frozen Layer-3 D strategy."""

import torch

from .semantic_representative import (
    PrequentialRiskBudgetedDualHeadArbitrationERACE,
)


class CurrentRiskGuardedPrequentialArbitrationERACE(
    PrequentialRiskBudgetedDualHeadArbitrationERACE
):
    """Project D's prequential alpha onto a zero-slack current-risk budget."""

    _CURRENT_RISK_BUDGET = 0.0
    _CURRENT_GUARD_BISECTION_STEPS = 8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_guard_checks = 0
        self._current_guard_activations = 0
        self._current_guard_raw_alpha_sum = 0.0
        self._current_guard_alpha_reduction_sum = 0.0
        self._current_guard_raw_violation_sum = 0.0
        self._current_guard_raw_violation_max = 0.0
        self._current_guard_guarded_violation_max = 0.0
        self._current_guard_replay_improvement_sum = 0.0

    @classmethod
    def _project_alpha_to_current_risk_budget(
        cls,
        raw_alpha,
        current_training,
        current_calibration,
        current_labels,
        label_smoothing,
    ):
        """Return the largest alpha no riskier than the current-head baseline."""
        raw_alpha = float(min(1.0, max(0.0, raw_alpha)))
        current_base = cls._class_balanced_smoothed_loss(
            current_training, current_labels, label_smoothing
        )
        current_delta = current_calibration - current_training

        def current_loss(alpha):
            return cls._class_balanced_smoothed_loss(
                current_training + alpha * current_delta,
                current_labels,
                label_smoothing,
            )

        raw_current = current_loss(raw_alpha)
        scale = torch.maximum(current_base.abs(), torch.ones_like(current_base))
        numerical_tolerance = 16.0 * torch.finfo(current_base.dtype).eps * scale
        limit = current_base + float(cls._CURRENT_RISK_BUDGET) + numerical_tolerance
        finite = torch.stack([current_base, raw_current, limit])
        if not bool(torch.isfinite(finite).all()):
            raise FloatingPointError("nonfinite current-risk guard loss")

        guarded_alpha = raw_alpha
        if bool(raw_current > limit) and raw_alpha > 0.0:
            low = torch.zeros_like(current_base)
            high = torch.full_like(current_base, raw_alpha)
            for _ in range(cls._CURRENT_GUARD_BISECTION_STEPS):
                middle = (low + high) / 2.0
                feasible = current_loss(middle) <= limit
                low = torch.where(feasible, middle, low)
                high = torch.where(feasible, high, middle)
            guarded_alpha = float(low.detach().cpu())

        guarded_current = current_loss(guarded_alpha)
        audit = {
            "raw_alpha": raw_alpha,
            "guarded_alpha": guarded_alpha,
            "current_base": float(current_base.detach().cpu()),
            "raw_current": float(raw_current.detach().cpu()),
            "guarded_current": float(guarded_current.detach().cpu()),
            "numerical_tolerance": float(numerical_tolerance.detach().cpu()),
        }
        return guarded_alpha, audit

    def _optimal_arbitration_alpha(
        self,
        current_training_logits,
        current_calibration_logits,
        current_labels,
        replay_training_logits,
        replay_calibration_logits,
        replay_labels,
        label_smoothing,
    ):
        raw_alpha = (
            PrequentialRiskBudgetedDualHeadArbitrationERACE.
            _optimal_arbitration_alpha(
                current_training_logits,
                current_calibration_logits,
                current_labels,
                replay_training_logits,
                replay_calibration_logits,
                replay_labels,
                label_smoothing,
            )
        )
        guarded_alpha, audit = self._project_alpha_to_current_risk_budget(
            raw_alpha,
            current_training_logits,
            current_calibration_logits,
            current_labels,
            label_smoothing,
        )
        replay_base = self._class_balanced_smoothed_loss(
            replay_training_logits, replay_labels, label_smoothing
        )
        replay_guarded = self._class_balanced_smoothed_loss(
            torch.lerp(
                replay_training_logits,
                replay_calibration_logits,
                guarded_alpha,
            ),
            replay_labels,
            label_smoothing,
        )
        raw_violation = max(0.0, audit["raw_current"] - audit["current_base"])
        guarded_violation = max(
            0.0, audit["guarded_current"] - audit["current_base"]
        )
        reduction = max(0.0, raw_alpha - guarded_alpha)
        self._current_guard_checks += 1
        self._current_guard_activations += int(reduction > 1e-12)
        self._current_guard_raw_alpha_sum += raw_alpha
        self._current_guard_alpha_reduction_sum += reduction
        self._current_guard_raw_violation_sum += raw_violation
        self._current_guard_raw_violation_max = max(
            self._current_guard_raw_violation_max, raw_violation
        )
        self._current_guard_guarded_violation_max = max(
            self._current_guard_guarded_violation_max, guarded_violation
        )
        self._current_guard_replay_improvement_sum += float(
            (replay_base - replay_guarded).detach().cpu()
        )
        return guarded_alpha

    def rbcl_summary(self):
        result = super().rbcl_summary()
        checks = max(1, self._current_guard_checks)
        arbitration = result["risk_budgeted_head_arbitration"]
        arbitration.update(
            {
                "function": "current-risk-guarded prequential deployment arbitration",
                "alpha_source": (
                    "online cumulative mean of frozen-D batchwise optima projected "
                    "onto a zero-slack pre-update current-risk budget"
                ),
                "current_risk_guard_enabled": self.calibration_lr_scale > 0.0,
            }
        )
        result["current_risk_guarded_prequential_arbitration"] = {
            "enabled": self.calibration_lr_scale > 0.0,
            "frozen_parent": "persistent_srrd_prequential_arbitration_1",
            "current_risk_budget": self._CURRENT_RISK_BUDGET,
            "current_risk_budget_is_performance_tuned": False,
            "projection_rule": "largest feasible alpha on [0, frozen-D alpha]",
            "projection_bisection_steps": self._CURRENT_GUARD_BISECTION_STEPS,
            "guard_checks": self._current_guard_checks,
            "guard_activations": self._current_guard_activations,
            "activation_fraction": self._current_guard_activations / checks,
            "mean_raw_alpha": self._current_guard_raw_alpha_sum / checks,
            "mean_alpha_reduction": self._current_guard_alpha_reduction_sum / checks,
            "mean_raw_current_risk_violation": (
                self._current_guard_raw_violation_sum / checks
            ),
            "maximum_raw_current_risk_violation": self._current_guard_raw_violation_max,
            "maximum_guarded_current_risk_violation": (
                self._current_guard_guarded_violation_max
            ),
            "mean_guarded_replay_risk_improvement": (
                self._current_guard_replay_improvement_sum / checks
            ),
            "uses_pre_update_arrived_current_evidence_only": True,
            "uses_validation_pss_future_data_or_task_boundary": False,
            "additional_replay_draws": 0,
            "additional_backbone_forwards": 0,
            "additional_head_forwards_beyond_frozen_D": 0,
            "changes_training_model_memory_or_replay_path": False,
        }
        return result
