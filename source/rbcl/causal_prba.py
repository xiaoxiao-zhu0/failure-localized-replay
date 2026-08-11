"""Prequential risk-budgeted deployment-head arbitration for Causal ER-ACE."""

from __future__ import annotations

import copy
import hashlib
import time

import torch
import torch.nn.functional as F

from avalanche.training.utils import get_last_fc_layer

from .calibration_audit import (
    empty_dual_head_audit,
    summarize_dual_head_audit,
    update_dual_head_audit,
)
from .causal_er_ace import CausalERACE


class CausalReplayFeatureDualHeadCalibrationERACE(CausalERACE):
    """Add a replay-trained deployment head to the plain Causal ER-ACE parent.

    The deployment head reuses classifier inputs already produced by the
    parent current/replay forwards. It never changes the parent objective,
    replay draw, memory update, model parameters, or optimizer state.
    """

    def __init__(
        self,
        *,
        calibration_lr_scale: float = 1.0,
        calibration_label_smoothing: float = 0.1,
        calibration_replay_detailed_audit: bool = True,
        **kwargs,
    ):
        if calibration_lr_scale < 0.0:
            raise ValueError("calibration_lr_scale must be nonnegative")
        if not 0.0 <= calibration_label_smoothing < 1.0:
            raise ValueError("calibration_label_smoothing must be in [0, 1)")
        super().__init__(**kwargs)
        self.calibration_lr_scale = float(calibration_lr_scale)
        self.calibration_label_smoothing = float(calibration_label_smoothing)
        self.calibration_replay_detailed_audit = bool(
            calibration_replay_detailed_audit
        )
        _, training_head = get_last_fc_layer(self.model)
        self._calibration_head = copy.deepcopy(training_head).to(self.device)
        self._calibration_head.eval()
        self._calibration_capture_enabled = False
        self._calibration_captured_features: list[torch.Tensor] = []
        self._calibration_pending_batch = None
        self._calibration_replay_calls = 0
        self._calibration_updates = 0
        self._calibration_samples = 0
        self._calibration_class_observations = 0
        self._calibration_nonfinite_skips = 0
        self._calibration_loss_sum = 0.0
        self._calibration_min_loss = None
        self._calibration_max_loss = 0.0
        self._calibration_head_update_host_seconds = 0.0
        self._calibration_replay_head_audit = empty_dual_head_audit()
        self._calibration_train_phase_index = -1
        self._calibration_class_phase: dict[int, int] = {}
        self._calibration_eval_current = None
        self._calibration_eval_history: list[dict[str, object]] = []
        self._calibration_eval_forward_calls = 0

        def capture_head_input(_module, inputs):
            if self._calibration_capture_enabled:
                self._calibration_captured_features.append(inputs[0].detach())

        self._calibration_feature_hook = (
            training_head.register_forward_pre_hook(capture_head_input)
            if self.calibration_lr_scale > 0.0
            else None
        )

    @staticmethod
    def _class_balanced_smoothed_loss(logits, labels, label_smoothing):
        losses = F.cross_entropy(
            logits,
            labels,
            reduction="none",
            label_smoothing=float(label_smoothing),
        )
        classes = torch.unique(labels, sorted=True)
        if int(classes.numel()) == 0:
            raise ValueError("calibration labels must be nonempty")
        return torch.stack(
            [losses[labels == label].mean() for label in classes]
        ).mean()

    @staticmethod
    def _head_learning_rate(optimizer, parameters):
        parameter_ids = {id(parameter) for parameter in parameters}
        learning_rates = []
        for group in optimizer.param_groups:
            if any(id(parameter) in parameter_ids for parameter in group["params"]):
                learning_rates.append(float(group["lr"]))
        if not learning_rates:
            raise RuntimeError("classifier head parameters are absent from optimizer")
        return min(learning_rates)

    @staticmethod
    def _module_state_hash(module):
        digest = hashlib.sha256()
        for name, value in sorted(module.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _before_training_exp(self, **kwargs):
        if self.calibration_lr_scale > 0.0:
            self._calibration_train_phase_index += 1
        super()._before_training_exp(**kwargs)

    def _before_forward(self, **kwargs):
        if self.calibration_lr_scale <= 0.0:
            return super()._before_forward(**kwargs)
        if self.is_training:
            for label in self.mb_y.detach().unique(sorted=True).cpu().tolist():
                self._calibration_class_phase.setdefault(
                    int(label), self._calibration_train_phase_index
                )
        self._calibration_captured_features = []
        self._calibration_capture_enabled = self.is_training
        super()._before_forward(**kwargs)

    def _before_eval(self, **kwargs):
        self._calibration_eval_current = (
            empty_dual_head_audit()
            if self.calibration_lr_scale > 0.0 and self._calibration_updates > 0
            else None
        )
        super()._before_eval(**kwargs)

    def _deployment_audit_state(self):
        return {
            "calibration_updates": self._calibration_updates,
            "calibration_samples": self._calibration_samples,
        }

    def _after_eval(self, **kwargs):
        if self._calibration_eval_current is not None:
            observed = sorted(int(label) for label in self._observed_classes)
            phases = {
                label: self._calibration_class_phase.get(label, -1)
                for label in observed
            }
            newest = max(phases.values(), default=-1)
            self._calibration_eval_history.append(
                {
                    "evaluation_index": len(self._calibration_eval_history),
                    "training_phase_index": self._calibration_train_phase_index,
                    "observed_classes": observed,
                    "old_classes": [
                        label for label in observed if phases[label] < newest
                    ],
                    "new_classes": [
                        label for label in observed if phases[label] == newest
                    ],
                    "deployment_state": self._deployment_audit_state(),
                    "dual_head_metrics": summarize_dual_head_audit(
                        self._calibration_eval_current
                    ),
                }
            )
        self._calibration_eval_current = None
        super()._after_eval(**kwargs)

    def _record_calibration_evaluation(self, training_logits, deployment_logits):
        if self._calibration_eval_current is None:
            return
        labels = self.mb_y.detach()
        observed = sorted(int(label) for label in self._observed_classes)
        if not observed:
            return
        observed_tensor = torch.tensor(
            observed, dtype=torch.long, device=labels.device
        )
        mask = labels[:, None].eq(observed_tensor[None, :]).any(dim=1)
        if not bool(mask.any()):
            return
        phases = {
            label: self._calibration_class_phase.get(label, -1)
            for label in observed
        }
        newest = max(phases.values(), default=-1)
        update_dual_head_audit(
            self._calibration_eval_current,
            training_logits[mask].detach(),
            deployment_logits[mask].detach(),
            labels[mask],
            old_classes=[label for label in observed if phases[label] < newest],
            new_classes=[label for label in observed if phases[label] == newest],
        )
        self._calibration_eval_forward_calls += 1

    def forward(self):
        if (
            self.is_training
            or self.calibration_lr_scale <= 0.0
            or self._calibration_updates <= 0
        ):
            return super().forward()
        _, training_head = get_last_fc_layer(self.model)
        captured = {}

        def capture(_module, inputs):
            captured["features"] = inputs[0].detach()

        handle = training_head.register_forward_pre_hook(capture)
        try:
            training_logits = super().forward()
        finally:
            handle.remove()
        if "features" not in captured:
            raise RuntimeError("classifier head input was not observed during evaluation")
        self._calibration_head.eval()
        calibration_logits = self._calibration_head(captured["features"])
        deployment_logits = self._deployment_logits(
            training_logits, calibration_logits
        )
        self._record_calibration_evaluation(training_logits, deployment_logits)
        return deployment_logits

    def _deployment_logits(self, training_logits, calibration_logits):
        return calibration_logits

    def _add_auxiliary_loss(
        self,
        current_x,
        current_y,
        current_tid,
        replay,
        replay_output=None,
    ):
        super()._add_auxiliary_loss(
            current_x, current_y, current_tid, replay, replay_output
        )
        self._calibration_pending_batch = None
        if self.calibration_lr_scale <= 0.0 or replay is None:
            self._calibration_capture_enabled = False
            self._calibration_captured_features = []
            return
        replay_count = int(replay[1].numel())
        captured = list(self._calibration_captured_features)
        candidates = [
            features for features in captured if int(features.shape[0]) == replay_count
        ]
        self._calibration_capture_enabled = False
        self._calibration_captured_features = []
        if not candidates:
            raise RuntimeError("replay classifier features were not captured")
        if not captured:
            raise RuntimeError("current classifier features were not captured")
        self._calibration_pending_batch = (
            captured[0].detach(),
            current_y.detach(),
            self.mb_output.detach(),
            candidates[-1].detach(),
            replay[1].detach(),
            replay_output.detach(),
        )

    def _after_calibration_head_update(
        self,
        current_features,
        current_labels,
        current_training_logits,
        replay_features,
        replay_labels,
        replay_training_logits,
    ):
        """Extension point for deployment-head arbitration."""

    def _after_anchor_update(self, current_x, current_y, current_tid):
        super()._after_anchor_update(current_x, current_y, current_tid)
        pending = self._calibration_pending_batch
        self._calibration_pending_batch = None
        if self.calibration_lr_scale <= 0.0 or pending is None:
            return
        (
            current_features,
            current_labels,
            current_training_logits,
            replay_features,
            replay_labels,
            replay_training_logits,
        ) = pending
        self._calibration_replay_calls += 1
        self._calibration_head.train()
        parameters = tuple(
            parameter
            for parameter in self._calibration_head.parameters()
            if parameter.requires_grad
        )
        host_started = time.perf_counter()
        logits = self._calibration_head(replay_features)
        loss = self._class_balanced_smoothed_loss(
            logits, replay_labels, self.calibration_label_smoothing
        )
        if not bool(torch.isfinite(loss)):
            self._calibration_head_update_host_seconds += (
                time.perf_counter() - host_started
            )
            self._calibration_nonfinite_skips += 1
            self._calibration_head.eval()
            return
        gradients = torch.autograd.grad(loss, parameters)
        _, training_head = get_last_fc_layer(self.model)
        training_parameters = tuple(
            parameter for parameter in training_head.parameters() if parameter.requires_grad
        )
        step = self.calibration_lr_scale * self._head_learning_rate(
            self.optimizer, training_parameters
        )
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients):
                parameter.add_(gradient, alpha=-step)
        self._calibration_head_update_host_seconds += time.perf_counter() - host_started
        if self.calibration_replay_detailed_audit:
            update_dual_head_audit(
                self._calibration_replay_head_audit,
                replay_training_logits,
                logits.detach(),
                replay_labels,
                old_classes=torch.unique(replay_labels, sorted=True).cpu().tolist(),
            )
        value = float(loss.detach().cpu())
        self._calibration_updates += 1
        self._calibration_samples += int(replay_labels.numel())
        self._calibration_class_observations += int(
            torch.unique(replay_labels).numel()
        )
        self._calibration_loss_sum += value
        self._calibration_min_loss = (
            value
            if self._calibration_min_loss is None
            else min(self._calibration_min_loss, value)
        )
        self._calibration_max_loss = max(self._calibration_max_loss, value)
        self._calibration_head.eval()
        self._after_calibration_head_update(
            current_features,
            current_labels,
            current_training_logits,
            replay_features,
            replay_labels,
            replay_training_logits,
        )

    def rbcl_summary(self):
        report = super().rbcl_summary()
        updates = max(1, self._calibration_updates)
        training_head = get_last_fc_layer(self.model)[1]
        report["replay_feature_dual_head_calibration"] = {
            "enabled": self.calibration_lr_scale > 0.0,
            "parent_method": "causal_er_ace",
            "function": "memory-only deployment-head calibration with parent replay-feature reuse",
            "calibration_lr_scale": self.calibration_lr_scale,
            "label_smoothing": self.calibration_label_smoothing,
            "replay_calls": self._calibration_replay_calls,
            "calibration_updates": self._calibration_updates,
            "calibration_samples": self._calibration_samples,
            "calibration_class_observations": self._calibration_class_observations,
            "mean_classes_per_update": self._calibration_class_observations / updates,
            "mean_calibration_loss": self._calibration_loss_sum / updates,
            "minimum_calibration_loss": self._calibration_min_loss,
            "maximum_calibration_loss": self._calibration_max_loss,
            "head_update_host_dispatch_seconds": self._calibration_head_update_host_seconds,
            "replay_head_comparison": summarize_dual_head_audit(
                self._calibration_replay_head_audit
            ),
            "replay_head_comparison_detailed": self.calibration_replay_detailed_audit,
            "evaluation_bias_audit": {
                "enabled": self.calibration_lr_scale > 0.0,
                "evaluation_forward_calls": self._calibration_eval_forward_calls,
                "history": self._calibration_eval_history,
                "uses_eval_labels_for_audit_only": True,
                "future_unobserved_target_classes_are_excluded": True,
            },
            "nonfinite_skips": self._calibration_nonfinite_skips,
            "extra_head_parameters": sum(
                int(parameter.numel())
                for parameter in self._calibration_head.parameters()
            ),
            "training_head_hash": self._module_state_hash(training_head),
            "deployment_head_hash": self._module_state_hash(
                self._calibration_head
            ),
            "deployment_uses_calibration_head": (
                self.calibration_lr_scale > 0.0 and self._calibration_updates > 0
            ),
            "calibration_uses_replay_only": True,
            "calibration_loss_is_class_balanced_over_present_classes": True,
            "reuses_parent_replay_features": True,
            "additional_replay_draws": 0,
            "additional_backbone_forwards": 0,
            "additional_training_calibration_head_forwards": self._calibration_replay_calls,
            "additional_evaluation_calibration_head_forwards": self._calibration_eval_forward_calls,
            "training_feature_hook_registered": self._calibration_feature_hook is not None,
            "disabled_path_registers_no_training_feature_hook": (
                self.calibration_lr_scale > 0.0
                or self._calibration_feature_hook is None
            ),
            "main_model_parameters_are_not_modified_by_calibration": True,
            "main_optimizer_state_is_not_modified_by_calibration": True,
            "memory_and_replay_identities_are_unchanged": True,
            "uses_task_id_boundary_validation_pss_or_future_data": False,
            "zero_lr_scale_is_exact_parent_noop": True,
        }
        return report


class PrequentialRiskBudgetedDualHeadCausalERACE(
    CausalReplayFeatureDualHeadCalibrationERACE
):
    """Select a causal deployment blend before either head sees the update."""

    _ARBITRATION_BISECTION_STEPS = 8
    _ARBITRATION_CURRENT_WEIGHT = 0.5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._arbitration_batches = 0
        self._arbitration_alpha_sum = 0.0
        self._arbitration_alpha = 0.0
        self._arbitration_alpha_min = None
        self._arbitration_alpha_max = None
        self._arbitration_zero_count = 0
        self._arbitration_interior_count = 0
        self._arbitration_one_count = 0
        self._arbitration_current_ce_delta_sum = 0.0
        self._arbitration_current_ce_delta_max = 0.0
        self._arbitration_replay_ce_improvement_sum = 0.0
        self._arbitration_joint_ce_improvement_sum = 0.0
        self._arbitration_host_seconds = 0.0
        self._arbitration_nonfinite_skips = 0

    @staticmethod
    def _class_balanced_weights(labels, dtype):
        classes, inverse, counts = torch.unique(
            labels, sorted=True, return_inverse=True, return_counts=True
        )
        if int(classes.numel()) == 0:
            raise ValueError("arbitration labels must be nonempty")
        counts = counts.to(device=labels.device, dtype=dtype)
        return counts[inverse].reciprocal() / float(classes.numel())

    @classmethod
    def _balanced_smoothed_ce_derivative(
        cls, base_logits, logit_delta, labels, label_smoothing, alpha
    ):
        mixed = base_logits + alpha * logit_delta
        probabilities = F.softmax(mixed, dim=1)
        rows = torch.arange(labels.numel(), device=labels.device)
        target_delta = (1.0 - float(label_smoothing)) * logit_delta[
            rows, labels
        ]
        target_delta += float(label_smoothing) * logit_delta.mean(dim=1)
        per_item = (probabilities * logit_delta).sum(dim=1) - target_delta
        weights = cls._class_balanced_weights(labels, per_item.dtype)
        return (weights * per_item).sum()

    @classmethod
    def _joint_balanced_smoothed_loss(
        cls,
        current_logits,
        current_labels,
        replay_logits,
        replay_labels,
        label_smoothing,
    ):
        return 0.5 * (
            cls._class_balanced_smoothed_loss(
                current_logits, current_labels, label_smoothing
            )
            + cls._class_balanced_smoothed_loss(
                replay_logits, replay_labels, label_smoothing
            )
        )

    @classmethod
    def _optimal_arbitration_alpha(
        cls,
        current_training_logits,
        current_calibration_logits,
        current_labels,
        replay_training_logits,
        replay_calibration_logits,
        replay_labels,
        label_smoothing,
    ):
        current_delta = current_calibration_logits - current_training_logits
        replay_delta = replay_calibration_logits - replay_training_logits

        def derivative(alpha):
            current_weight = float(cls._ARBITRATION_CURRENT_WEIGHT)
            return current_weight * cls._balanced_smoothed_ce_derivative(
                current_training_logits,
                current_delta,
                current_labels,
                label_smoothing,
                alpha,
            ) + (1.0 - current_weight) * cls._balanced_smoothed_ce_derivative(
                replay_training_logits,
                replay_delta,
                replay_labels,
                label_smoothing,
                alpha,
            )

        at_zero = derivative(0.0)
        at_one = derivative(1.0)
        if not bool(torch.isfinite(torch.stack([at_zero, at_one])).all()):
            raise FloatingPointError("nonfinite arbitration derivative")
        low = torch.zeros_like(at_zero)
        high = torch.ones_like(at_zero)
        for _ in range(cls._ARBITRATION_BISECTION_STEPS):
            middle = (low + high) / 2.0
            value = derivative(middle)
            low = torch.where(value <= 0.0, middle, low)
            high = torch.where(value <= 0.0, high, middle)
        interior = (low + high) / 2.0
        alpha = torch.where(
            at_zero >= 0.0,
            torch.zeros_like(interior),
            torch.where(at_one <= 0.0, torch.ones_like(interior), interior),
        )
        if not bool(torch.isfinite(alpha)):
            raise FloatingPointError("nonfinite arbitration derivative")
        return float(alpha.detach().cpu())

    def _record_arbitration_logits(
        self,
        current_training,
        current_calibration,
        current_labels,
        replay_training,
        replay_calibration,
        replay_labels,
        started,
    ):
        try:
            with torch.no_grad():
                alpha = self._optimal_arbitration_alpha(
                    current_training,
                    current_calibration,
                    current_labels,
                    replay_training,
                    replay_calibration,
                    replay_labels,
                    self.calibration_label_smoothing,
                )
                mixed_current = torch.lerp(
                    current_training, current_calibration, alpha
                )
                mixed_replay = torch.lerp(
                    replay_training, replay_calibration, alpha
                )
                current_base = self._class_balanced_smoothed_loss(
                    current_training,
                    current_labels,
                    self.calibration_label_smoothing,
                )
                current_mixed = self._class_balanced_smoothed_loss(
                    mixed_current,
                    current_labels,
                    self.calibration_label_smoothing,
                )
                replay_base = self._class_balanced_smoothed_loss(
                    replay_training,
                    replay_labels,
                    self.calibration_label_smoothing,
                )
                replay_mixed = self._class_balanced_smoothed_loss(
                    mixed_replay,
                    replay_labels,
                    self.calibration_label_smoothing,
                )
                values = torch.stack(
                    [current_base, current_mixed, replay_base, replay_mixed]
                )
                if not bool(torch.isfinite(values).all()):
                    raise FloatingPointError("nonfinite arbitration loss")
                current_delta, replay_improvement, joint_improvement = torch.stack(
                    [
                        current_mixed - current_base,
                        replay_base - replay_mixed,
                        0.5
                        * (
                            current_base
                            + replay_base
                            - current_mixed
                            - replay_mixed
                        ),
                    ]
                ).detach().cpu().tolist()
        except FloatingPointError:
            self._arbitration_nonfinite_skips += 1
            self._arbitration_host_seconds += time.perf_counter() - started
            return

        self._arbitration_batches += 1
        self._arbitration_alpha_sum += alpha
        self._arbitration_alpha = (
            self._arbitration_alpha_sum / self._arbitration_batches
        )
        self._arbitration_alpha_min = (
            alpha
            if self._arbitration_alpha_min is None
            else min(self._arbitration_alpha_min, alpha)
        )
        self._arbitration_alpha_max = (
            alpha
            if self._arbitration_alpha_max is None
            else max(self._arbitration_alpha_max, alpha)
        )
        tolerance = 2.0 ** (-self._ARBITRATION_BISECTION_STEPS)
        if alpha <= tolerance:
            self._arbitration_zero_count += 1
        elif alpha >= 1.0 - tolerance:
            self._arbitration_one_count += 1
        else:
            self._arbitration_interior_count += 1
        self._arbitration_current_ce_delta_sum += current_delta
        self._arbitration_current_ce_delta_max = max(
            self._arbitration_current_ce_delta_max, current_delta
        )
        self._arbitration_replay_ce_improvement_sum += replay_improvement
        self._arbitration_joint_ce_improvement_sum += joint_improvement
        self._arbitration_host_seconds += time.perf_counter() - started

    def _add_auxiliary_loss(
        self,
        current_x,
        current_y,
        current_tid,
        replay,
        replay_output=None,
    ):
        super()._add_auxiliary_loss(
            current_x, current_y, current_tid, replay, replay_output
        )
        pending = self._calibration_pending_batch
        if self.calibration_lr_scale <= 0.0 or pending is None:
            return
        (
            current_features,
            current_labels,
            current_training_logits,
            replay_features,
            replay_labels,
            replay_training_logits,
        ) = pending
        started = time.perf_counter()
        self._calibration_head.eval()
        with torch.no_grad():
            current_calibration = self._calibration_head(current_features)
            replay_calibration = self._calibration_head(replay_features)
        self._record_arbitration_logits(
            current_training_logits,
            current_calibration,
            current_labels,
            replay_training_logits,
            replay_calibration,
            replay_labels,
            started,
        )

    def _after_calibration_head_update(
        self,
        current_features,
        current_labels,
        current_training_logits,
        replay_features,
        replay_labels,
        replay_training_logits,
    ):
        # The coefficient is deliberately recorded before this update.
        return

    def _deployment_logits(self, training_logits, calibration_logits):
        return torch.lerp(
            training_logits,
            calibration_logits,
            float(self._arbitration_alpha),
        )

    def _deployment_audit_state(self):
        batches = max(1, self._arbitration_batches)
        state = super()._deployment_audit_state()
        state.update(
            {
                "arbitration_batches": self._arbitration_batches,
                "deployment_alpha": self._arbitration_alpha,
                "minimum_batch_alpha": self._arbitration_alpha_min,
                "maximum_batch_alpha": self._arbitration_alpha_max,
                "mean_current_ce_delta": self._arbitration_current_ce_delta_sum
                / batches,
                "mean_replay_ce_improvement": self._arbitration_replay_ce_improvement_sum
                / batches,
                "mean_joint_ce_improvement": self._arbitration_joint_ce_improvement_sum
                / batches,
            }
        )
        return state

    def rbcl_summary(self):
        report = super().rbcl_summary()
        batches = max(1, self._arbitration_batches)
        enabled = self.calibration_lr_scale > 0.0
        calibration = report["replay_feature_dual_head_calibration"]
        calibration["deployment_uses_calibration_head"] = (
            enabled
            and self._calibration_updates > 0
            and self._arbitration_alpha > 0.0
        )
        calibration["deployment_uses_risk_budgeted_blend"] = (
            enabled and self._arbitration_batches > 0
        )
        calibration["additional_training_arbitration_head_forwards"] = (
            2 * self._arbitration_batches
        )
        report["risk_budgeted_head_arbitration"] = {
            "enabled": enabled,
            "parent_method": "causal_er_ace",
            "function": "prequential causal risk-budgeted arbitration between training and calibration heads",
            "deployment_rule": "z_train + alpha * (z_calibration - z_train)",
            "batch_objective": "equal-weight current/replay class-balanced smoothed cross entropy",
            "alpha_source": "online cumulative mean of pre-update batchwise convex optima",
            "fixed_or_tuned_alpha": False,
            "bisection_steps_for_numerical_solution": self._ARBITRATION_BISECTION_STEPS,
            "numerical_alpha_resolution": 2.0
            ** (-self._ARBITRATION_BISECTION_STEPS),
            "arbitration_batches": self._arbitration_batches,
            "deployment_alpha": self._arbitration_alpha,
            "minimum_batch_alpha": self._arbitration_alpha_min,
            "maximum_batch_alpha": self._arbitration_alpha_max,
            "zero_alpha_batches": self._arbitration_zero_count,
            "interior_alpha_batches": self._arbitration_interior_count,
            "one_alpha_batches": self._arbitration_one_count,
            "mean_current_ce_delta": self._arbitration_current_ce_delta_sum
            / batches,
            "maximum_current_ce_delta": self._arbitration_current_ce_delta_max,
            "mean_replay_ce_improvement": self._arbitration_replay_ce_improvement_sum
            / batches,
            "mean_joint_ce_improvement": self._arbitration_joint_ce_improvement_sum
            / batches,
            "host_dispatch_seconds": self._arbitration_host_seconds,
            "mean_host_dispatch_seconds": self._arbitration_host_seconds / batches,
            "nonfinite_skips": self._arbitration_nonfinite_skips,
            "prequential_test_then_train": True,
            "arbitration_is_recorded_before_main_optimizer_step": True,
            "arbitration_is_recorded_before_current_calibration_update": True,
            "same_update_replay_fit_and_budget_evidence_are_separated": True,
            "post_update_replay_reuse_leakage": False,
            "uses_arrived_current_and_replay_batches_only": True,
            "uses_validation_pss_future_data_or_task_boundary": False,
            "additional_replay_draws": 0,
            "additional_backbone_forwards": 0,
            "additional_head_only_forwards": 2 * self._arbitration_batches,
            "disabled_path_is_exact_causal_er_ace_noop": True,
        }
        return report
