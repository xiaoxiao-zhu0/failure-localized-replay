"""Uniform and risk-budgeted loss allocation for continual learning.

This file is the core custom implementation for the paper:
- Uniform Budget: same teacher/distillation/compute path, equal weights.
- Risk Budget: same path, with configurable product or C-gated-I fusion.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.utils import get_last_fc_layer


class RiskBudgetingPlugin(SupervisedPlugin):
    """Risk-guided or uniform budget allocation as an Avalanche plugin.

    Paper mapping:
    - Section "Risk definition": virtual-update replay loss estimates instability.
    - Section "Budget allocation": risk or uniform weights rewrite the loss.
    - Section "Fairness": both modes can execute the same teacher forward path.
    """

    def __init__(
        self,
        *,
        budget_mode: str = "risk",
        allocation: str = "soft",
        budget_ratio: float = 1.0,
        temperature: float = 1.0,
        distill_lambda: float = 0.5,
        distill_temperature: float = 2.0,
        disagreement_lambda: float = 1.0,
        instability_mode: str = "virtual_replay_loss",
        fusion_mode: str = "c_gated",
        instability_lambda: float = 0.25,
        prior_momentum: float = 0.9,
        consequence_mode: str = "loss_ema",
        allocation_scope: str = "global",
        gradient_audit: bool = False,
        gradient_audit_every: int = 100,
        counterfactual_audit: bool = False,
        counterfactual_audit_every: int = 200,
        counterfactual_probe_size: int = 64,
        counterfactual_alpha_audit: bool = False,
        counterfactual_alpha_values: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        counterfactual_rho: float = 0.95,
        plasticity_budget_control: bool = False,
        plasticity_budget_every: int = 500,
        replay_geometry_audit: bool = False,
        replay_geometry_audit_every: int = 500,
        conflict_residual_audit: bool = False,
        conflict_residual_audit_every: int = 500,
        slow_fast_state_audit: bool = False,
        slow_fast_state_audit_every: int = 500,
        slow_fast_ema_decay: float = 0.99,
        boundary_debt_audit: bool = False,
        boundary_debt_probe_size: int = 256,
        boundary_debt_replay: bool = False,
        boundary_debt_temperature: float = 0.25,
        boundary_debt_mix: float = 0.5,
        boundary_repair: bool = False,
        boundary_repair_pairs: int = 2,
        boundary_repair_samples_per_class: int = 32,
        boundary_repair_margin: float = 0.5,
        boundary_repair_lambda: float = 0.5,
        boundary_repair_current_ce_tolerance: float = 0.01,
        boundary_feature_repair: bool = False,
        use_instability: bool = True,
        use_consequence: bool = True,
        use_uncertainty: bool = True,
        use_disagreement: bool = True,
        use_prior: bool = True,
        high_low_fraction: float = 0.25,
        fair_compute: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        if budget_mode not in {"risk", "uniform", "none"}:
            raise ValueError("budget_mode must be one of: risk, uniform, none")
        if allocation not in {"soft", "topk"}:
            raise ValueError("allocation must be one of: soft, topk")
        if not 0 < budget_ratio <= 1:
            raise ValueError("budget_ratio must be in (0, 1]")

        self.budget_mode = budget_mode
        self.allocation = allocation
        self.budget_ratio = budget_ratio
        self.temperature = temperature
        self.distill_lambda = distill_lambda
        self.distill_temperature = distill_temperature
        self.disagreement_lambda = disagreement_lambda
        if instability_mode not in {"entropy_kl", "virtual_replay_loss"}:
            raise ValueError(
                "instability_mode must be one of: entropy_kl, virtual_replay_loss"
            )
        self.instability_mode = instability_mode
        if fusion_mode not in {"product", "c_gated"}:
            raise ValueError("fusion_mode must be one of: product, c_gated")
        if instability_lambda < 0:
            raise ValueError("instability_lambda must be >= 0")
        self.fusion_mode = fusion_mode
        self.instability_lambda = instability_lambda
        self.prior_momentum = prior_momentum
        if consequence_mode not in {
            "loss_ema",
            "historical_forgetting",
            "raw_forgetting",
            "validation_error",
        }:
            raise ValueError(
                "consequence_mode must be one of: loss_ema, historical_forgetting, "
                "raw_forgetting, validation_error"
            )
        self.consequence_mode = consequence_mode
        if allocation_scope not in {"global", "replay_group"}:
            raise ValueError("allocation_scope must be one of: global, replay_group")
        self.allocation_scope = allocation_scope
        if gradient_audit_every <= 0:
            raise ValueError("gradient_audit_every must be > 0")
        self.gradient_audit = gradient_audit
        self.gradient_audit_every = gradient_audit_every
        if counterfactual_audit_every <= 0:
            raise ValueError("counterfactual_audit_every must be > 0")
        if counterfactual_probe_size <= 0:
            raise ValueError("counterfactual_probe_size must be > 0")
        self.counterfactual_audit = counterfactual_audit
        self.counterfactual_audit_every = counterfactual_audit_every
        self.counterfactual_probe_size = counterfactual_probe_size
        if not counterfactual_alpha_values:
            raise ValueError("counterfactual_alpha_values must not be empty")
        self.counterfactual_alpha_audit = counterfactual_alpha_audit
        self.counterfactual_alpha_values = tuple(
            sorted({float(alpha) for alpha in counterfactual_alpha_values})
        )
        if any(alpha < 0.0 or alpha > 1.0 for alpha in self.counterfactual_alpha_values):
            raise ValueError("counterfactual_alpha_values must be in [0, 1]")
        if 0.0 not in self.counterfactual_alpha_values:
            raise ValueError("counterfactual_alpha_values must include 0.0")
        if not 0.0 < counterfactual_rho <= 1.0:
            raise ValueError("counterfactual_rho must be in (0, 1]")
        self.counterfactual_rho = float(counterfactual_rho)
        if plasticity_budget_every <= 0:
            raise ValueError("plasticity_budget_every must be > 0")
        self.plasticity_budget_control = bool(plasticity_budget_control)
        self.plasticity_budget_every = int(plasticity_budget_every)
        self._plasticity_replay_scale = 1.0
        if replay_geometry_audit_every <= 0:
            raise ValueError("replay_geometry_audit_every must be > 0")
        self.replay_geometry_audit = bool(replay_geometry_audit)
        self.replay_geometry_audit_every = int(replay_geometry_audit_every)
        self.conflict_residual_audit = bool(conflict_residual_audit)
        self.conflict_residual_audit_every = int(conflict_residual_audit_every)
        if slow_fast_state_audit_every <= 0:
            raise ValueError("slow_fast_state_audit_every must be > 0")
        if not 0.0 < slow_fast_ema_decay < 1.0:
            raise ValueError("slow_fast_ema_decay must be in (0, 1)")
        self.slow_fast_state_audit = bool(slow_fast_state_audit)
        self.slow_fast_state_audit_every = int(slow_fast_state_audit_every)
        self.slow_fast_ema_decay = float(slow_fast_ema_decay)
        if boundary_debt_probe_size <= 0:
            raise ValueError("boundary_debt_probe_size must be > 0")
        self.boundary_debt_audit = boundary_debt_audit
        self.boundary_debt_probe_size = boundary_debt_probe_size
        if boundary_debt_temperature <= 0:
            raise ValueError("boundary_debt_temperature must be > 0")
        if not 0.0 <= boundary_debt_mix <= 1.0:
            raise ValueError("boundary_debt_mix must be in [0, 1]")
        self.boundary_debt_replay = boundary_debt_replay
        self.boundary_debt_temperature = float(boundary_debt_temperature)
        self.boundary_debt_mix = float(boundary_debt_mix)
        if boundary_repair_pairs <= 0 or boundary_repair_samples_per_class <= 0:
            raise ValueError("boundary repair pair and sample counts must be > 0")
        if boundary_repair_margin < 0 or boundary_repair_lambda < 0:
            raise ValueError("boundary repair margin and lambda must be >= 0")
        if boundary_repair_current_ce_tolerance < 0:
            raise ValueError("boundary_repair_current_ce_tolerance must be >= 0")
        self.boundary_repair = boundary_repair
        self.boundary_repair_pairs = int(boundary_repair_pairs)
        self.boundary_repair_samples_per_class = int(boundary_repair_samples_per_class)
        self.boundary_repair_margin = float(boundary_repair_margin)
        self.boundary_repair_lambda = float(boundary_repair_lambda)
        self.boundary_repair_current_ce_tolerance = float(
            boundary_repair_current_ce_tolerance
        )
        self.boundary_feature_repair = boundary_feature_repair
        self.use_instability = use_instability
        self.use_consequence = use_consequence
        self.use_uncertainty = use_uncertainty
        self.use_disagreement = use_disagreement
        self.use_prior = use_prior
        self.high_low_fraction = high_low_fraction
        self.fair_compute = fair_compute
        self.eps = eps

        self.teacher = None
        self.class_prior: Optional[torch.Tensor] = None
        self.teacher_forward_calls = 0
        self.teacher_forward_examples = 0
        self.history: List[Dict[str, float]] = []
        self.class_best_accuracy: Dict[int, float] = {}
        self.class_validation_accuracy: Dict[int, float] = {}
        self.consequence_history: List[Dict[str, object]] = []
        self.gradient_audit_history: List[Dict[str, float]] = []
        self.counterfactual_audit_history: List[Dict[str, float]] = []
        self.counterfactual_alpha_audit_history: List[Dict[str, float]] = []
        self.plasticity_budget_history: List[Dict[str, float]] = []
        self.replay_geometry_audit_history: List[Dict[str, float]] = []
        self.conflict_residual_audit_history: List[Dict[str, float]] = []
        self.slow_fast_state_audit_history: List[Dict[str, float]] = []
        self._slow_state_model = None
        self._slow_state_updates = 0
        self._current_validation_loader = None
        self._current_validation_iterator = None
        self.boundary_debt_audit_history: List[Dict[str, float]] = []
        self._boundary_debt_loader = None
        self._boundary_debt_experience_id = -1
        self._boundary_debt_pending: List[Dict[str, float]] = []
        self._seen_validation_classes: set[int] = set()
        self.class_boundary_debt: Dict[int, float] = {}
        self.boundary_debt_replay_history: List[Dict[str, object]] = []
        self.boundary_debt_pairs: Dict[tuple[int, int], float] = {}
        self.boundary_repair_history: List[Dict[str, object]] = []

    def set_current_validation_experience(
        self, validation_experience, *, experience_id: int
    ) -> None:
        """Set a fixed train-held-out probe for the next counterfactual audit.

        The probe belongs to the *current* training experience and is already
        excluded from gradient updates and replay memory by ``data.py``.  It is
        used only to test a proposed update against the same update with
        uniform replay weights; neither candidate is applied by this audit.
        """
        if not (
            self.counterfactual_audit
            or self.counterfactual_alpha_audit
            or self.plasticity_budget_control
            or self.conflict_residual_audit
            or self.slow_fast_state_audit
            or self.boundary_debt_audit
            or self.boundary_debt_replay
            or self.boundary_repair
            or self.boundary_feature_repair
        ):
            return
        dataset = validation_experience.dataset.eval()
        if (
            self.counterfactual_audit
            or self.counterfactual_alpha_audit
            or self.plasticity_budget_control
            or self.conflict_residual_audit
            or self.slow_fast_state_audit
        ):
            size = min(len(dataset), self.counterfactual_probe_size)
            if size == 0:
                self._current_validation_loader = None
                self._current_validation_iterator = None
            else:
                generator = torch.Generator().manual_seed(10_000 + int(experience_id))
                indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
                probe = Subset(dataset, indices)
                self._current_validation_loader = DataLoader(
                    probe, batch_size=size, shuffle=False, num_workers=0
                )
                self._current_validation_iterator = iter(self._current_validation_loader)

        if (
            self.boundary_debt_audit
            or self.boundary_debt_replay
            or self.boundary_repair
            or self.boundary_feature_repair
            or self.boundary_feature_repair
        ):
            size = min(len(dataset), self.boundary_debt_probe_size)
            generator = torch.Generator().manual_seed(20_000 + int(experience_id))
            indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
            probe = Subset(dataset, indices)
            self._boundary_debt_loader = DataLoader(
                probe, batch_size=size, shuffle=False, num_workers=0
            )
            self._boundary_debt_experience_id = int(experience_id)

    def before_training_exp(self, strategy, **kwargs):
        """Ensure the frozen teacher is on the same device as the strategy."""
        if self.teacher is not None:
            self.teacher.to(strategy.device)
            self.teacher.eval()
        if self.slow_fast_state_audit and self._slow_state_model is None:
            self._slow_state_model = copy.deepcopy(strategy.model).to(strategy.device)
            self._slow_state_model.eval()
            for parameter in self._slow_state_model.parameters():
                parameter.requires_grad_(False)
        self._record_boundary_debt_before_training(strategy)

    def before_backward(self, strategy, **kwargs):
        """Replace the default mean CE with budget-weighted CE/distillation."""
        logits = strategy.mb_output
        targets = strategy.mb_y.long()
        num_classes = logits.shape[1]
        self._ensure_prior(num_classes, logits.device)

        ce_vec = F.cross_entropy(logits, targets, reduction="none")
        correct_vec = (logits.argmax(dim=1) == targets).float()

        teacher_logits = self._teacher_forward(strategy)
        instability = self._estimate_instability(strategy, logits, teacher_logits)
        consequence = self._estimate_consequence(targets, logits.device)
        risk = self._fuse_risk(instability, consequence)

        weights = self._allocation_weights(risk, strategy)
        replay_scale = float(getattr(strategy, "rbcl_replay_loss_scale", 1.0))
        current_n = int(getattr(strategy, "rbcl_current_batch_size", 0))
        if replay_scale != 1.0 and 0 < current_n < int(weights.numel()):
            # A strategy-owned, task-ID-free group-mass adjustment. The prefix
            # is the arriving batch and the suffix is memory injected by the
            # replay plugin; it never depends on class labels or experience ID.
            weights = weights * torch.cat(
                [
                    torch.ones(current_n, device=weights.device),
                    torch.full(
                        (weights.numel() - current_n,),
                        replay_scale,
                        device=weights.device,
                    ),
                ]
            )
        uncertainty_lambda = float(
            getattr(strategy, "rbcl_current_uncertainty_lambda", 0.0)
        )
        if uncertainty_lambda != 0.0 and 0 < current_n < int(weights.numel()):
            probabilities = torch.softmax(logits[:current_n], dim=1)
            entropy = -(
                probabilities * probabilities.clamp_min(self.eps).log()
            ).sum(dim=1) / math.log(max(2, int(logits.shape[1])))
            # Prediction-only plasticity: uncertain arriving samples receive a
            # small extra loss mass. This is not a label/task boundary signal.
            weights[:current_n] = weights[:current_n] * (
                1.0 + uncertainty_lambda * entropy.detach()
            )
        if self.boundary_debt_replay:
            weights = self._boundary_debt_replay_weights(strategy, weights)
        per_sample_loss = ce_vec
        if teacher_logits is not None and self.distill_lambda > 0:
            kd_vec = self._distillation_vector(logits, teacher_logits)
            per_sample_loss = ce_vec + self.distill_lambda * kd_vec

        plasticity_scale = self._select_plasticity_replay_scale(
            strategy, per_sample_loss=per_sample_loss
        )
        if plasticity_scale != 1.0 and 0 < current_n < int(weights.numel()):
            weights = weights * torch.cat(
                [
                    torch.ones(current_n, device=weights.device),
                    torch.full(
                        (weights.numel() - current_n,),
                        plasticity_scale,
                        device=weights.device,
                    ),
                ]
            )

        # This is the exact objective used by Uniform/Risk Budget experiments.
        strategy.loss = self._weighted_mean(per_sample_loss, weights)

        if self.consequence_mode == "loss_ema":
            self._update_class_prior(targets, ce_vec.detach())
        self._record_gradient_audit(strategy, ce_vec, risk)
        self._record_replay_geometry_audit(strategy, ce_vec)
        self._record_conflict_residual_audit(strategy, ce_vec)
        self._record_slow_fast_state_audit(strategy)
        self._record_counterfactual_audit(strategy, per_sample_loss, risk)
        self._record_counterfactual_alpha_audit(strategy, per_sample_loss)
        self._record_step(
            strategy,
            instability,
            consequence,
            risk,
            weights,
            ce_vec.detach(),
            correct_vec.detach(),
        )

    def after_training_exp(self, strategy, **kwargs):
        """Freeze f_t as teacher f_{t-1} for the next continual experience."""
        self._record_boundary_debt_after_training(strategy)
        self._run_boundary_repair(strategy)
        self.teacher = copy.deepcopy(strategy.model)
        self.teacher.to(strategy.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def after_training_iteration(self, strategy, **kwargs):
        """Update the diagnostic EMA state without changing the trained model."""
        if not self.slow_fast_state_audit or self._slow_state_model is None:
            return
        decay = self.slow_fast_ema_decay
        fast_parameters = dict(strategy.model.named_parameters())
        for name, slow_parameter in self._slow_state_model.named_parameters():
            fast_parameter = fast_parameters[name]
            slow_parameter.mul_(decay).add_(fast_parameter.detach(), alpha=1.0 - decay)
        fast_buffers = dict(strategy.model.named_buffers())
        for name, slow_buffer in self._slow_state_model.named_buffers():
            fast_buffer = fast_buffers[name].detach()
            if slow_buffer.dtype.is_floating_point:
                slow_buffer.mul_(decay).add_(fast_buffer, alpha=1.0 - decay)
            else:
                slow_buffer.copy_(fast_buffer)
        self._slow_state_updates += 1

    @torch.no_grad()
    def update_historical_consequence(
        self,
        strategy,
        validation_experiences,
        *,
        batch_size: int,
        num_workers: int = 0,
        experience_id: int = -1,
    ) -> None:
        """Update class consequence from held-out, already-seen validation data.

        Validation samples are split from the training stream before strategy.train
        and never enter gradient updates or ReplayPlugin memory. This avoids test
        leakage while making consequence a historical functional-loss proxy.
        """
        if self.consequence_mode not in {
            "historical_forgetting",
            "raw_forgetting",
            "validation_error",
        }:
            return

        was_training = strategy.model.training
        strategy.model.eval()
        correct: Dict[int, int] = {}
        total: Dict[int, int] = {}
        for validation_experience in validation_experiences:
            dataset = validation_experience.dataset.eval()
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            for batch in loader:
                x, y = batch[0].to(strategy.device), batch[1].long().to(strategy.device)
                task_ids = batch[2].to(strategy.device) if len(batch) >= 3 else None
                logits = avalanche_forward(strategy.model, x, task_ids)
                pred = logits.argmax(dim=1)
                for cls in y.unique():
                    cls_id = int(cls.item())
                    mask = y == cls
                    correct[cls_id] = correct.get(cls_id, 0) + int(
                        (pred[mask] == y[mask]).sum()
                    )
                    total[cls_id] = total.get(cls_id, 0) + int(mask.sum())
        if was_training:
            strategy.model.train()

        exp_id = int(experience_id)
        snapshot: Dict[str, float] = {}
        for cls_id, count in total.items():
            accuracy = correct[cls_id] / max(1, count)
            best = max(self.class_best_accuracy.get(cls_id, accuracy), accuracy)
            forgetting = max(0.0, best - accuracy)
            self.class_best_accuracy[cls_id] = best
            self.class_validation_accuracy[cls_id] = accuracy
            self._ensure_prior(cls_id + 1, strategy.device)
            old_cost = float(self.class_prior[cls_id].detach().cpu())
            if self.consequence_mode == "historical_forgetting":
                # Main method: an EMA of historical functional loss. The offset
                # makes an unseen/perfectly retained class neutral rather than
                # assigning it a zero loss weight.
                observed_cost = 1.0 + forgetting
                next_cost = (
                    self.prior_momentum * old_cost
                    + (1.0 - self.prior_momentum) * observed_cost
                )
            elif self.consequence_mode == "raw_forgetting":
                # Engineering-3 proxy: current held-out forgetting only, with
                # no temporal smoothing or historical aggregation.
                observed_cost = 1.0 + forgetting
                next_cost = observed_cost
            else:
                # Engineering-3 proxy: current held-out classification error.
                # It has no best-accuracy reference and no EMA history.
                observed_cost = 1.0 + (1.0 - accuracy)
                next_cost = observed_cost
            self.class_prior[cls_id] = next_cost
            snapshot[str(cls_id)] = {
                "accuracy": accuracy,
                "best_accuracy": best,
                "forgetting": forgetting,
                "consequence": float(self.class_prior[cls_id].detach().cpu()),
            }
        self.consequence_history.append(
            {"experience": exp_id, "classes": snapshot}
        )

    @torch.no_grad()
    def _teacher_forward(self, strategy) -> Optional[torch.Tensor]:
        """Run the teacher path used by both Risk and Uniform Budget modes."""
        if self.teacher is None:
            return None
        if not (self.fair_compute or self.distill_lambda > 0 or self.use_disagreement):
            return None

        self.teacher_forward_calls += 1
        self.teacher_forward_examples += int(strategy.mb_x.shape[0])
        task_ids = strategy.mbatch[-1] if strategy.mbatch is not None and len(strategy.mbatch) >= 3 else None
        return avalanche_forward(self.teacher, strategy.mb_x, task_ids)

    def _estimate_instability(
        self,
        strategy,
        logits: torch.Tensor,
        teacher_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Estimate I_t(z); virtual replay interference is the default."""
        if not self.use_instability:
            return torch.ones(logits.shape[0], dtype=torch.float32, device=logits.device)

        if self.instability_mode == "virtual_replay_loss":
            return self._virtual_replay_loss_increase(strategy, logits)

        instability = logits.new_zeros(logits.shape[0])

        if self.use_uncertainty:
            probs = logits.softmax(dim=1)
            log_probs = logits.log_softmax(dim=1)
            entropy = -(probs * log_probs).sum(dim=1)
            instability = instability + entropy

        if self.use_disagreement and teacher_logits is not None:
            disagreement = self._distillation_vector(logits, teacher_logits)
            instability = instability + self.disagreement_lambda * disagreement

        return instability

    def _virtual_replay_loss_increase(self, strategy, logits: torch.Tensor) -> torch.Tensor:
        """MIR-style instability from a one-step virtual update.

        ReplayDataLoader concatenates current-data samples before memory samples.
        We update a disposable model copy on the current-data prefix, then score
        the per-memory-sample CE increase. Current samples receive a neutral
        score, so this estimator never mistakes current difficulty for old-task
        interference. If no replay prefix/suffix split is available, the score
        is neutral rather than inferred from an unsafe heuristic.
        """
        batch_sizes = getattr(getattr(strategy, "dataloader", None), "batch_sizes", ())
        if len(batch_sizes) < 2:
            return torch.ones_like(logits[:, 0])
        current_n = int(batch_sizes[0])
        total_n = int(logits.shape[0])
        if current_n <= 0 or current_n >= total_n:
            return torch.ones_like(logits[:, 0])

        x = strategy.mb_x
        y = strategy.mb_y.long()
        task_ids = strategy.mbatch[-1] if strategy.mbatch is not None and len(strategy.mbatch) >= 3 else None
        current_task = task_ids[:current_n] if task_ids is not None else None
        memory_task = task_ids[current_n:] if task_ids is not None else None

        virtual_model = copy.deepcopy(strategy.model).to(strategy.device)
        virtual_model.eval()
        params = [p for p in virtual_model.parameters() if p.requires_grad]
        if not params:
            return torch.ones_like(logits[:, 0])

        current_logits = avalanche_forward(virtual_model, x[:current_n], current_task)
        current_loss = F.cross_entropy(current_logits, y[:current_n])
        grads = torch.autograd.grad(current_loss, params, allow_unused=True)
        lr = float(strategy.optimizer.param_groups[0]["lr"])
        with torch.no_grad():
            before = F.cross_entropy(
                avalanche_forward(virtual_model, x[current_n:], memory_task),
                y[current_n:], reduction="none"
            )
            for param, grad in zip(params, grads):
                if grad is not None:
                    param.add_(grad, alpha=-lr)
            after = F.cross_entropy(
                avalanche_forward(virtual_model, x[current_n:], memory_task),
                y[current_n:], reduction="none"
            )
        del virtual_model
        memory_increase = (after - before).clamp_min(0.0).detach()
        # Preserve a neutral baseline of one for all samples. The positive
        # increase is an *extra* preservation priority for replay samples;
        # using the raw increase would often make old samples score below the
        # neutral current-data score and invert the intended allocation.
        return torch.cat(
            [
                torch.ones(current_n, device=logits.device),
                torch.ones_like(memory_increase) + memory_increase,
            ]
        )

    def _estimate_consequence(self, targets: torch.Tensor, device) -> torch.Tensor:
        """Estimate C_t(z) using class-level historical loss EMA."""
        if not self.use_consequence:
            return torch.ones_like(targets, dtype=torch.float32, device=device)
        if not self.use_prior or self.class_prior is None:
            return torch.ones_like(targets, dtype=torch.float32, device=device)
        safe_targets = targets.clamp(max=self.class_prior.numel() - 1)
        return self.class_prior.to(device)[safe_targets].float()

    def _fuse_risk(
        self, instability: torch.Tensor, consequence: torch.Tensor
    ) -> torch.Tensor:
        """Combine I and C while preserving exact component ablations."""
        i_bar = self._normalize(instability)
        c_bar = self._normalize(consequence)
        if not self.use_instability:
            return c_bar
        if not self.use_consequence:
            return i_bar
        if self.fusion_mode == "product":
            return i_bar * c_bar
        return c_bar * (1.0 + self.instability_lambda * i_bar)

    def _allocation_weights(self, risk: torch.Tensor, strategy=None) -> torch.Tensor:
        """Allocate the update/preservation budget from risk or uniform scores."""
        if self.budget_mode == "none":
            return torch.ones_like(risk)

        if self.budget_mode == "uniform":
            scores = torch.ones_like(risk)
        else:
            scores = risk

        if self.allocation == "soft":
            if self.budget_mode == "uniform":
                return torch.ones_like(scores)
            if self.allocation_scope == "replay_group":
                return self._replay_group_weights(scores, strategy)
            tau = max(self.temperature, self.eps)
            return torch.softmax(scores / tau, dim=0) * scores.numel()

        k = max(1, int(math.ceil(scores.numel() * self.budget_ratio)))
        weights = torch.zeros_like(scores)
        if self.budget_mode == "uniform":
            selected = torch.randperm(scores.numel(), device=scores.device)[:k]
        else:
            selected = torch.topk(scores, k=k, largest=True).indices
        weights[selected] = 1.0
        return weights

    def _replay_group_weights(self, scores: torch.Tensor, strategy) -> torch.Tensor:
        """Reweight only memory samples while preserving current-group mass.

        ReplayDataLoader places current samples before replay samples.  Giving
        current samples a unit weight and normalizing softmax only over the
        replay suffix guarantees that both groups retain their original total
        contribution: sum(w_current)=n_current and sum(w_memory)=n_memory.
        Consequently, the current-example coefficient cannot be reduced merely
        because replay samples receive heterogeneous priorities.
        """
        batch_sizes = getattr(getattr(strategy, "dataloader", None), "batch_sizes", ())
        if len(batch_sizes) < 2:
            return torch.ones_like(scores)
        current_n = int(batch_sizes[0])
        total_n = int(scores.numel())
        if current_n <= 0 or current_n >= total_n:
            return torch.ones_like(scores)

        memory_scores = scores[current_n:]
        tau = max(self.temperature, self.eps)
        memory_weights = torch.softmax(memory_scores / tau, dim=0) * memory_scores.numel()
        return torch.cat([torch.ones(current_n, device=scores.device), memory_weights])

    def _boundary_debt_replay_weights(
        self, strategy, base_weights: torch.Tensor
    ) -> torch.Tensor:
        """Prioritize old classes with high predicted boundary debt.

        This changes only the distribution *within* the replay suffix.  The
        current prefix stays at unit weight and the replay suffix always sums
        to its original batch size, so neither replay count nor current/replay
        group mass is changed relative to Uniform replay.
        """
        batch_sizes = getattr(getattr(strategy, "dataloader", None), "batch_sizes", ())
        if len(batch_sizes) < 2:
            return base_weights
        current_n = int(batch_sizes[0])
        total_n = int(base_weights.numel())
        if current_n <= 0 or current_n >= total_n or not self.class_boundary_debt:
            return base_weights

        memory_targets = strategy.mb_y.long()[current_n:]
        debt = torch.tensor(
            [self.class_boundary_debt.get(int(label), 0.0) for label in memory_targets],
            dtype=base_weights.dtype,
            device=base_weights.device,
        )
        spread = debt.max() - debt.min()
        if float(spread.detach().item()) <= self.eps:
            return base_weights
        normalized = (debt - debt.min()) / (spread + self.eps)
        priority = torch.softmax(
            normalized / self.boundary_debt_temperature, dim=0
        ) * normalized.numel()
        memory_weights = (
            (1.0 - self.boundary_debt_mix)
            + self.boundary_debt_mix * priority
        )
        return torch.cat(
            [torch.ones(current_n, device=base_weights.device), memory_weights]
        )

    def _record_gradient_audit(
        self, strategy, ce_vec: torch.Tensor, risk: torch.Tensor
    ) -> None:
        """Log a non-mutating classifier-head current/replay conflict proxy.

        This diagnostic is intentionally based on unweighted CE only: it asks
        whether the current and memory classification objectives point in
        opposing directions before any allocation rule modifies their mass.
        ``autograd.grad`` populates neither ``.grad`` nor optimizer state, so
        the audit cannot alter the subsequent training update.
        """
        if not self.gradient_audit:
            return
        iteration = int(strategy.clock.train_iterations)
        if iteration % self.gradient_audit_every != 0:
            return
        batch_sizes = getattr(getattr(strategy, "dataloader", None), "batch_sizes", ())
        current_n = int(
            getattr(strategy, "rbcl_current_batch_size", batch_sizes[0] if batch_sizes else 0)
        )
        total_n = int(ce_vec.numel())
        if current_n <= 0 or current_n >= total_n:
            return

        try:
            _, head = get_last_fc_layer(strategy.model)
        except ValueError:
            return
        head_params = tuple(param for param in head.parameters() if param.requires_grad)
        if not head_params:
            return

        current_grad = self._loss_gradient_vector(ce_vec[:current_n].mean(), head_params)
        memory_grad = self._loss_gradient_vector(ce_vec[current_n:].mean(), head_params)
        pair = self._gradient_pair_stats(current_grad, memory_grad)

        memory_targets = strategy.mb_y.long()[current_n:]
        memory_risk = risk.detach()[current_n:]
        class_cosines: List[float] = []
        class_conflicts: List[float] = []
        class_risks: List[float] = []
        for class_id in memory_targets.unique():
            mask = memory_targets == class_id
            class_grad = self._loss_gradient_vector(ce_vec[current_n:][mask].mean(), head_params)
            class_pair = self._gradient_pair_stats(current_grad, class_grad)
            class_cosines.append(class_pair["cosine"])
            class_conflicts.append(class_pair["conflict"])
            class_risks.append(float(memory_risk[mask].mean().cpu()))

        risk_total = sum(class_risks)
        if risk_total <= self.eps:
            risk_total = float(len(class_risks))
            class_risks = [1.0] * len(class_risks)
        weighted_cosine = sum(
            score * value for score, value in zip(class_risks, class_cosines)
        ) / risk_total
        weighted_conflict = sum(
            score * value for score, value in zip(class_risks, class_conflicts)
        ) / risk_total
        self.gradient_audit_history.append(
            {
                "experience": float(getattr(strategy.experience, "current_experience", -1)),
                "iteration": float(iteration),
                "current_memory_cosine": pair["cosine"],
                "current_memory_conflict": pair["conflict"],
                "risk_weighted_class_cosine": weighted_cosine,
                "risk_weighted_class_conflict_fraction": weighted_conflict,
                "class_conflict_fraction": sum(class_conflicts) / len(class_conflicts),
                "memory_class_count": float(len(class_conflicts)),
            }
        )

    def _record_replay_geometry_audit(
        self, strategy, ce_vec: torch.Tensor
    ) -> None:
        """Measure sample-level replay compatibility without changing updates.

        Aggregate current-vs-memory conflict can hide a small useful subset of
        replay samples.  This audit records that subset's availability using
        only classifier-head gradients before the actual backward update.
        """
        if not self.replay_geometry_audit:
            return
        iteration = int(strategy.clock.train_iterations)
        if iteration % self.replay_geometry_audit_every != 0:
            return
        batch_sizes = getattr(getattr(strategy, "dataloader", None), "batch_sizes", ())
        current_n = int(
            getattr(strategy, "rbcl_current_batch_size", batch_sizes[0] if batch_sizes else 0)
        )
        total_n = int(ce_vec.numel())
        if current_n <= 0 or current_n >= total_n:
            return
        try:
            _, head = get_last_fc_layer(strategy.model)
        except ValueError:
            return
        head_params = tuple(param for param in head.parameters() if param.requires_grad)
        if not head_params:
            return

        current_grad = self._loss_gradient_vector(ce_vec[:current_n].mean(), head_params)
        cosines: List[float] = []
        for sample_loss in ce_vec[current_n:]:
            sample_grad = self._loss_gradient_vector(sample_loss, head_params)
            cosines.append(self._gradient_pair_stats(current_grad, sample_grad)["cosine"])
        if not cosines:
            return
        ordered = sorted(cosines)
        top_k = max(1, int(math.ceil(len(ordered) * 0.25)))
        top = ordered[-top_k:]
        bottom = ordered[:top_k]
        self.replay_geometry_audit_history.append(
            {
                "experience": float(getattr(strategy.experience, "current_experience", -1)),
                "iteration": float(iteration),
                "memory_sample_count": float(len(cosines)),
                "mean_sample_cosine": sum(cosines) / len(cosines),
                "compatible_fraction": sum(value >= 0.0 for value in cosines) / len(cosines),
                "top_quartile_cosine": sum(top) / len(top),
                "bottom_quartile_cosine": sum(bottom) / len(bottom),
                "memory_ce_mean": float(ce_vec[current_n:].mean().detach().cpu()),
            }
        )

    def _next_current_validation_probe(self, strategy):
        """Return a reproducible held-out current-class probe, if configured."""
        if self._current_validation_loader is None:
            return None
        try:
            batch = next(self._current_validation_iterator)
        except StopIteration:
            self._current_validation_iterator = iter(self._current_validation_loader)
            batch = next(self._current_validation_iterator)
        x = batch[0].to(strategy.device)
        y = batch[1].long().to(strategy.device)
        task_ids = batch[2].to(strategy.device) if len(batch) >= 3 else None
        return x, y, task_ids

    def _record_conflict_residual_audit(self, strategy, ce_vec: torch.Tensor) -> None:
        """Compare head-only uniform and conflict-residual replay on copies."""
        if not self.conflict_residual_audit:
            return
        iteration = int(strategy.clock.train_iterations)
        if iteration % self.conflict_residual_audit_every != 0:
            return
        batch_sizes = getattr(getattr(strategy, "dataloader", None), "batch_sizes", ())
        current_n = int(getattr(strategy, "rbcl_current_batch_size", batch_sizes[0] if batch_sizes else 0))
        if current_n <= 0 or current_n >= int(ce_vec.numel()):
            return
        probe = self._next_current_validation_probe(strategy)
        if probe is None:
            return
        try:
            _, head = get_last_fc_layer(strategy.model)
        except ValueError:
            return
        params = tuple(p for p in head.parameters() if p.requires_grad)
        current_grad = self._loss_gradient_vector(ce_vec[:current_n].mean(), params)
        memory_grad = self._loss_gradient_vector(ce_vec[current_n:].mean(), params)
        denom = current_grad.dot(current_grad).clamp_min(self.eps)
        conflict_coeff = torch.minimum(
            torch.tensor(0.0, device=denom.device), memory_grad.dot(current_grad) / denom
        )
        residual_memory = memory_grad - conflict_coeff * current_grad

        def copy_with(vector):
            candidate = copy.deepcopy(strategy.model).to(strategy.device)
            _, candidate_head = get_last_fc_layer(candidate)
            lr = float(strategy.optimizer.param_groups[0]["lr"])
            offset = 0
            with torch.no_grad():
                for param in candidate_head.parameters():
                    if not param.requires_grad:
                        continue
                    width = param.numel()
                    param.add_(vector[offset:offset + width].reshape_as(param), alpha=-lr)
                    offset += width
            candidate.eval()
            return candidate

        current_model = copy_with(current_grad)
        uniform_model = copy_with((current_grad + memory_grad) / 2.0)
        residual_model = copy_with((current_grad + residual_memory) / 2.0)
        # First-order two-stage repair: after the residual main update, apply
        # one unscaled memory correction. This is evaluated on a copy only.
        repair_model = copy_with((current_grad + residual_memory) / 2.0 + memory_grad)
        x_probe, y_probe, probe_task_ids = probe
        memory_x, memory_y = strategy.mb_x[current_n:], strategy.mb_y.long()[current_n:]
        task_ids = strategy.mbatch[-1] if strategy.mbatch is not None and len(strategy.mbatch) >= 3 else None
        memory_task_ids = task_ids[current_n:] if task_ids is not None else None
        with torch.no_grad():
            before = F.cross_entropy(avalanche_forward(strategy.model, x_probe, probe_task_ids), y_probe)
            current_ce = F.cross_entropy(avalanche_forward(current_model, x_probe, probe_task_ids), y_probe)
            uniform_ce = F.cross_entropy(avalanche_forward(uniform_model, x_probe, probe_task_ids), y_probe)
            residual_ce = F.cross_entropy(avalanche_forward(residual_model, x_probe, probe_task_ids), y_probe)
            repair_ce = F.cross_entropy(avalanche_forward(repair_model, x_probe, probe_task_ids), y_probe)
            current_mem = F.cross_entropy(avalanche_forward(current_model, memory_x, memory_task_ids), memory_y)
            uniform_mem = F.cross_entropy(avalanche_forward(uniform_model, memory_x, memory_task_ids), memory_y)
            residual_mem = F.cross_entropy(avalanche_forward(residual_model, memory_x, memory_task_ids), memory_y)
            repair_mem = F.cross_entropy(avalanche_forward(repair_model, memory_x, memory_task_ids), memory_y)
        cp = float((before - current_ce).cpu())
        up = float((before - uniform_ce).cpu())
        rp = float((before - residual_ce).cpu())
        ug = float((current_mem - uniform_mem).cpu())
        rg = float((current_mem - residual_mem).cpu())
        xp = float((before - repair_ce).cpu())
        xg = float((current_mem - repair_mem).cpu())
        self.conflict_residual_audit_history.append({
            "experience": float(strategy.experience.current_experience), "iteration": float(iteration),
            "current_only_progress": cp, "uniform_progress": up, "residual_progress": rp,
            "uniform_memory_gain": ug, "residual_memory_gain": rg,
            "repair_progress": xp, "repair_memory_gain": xg,
            "residual_keeps_budget": float(cp > 0 and rp >= self.counterfactual_rho * cp),
            "residual_beats_uniform_progress": float(rp >= up),
            "residual_keeps_memory_gain": float(rg >= ug),
            "memory_conflict_coefficient": float(conflict_coeff.cpu()),
            "repair_keeps_budget": float(cp > 0 and xp >= self.counterfactual_rho * cp),
            "repair_improves_memory_over_residual": float(xg >= rg),
            "repair_pareto": float(cp > 0 and xp >= self.counterfactual_rho * cp and xg >= rg),
        })
        del current_model, uniform_model, residual_model, repair_model

    @staticmethod
    def _confidence_fused_probabilities(
        fast_logits: torch.Tensor, slow_logits: torch.Tensor
    ) -> torch.Tensor:
        """Fuse fast/slow predictions using task-free normalized confidence."""
        fast_probabilities = torch.softmax(fast_logits, dim=1)
        slow_probabilities = torch.softmax(slow_logits, dim=1)
        normalizer = math.log(max(2, int(fast_logits.shape[1])))
        fast_entropy = -(
            fast_probabilities * fast_probabilities.clamp_min(1e-8).log()
        ).sum(dim=1) / normalizer
        slow_entropy = -(
            slow_probabilities * slow_probabilities.clamp_min(1e-8).log()
        ).sum(dim=1) / normalizer
        fast_confidence = (1.0 - fast_entropy).clamp_min(1e-4)
        slow_confidence = (1.0 - slow_entropy).clamp_min(1e-4)
        slow_weight = slow_confidence / (fast_confidence + slow_confidence)
        return (
            (1.0 - slow_weight).unsqueeze(1) * fast_probabilities
            + slow_weight.unsqueeze(1) * slow_probabilities
        )

    def _record_slow_fast_state_audit(self, strategy) -> None:
        """Audit an EMA slow state and confidence fusion without training it."""
        if not self.slow_fast_state_audit or self._slow_state_model is None:
            return
        iteration = int(strategy.clock.train_iterations)
        if iteration % self.slow_fast_state_audit_every != 0:
            return
        if self._slow_state_updates <= 0:
            return
        batch_sizes = getattr(getattr(strategy, "dataloader", None), "batch_sizes", ())
        current_n = int(
            getattr(strategy, "rbcl_current_batch_size", batch_sizes[0] if batch_sizes else 0)
        )
        if current_n <= 0 or current_n >= int(strategy.mb_y.numel()):
            return
        probe = self._next_current_validation_probe(strategy)
        if probe is None:
            return

        x_probe, y_probe, probe_task_ids = probe
        memory_x = strategy.mb_x[current_n:]
        memory_y = strategy.mb_y.long()[current_n:]
        task_ids = (
            strategy.mbatch[-1]
            if strategy.mbatch is not None and len(strategy.mbatch) >= 3
            else None
        )
        memory_task_ids = task_ids[current_n:] if task_ids is not None else None
        was_training = strategy.model.training
        strategy.model.eval()
        self._slow_state_model.eval()
        with torch.no_grad():
            fast_current_logits = avalanche_forward(
                strategy.model, x_probe, probe_task_ids
            )
            slow_current_logits = avalanche_forward(
                self._slow_state_model, x_probe, probe_task_ids
            )
            fast_memory_logits = avalanche_forward(
                strategy.model, memory_x, memory_task_ids
            )
            slow_memory_logits = avalanche_forward(
                self._slow_state_model, memory_x, memory_task_ids
            )
            fused_current = self._confidence_fused_probabilities(
                fast_current_logits, slow_current_logits
            )
            fused_memory = self._confidence_fused_probabilities(
                fast_memory_logits, slow_memory_logits
            )
            fast_current_ce = F.cross_entropy(fast_current_logits, y_probe)
            slow_current_ce = F.cross_entropy(slow_current_logits, y_probe)
            fused_current_ce = F.nll_loss(fused_current.clamp_min(1e-8).log(), y_probe)
            fast_memory_ce = F.cross_entropy(fast_memory_logits, memory_y)
            slow_memory_ce = F.cross_entropy(slow_memory_logits, memory_y)
            fused_memory_ce = F.nll_loss(fused_memory.clamp_min(1e-8).log(), memory_y)
            fast_current_accuracy = (
                fast_current_logits.argmax(dim=1) == y_probe
            ).float().mean()
            fused_current_accuracy = (
                fused_current.argmax(dim=1) == y_probe
            ).float().mean()
            fast_memory_accuracy = (
                fast_memory_logits.argmax(dim=1) == memory_y
            ).float().mean()
            fused_memory_accuracy = (
                fused_memory.argmax(dim=1) == memory_y
            ).float().mean()
        if was_training:
            strategy.model.train()

        current_delta = float((fused_current_ce - fast_current_ce).cpu())
        memory_gain = float((fast_memory_ce - fused_memory_ce).cpu())
        mean_accuracy_delta = 0.5 * float(
            (fused_current_accuracy - fast_current_accuracy).cpu()
            + (fused_memory_accuracy - fast_memory_accuracy).cpu()
        )
        self.slow_fast_state_audit_history.append(
            {
                "experience": float(strategy.experience.current_experience),
                "iteration": float(iteration),
                "slow_state_updates": float(self._slow_state_updates),
                "fast_current_ce": float(fast_current_ce.cpu()),
                "slow_current_ce": float(slow_current_ce.cpu()),
                "fusion_current_ce": float(fused_current_ce.cpu()),
                "fusion_current_ce_delta": current_delta,
                "fast_memory_ce": float(fast_memory_ce.cpu()),
                "slow_memory_ce": float(slow_memory_ce.cpu()),
                "fusion_memory_ce": float(fused_memory_ce.cpu()),
                "slow_memory_ce_improvement": float(
                    (fast_memory_ce - slow_memory_ce).cpu()
                ),
                "fusion_memory_ce_improvement": memory_gain,
                "fusion_current_accuracy_delta": float(
                    (fused_current_accuracy - fast_current_accuracy).cpu()
                ),
                "fusion_memory_accuracy_delta": float(
                    (fused_memory_accuracy - fast_memory_accuracy).cpu()
                ),
                "fusion_mean_accuracy_delta": mean_accuracy_delta,
                "safe_and_memory_useful": float(
                    current_delta <= 0.01
                    and memory_gain > 0.0
                    and mean_accuracy_delta >= 0.0
                ),
            }
        )

    def _one_step_model_copy(self, strategy, loss: torch.Tensor):
        """Construct an SGD first-order counterfactual without mutating state."""
        params = tuple(param for param in strategy.model.parameters() if param.requires_grad)
        if not params:
            return None
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        candidate = copy.deepcopy(strategy.model).to(strategy.device)
        lr = float(strategy.optimizer.param_groups[0]["lr"])
        with torch.no_grad():
            candidate_params = tuple(
                param for param in candidate.parameters() if param.requires_grad
            )
            for param, grad in zip(candidate_params, grads):
                if grad is not None:
                    param.add_(grad.detach(), alpha=-lr)
        candidate.eval()
        return candidate

    def _select_plasticity_replay_scale(
        self, strategy, *, per_sample_loss: torch.Tensor
    ) -> float:
        """Choose the largest replay mass that preserves current progress.

        This is the applied D8 controller, unlike D7's diagnostic-only audit.
        At a fixed update cadence it evaluates the *same normalized objective*
        used by the real uniform replay update on a held-out current-stream
        probe.  The selected loss mass is then retained until the next check.
        No task identifier, boundary label, class-count rule, or persistent
        storage is used by the decision.
        """
        if not self.plasticity_budget_control:
            return 1.0

        iteration = int(strategy.clock.train_iterations)
        if iteration % self.plasticity_budget_every != 0:
            return self._plasticity_replay_scale

        batch_sizes = getattr(getattr(strategy, "dataloader", None), "batch_sizes", ())
        current_n = int(
            getattr(strategy, "rbcl_current_batch_size", batch_sizes[0] if batch_sizes else 0)
        )
        total_n = int(per_sample_loss.numel())
        if current_n <= 0 or current_n >= total_n:
            return self._plasticity_replay_scale
        probe = self._next_current_validation_probe(strategy)
        if probe is None:
            return self._plasticity_replay_scale

        x_probe, y_probe, probe_task_ids = probe
        memory_x = strategy.mb_x[current_n:]
        memory_y = strategy.mb_y.long()[current_n:]
        task_ids = (
            strategy.mbatch[-1]
            if strategy.mbatch is not None and len(strategy.mbatch) >= 3
            else None
        )
        memory_task_ids = task_ids[current_n:] if task_ids is not None else None

        current_loss = per_sample_loss[:current_n].mean()
        current_only_model = self._one_step_model_copy(strategy, current_loss)
        if current_only_model is None:
            return self._plasticity_replay_scale
        with torch.no_grad():
            current_ce_before = F.cross_entropy(
                avalanche_forward(strategy.model, x_probe, probe_task_ids), y_probe
            )
            current_ce_current_only = F.cross_entropy(
                avalanche_forward(current_only_model, x_probe, probe_task_ids), y_probe
            )
            memory_ce_current_only = F.cross_entropy(
                avalanche_forward(current_only_model, memory_x, memory_task_ids), memory_y
            )
        current_only_progress = float(
            (current_ce_before - current_ce_current_only).cpu()
        )

        selected_alpha = 0.0
        selected_progress = 0.0
        selected_memory_gain = 0.0
        evaluated = 0
        safe_candidates = 0
        if current_only_progress > 0.0:
            for alpha in sorted(
                (value for value in self.counterfactual_alpha_values if value > 0.0),
                reverse=True,
            ):
                # This normalized group objective exactly matches the actual
                # uniform replay loss after the selected replay group mass is
                # applied below.
                mixed_loss = (
                    per_sample_loss[:current_n].mean()
                    + alpha * per_sample_loss[current_n:].mean()
                ) / (1.0 + alpha)
                candidate = self._one_step_model_copy(strategy, mixed_loss)
                if candidate is None:
                    continue
                evaluated += 1
                with torch.no_grad():
                    current_ce_mixed = F.cross_entropy(
                        avalanche_forward(candidate, x_probe, probe_task_ids), y_probe
                    )
                    memory_ce_mixed = F.cross_entropy(
                        avalanche_forward(candidate, memory_x, memory_task_ids), memory_y
                    )
                mixed_progress = float((current_ce_before - current_ce_mixed).cpu())
                memory_gain = float((memory_ce_current_only - memory_ce_mixed).cpu())
                keeps_budget = mixed_progress >= self.counterfactual_rho * current_only_progress
                useful = memory_gain > 0.0
                if keeps_budget and useful:
                    safe_candidates += 1
                    if selected_alpha == 0.0:
                        selected_alpha = float(alpha)
                        selected_progress = mixed_progress
                        selected_memory_gain = memory_gain
                del candidate

        self._plasticity_replay_scale = selected_alpha
        self.plasticity_budget_history.append(
            {
                "experience": float(strategy.experience.current_experience),
                "iteration": float(iteration),
                "current_only_progress": current_only_progress,
                "selected_alpha": selected_alpha,
                "selected_mixed_progress": selected_progress,
                "selected_memory_ce_improvement": selected_memory_gain,
                "positive_current_progress": float(current_only_progress > 0.0),
                "safe_candidate_count": float(safe_candidates),
                "evaluated_candidate_count": float(evaluated),
            }
        )
        del current_only_model
        return selected_alpha

    def _record_counterfactual_audit(
        self, strategy, per_sample_loss: torch.Tensor, risk: torch.Tensor
    ) -> None:
        """Audit whether priority replay has separable benefit before controlling it.

        This deliberately does *not* constrain the actual update.  At sparse
        iterations it compares two one-step SGD counterfactuals with identical
        current/replay group mass: Uniform replay and replay-group priority.
        A later feedback controller is justified only if priority updates occur
        that both reduce memory CE and do not raise current held-out CE.
        """
        if not self.counterfactual_audit:
            return
        iteration = int(strategy.clock.train_iterations)
        if iteration % self.counterfactual_audit_every != 0:
            return
        batch_sizes = getattr(getattr(strategy, "dataloader", None), "batch_sizes", ())
        current_n = int(
            getattr(strategy, "rbcl_current_batch_size", batch_sizes[0] if batch_sizes else 0)
        )
        total_n = int(per_sample_loss.numel())
        if current_n <= 0 or current_n >= total_n:
            return
        probe = self._next_current_validation_probe(strategy)
        if probe is None:
            return

        uniform_loss = per_sample_loss.mean()
        priority_weights = self._replay_group_weights(risk, strategy)
        priority_loss = self._weighted_mean(per_sample_loss, priority_weights)
        uniform_model = self._one_step_model_copy(strategy, uniform_loss)
        priority_model = self._one_step_model_copy(strategy, priority_loss)
        if uniform_model is None or priority_model is None:
            return

        x_probe, y_probe, probe_task_ids = probe
        memory_x = strategy.mb_x[current_n:]
        memory_y = strategy.mb_y.long()[current_n:]
        task_ids = (
            strategy.mbatch[-1]
            if strategy.mbatch is not None and len(strategy.mbatch) >= 3
            else None
        )
        memory_task_ids = task_ids[current_n:] if task_ids is not None else None
        with torch.no_grad():
            uniform_current_ce = F.cross_entropy(
                avalanche_forward(uniform_model, x_probe, probe_task_ids), y_probe
            )
            priority_current_ce = F.cross_entropy(
                avalanche_forward(priority_model, x_probe, probe_task_ids), y_probe
            )
            uniform_memory_ce = F.cross_entropy(
                avalanche_forward(uniform_model, memory_x, memory_task_ids), memory_y
            )
            priority_memory_ce = F.cross_entropy(
                avalanche_forward(priority_model, memory_x, memory_task_ids), memory_y
            )
        current_delta = float((priority_current_ce - uniform_current_ce).cpu())
        memory_delta = float((priority_memory_ce - uniform_memory_ce).cpu())
        self.counterfactual_audit_history.append(
            {
                "experience": float(strategy.experience.current_experience),
                "iteration": float(iteration),
                "current_validation_ce_priority_minus_uniform": current_delta,
                "memory_ce_priority_minus_uniform": memory_delta,
                "safe_and_useful": float(current_delta <= 0.0 and memory_delta < 0.0),
            }
        )
        del uniform_model, priority_model

    def _record_counterfactual_alpha_audit(
        self, strategy, per_sample_loss: torch.Tensor
    ) -> None:
        """Record a non-mutating current-only versus replay-strength audit.

        The real strategy update remains exactly the configured baseline update.
        At sparse iterations this method only clones the model and asks whether
        each fixed replay strength preserves at least ``rho`` of the held-out
        current-class progress achieved by a current-only update.  It is a
        feasibility measurement, not a controller and not evidence of later
        new-class generalisation by itself.
        """
        if not self.counterfactual_alpha_audit:
            return
        iteration = int(strategy.clock.train_iterations)
        if iteration % self.counterfactual_audit_every != 0:
            return
        batch_sizes = getattr(getattr(strategy, "dataloader", None), "batch_sizes", ())
        current_n = int(
            getattr(strategy, "rbcl_current_batch_size", batch_sizes[0] if batch_sizes else 0)
        )
        total_n = int(per_sample_loss.numel())
        if current_n <= 0 or current_n >= total_n:
            return
        probe = self._next_current_validation_probe(strategy)
        if probe is None:
            return

        current_loss = per_sample_loss[:current_n].mean()
        replay_loss = per_sample_loss[current_n:].mean()
        current_only_model = self._one_step_model_copy(strategy, current_loss)
        if current_only_model is None:
            return

        x_probe, y_probe, probe_task_ids = probe
        memory_x = strategy.mb_x[current_n:]
        memory_y = strategy.mb_y.long()[current_n:]
        task_ids = (
            strategy.mbatch[-1]
            if strategy.mbatch is not None and len(strategy.mbatch) >= 3
            else None
        )
        memory_task_ids = task_ids[current_n:] if task_ids is not None else None
        with torch.no_grad():
            current_ce_before = F.cross_entropy(
                avalanche_forward(strategy.model, x_probe, probe_task_ids), y_probe
            )
            current_ce_current_only = F.cross_entropy(
                avalanche_forward(current_only_model, x_probe, probe_task_ids), y_probe
            )
            memory_ce_current_only = F.cross_entropy(
                avalanche_forward(current_only_model, memory_x, memory_task_ids), memory_y
            )

        current_only_progress = float(
            (current_ce_before - current_ce_current_only).cpu()
        )
        for alpha in self.counterfactual_alpha_values:
            if alpha == 0.0:
                candidate = current_only_model
            else:
                candidate = self._one_step_model_copy(
                    strategy, current_loss + alpha * replay_loss
                )
            if candidate is None:
                continue
            with torch.no_grad():
                current_ce_mixed = F.cross_entropy(
                    avalanche_forward(candidate, x_probe, probe_task_ids), y_probe
                )
                memory_ce_mixed = F.cross_entropy(
                    avalanche_forward(candidate, memory_x, memory_task_ids), memory_y
                )
            mixed_progress = float((current_ce_before - current_ce_mixed).cpu())
            memory_improvement = float((memory_ce_current_only - memory_ce_mixed).cpu())
            keeps_progress_budget = (
                current_only_progress > 0.0
                and mixed_progress >= self.counterfactual_rho * current_only_progress
            )
            self.counterfactual_alpha_audit_history.append(
                {
                    "experience": float(strategy.experience.current_experience),
                    "iteration": float(iteration),
                    "alpha": float(alpha),
                    "current_ce_before": float(current_ce_before.cpu()),
                    "current_only_progress": current_only_progress,
                    "mixed_progress": mixed_progress,
                    "progress_deficit": current_only_progress - mixed_progress,
                    "memory_ce_improvement_vs_current_only": memory_improvement,
                    "keeps_progress_budget": float(keeps_progress_budget),
                    "safe_and_memory_useful": float(
                        keeps_progress_budget and memory_improvement > 0.0
                    ),
                }
            )
            if candidate is not current_only_model:
                del candidate
        del current_only_model

    @staticmethod
    def _loss_gradient_vector(loss: torch.Tensor, params) -> torch.Tensor:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        flattened = []
        for param, grad in zip(params, grads):
            if grad is None:
                flattened.append(torch.zeros_like(param).reshape(-1))
            else:
                flattened.append(grad.detach().reshape(-1))
        return torch.cat(flattened)

    def _gradient_pair_stats(
        self, current_grad: torch.Tensor, memory_grad: torch.Tensor
    ) -> Dict[str, float]:
        current_norm = current_grad.norm()
        memory_norm = memory_grad.norm()
        denom = (current_norm * memory_norm).clamp_min(self.eps)
        dot = torch.dot(current_grad, memory_grad)
        return {
            "cosine": float((dot / denom).cpu()),
            "conflict": float((dot < 0).float().cpu()),
        }

    def _distillation_vector(
        self,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample KL used for teacher-student disagreement and distillation."""
        temp = self.distill_temperature
        student_log_probs = F.log_softmax(logits / temp, dim=1)
        teacher_probs = F.softmax(teacher_logits / temp, dim=1)
        return (
            F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1)
            * temp
            * temp
        )

    def _weighted_mean(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        denom = weights.sum().clamp_min(self.eps)
        return (values * weights).sum() / denom

    def _normalize(self, values: torch.Tensor) -> torch.Tensor:
        values = values.detach()
        spread = values.max() - values.min()
        if spread.abs() <= self.eps:
            return torch.ones_like(values)
        return (values - values.min()) / (spread + self.eps)

    def _ensure_prior(self, num_classes: int, device) -> None:
        if self.class_prior is None:
            self.class_prior = torch.ones(num_classes, device=device)
        elif self.class_prior.numel() < num_classes:
            extra = torch.ones(num_classes - self.class_prior.numel(), device=device)
            self.class_prior = torch.cat([self.class_prior.to(device), extra])
        else:
            self.class_prior = self.class_prior.to(device)

    @torch.no_grad()
    def _update_class_prior(self, targets: torch.Tensor, ce_vec: torch.Tensor) -> None:
        """Update P_t(z): classes with larger historical loss get higher cost."""
        if self.class_prior is None:
            return
        for cls in targets.unique():
            cls_id = int(cls.item())
            mask = targets == cls
            observed = 1.0 + float(ce_vec[mask].mean().detach().cpu())
            old = float(self.class_prior[cls_id].detach().cpu())
            self.class_prior[cls_id] = (
                self.prior_momentum * old + (1.0 - self.prior_momentum) * observed
            )

    def _record_step(
        self,
        strategy,
        instability,
        consequence,
        risk,
        weights,
        ce_vec,
        correct_vec,
    ) -> None:
        exp_id = getattr(strategy.experience, "current_experience", -1)
        high_low = self._high_low_stats(risk.detach(), ce_vec.detach(), correct_vec.detach())
        self.history.append(
            {
                "experience": int(exp_id),
                "iteration": int(strategy.clock.train_iterations),
                "instability_mean": float(instability.detach().mean().cpu()),
                "consequence_mean": float(consequence.detach().mean().cpu()),
                "risk_mean": float(risk.detach().mean().cpu()),
                "risk_max": float(risk.detach().max().cpu()),
                "weight_mean": float(weights.detach().mean().cpu()),
                "active_ratio": float((weights.detach() > 0).float().mean().cpu()),
                **high_low,
                "teacher_forward_calls": float(self.teacher_forward_calls),
                "teacher_forward_examples": float(self.teacher_forward_examples),
            }
        )

    def _high_low_stats(
        self,
        risk: torch.Tensor,
        ce_vec: torch.Tensor,
        correct_vec: torch.Tensor,
    ) -> Dict[str, float]:
        """Track high-risk vs low-risk training behavior for interpretation."""
        n = int(risk.numel())
        if n == 0:
            return {}
        k = max(1, int(math.ceil(n * self.high_low_fraction)))
        high_idx = torch.topk(risk, k=k, largest=True).indices
        low_idx = torch.topk(risk, k=k, largest=False).indices
        return {
            "high_risk_mean": float(risk[high_idx].mean().cpu()),
            "low_risk_mean": float(risk[low_idx].mean().cpu()),
            "high_risk_loss": float(ce_vec[high_idx].mean().cpu()),
            "low_risk_loss": float(ce_vec[low_idx].mean().cpu()),
            "high_risk_acc": float(correct_vec[high_idx].mean().cpu()),
            "low_risk_acc": float(correct_vec[low_idx].mean().cpu()),
        }

    def _aggregate_history(self) -> Dict[str, object]:
        if not self.history:
            return {}

        numeric_keys = [
            "instability_mean",
            "consequence_mean",
            "risk_mean",
            "risk_max",
            "weight_mean",
            "active_ratio",
            "high_risk_mean",
            "low_risk_mean",
            "high_risk_loss",
            "low_risk_loss",
            "high_risk_acc",
            "low_risk_acc",
        ]
        aggregate = {}
        for key in numeric_keys:
            values = [entry[key] for entry in self.history if key in entry]
            if values:
                aggregate[key] = sum(values) / len(values)

        by_exp: Dict[int, Dict[str, object]] = {}
        for entry in self.history:
            exp_id = int(entry["experience"])
            exp_bucket = by_exp.setdefault(exp_id, {"count": 0, "sums": {}})
            exp_bucket["count"] += 1
            sums = exp_bucket["sums"]
            for key in numeric_keys:
                if key in entry:
                    sums[key] = sums.get(key, 0.0) + entry[key]

        per_experience = {}
        for exp_id, bucket in by_exp.items():
            count = max(1, int(bucket["count"]))
            per_experience[str(exp_id)] = {
                key: value / count for key, value in bucket["sums"].items()
            }

        aggregate["per_experience"] = per_experience
        return aggregate

    def _aggregate_gradient_audit(self) -> Dict[str, object]:
        if not self.gradient_audit_history:
            return {"enabled": self.gradient_audit, "count": 0, "per_experience": {}}
        keys = [
            "current_memory_cosine",
            "current_memory_conflict",
            "risk_weighted_class_cosine",
            "risk_weighted_class_conflict_fraction",
            "class_conflict_fraction",
            "memory_class_count",
        ]
        aggregate = {
            "enabled": self.gradient_audit,
            "every": self.gradient_audit_every,
            "count": len(self.gradient_audit_history),
        }
        aggregate.update(
            {
                key: sum(entry[key] for entry in self.gradient_audit_history)
                / len(self.gradient_audit_history)
                for key in keys
            }
        )
        per_experience: Dict[str, Dict[str, float]] = {}
        for exp_id in sorted({int(entry["experience"]) for entry in self.gradient_audit_history}):
            entries = [entry for entry in self.gradient_audit_history if int(entry["experience"]) == exp_id]
            per_experience[str(exp_id)] = {
                "count": float(len(entries)),
                **{key: sum(entry[key] for entry in entries) / len(entries) for key in keys},
            }
        aggregate["per_experience"] = per_experience
        return aggregate

    def _aggregate_replay_geometry_audit(self) -> Dict[str, object]:
        if not self.replay_geometry_audit_history:
            return {
                "enabled": self.replay_geometry_audit,
                "count": 0,
                "history": [],
            }
        entries = self.replay_geometry_audit_history
        keys = [
            "memory_sample_count",
            "mean_sample_cosine",
            "compatible_fraction",
            "top_quartile_cosine",
            "bottom_quartile_cosine",
            "memory_ce_mean",
        ]
        return {
            "enabled": self.replay_geometry_audit,
            "every": self.replay_geometry_audit_every,
            "count": len(entries),
            **{key: sum(entry[key] for entry in entries) / len(entries) for key in keys},
            "history": entries,
        }

    def _aggregate_conflict_residual_audit(self) -> Dict[str, object]:
        if not self.conflict_residual_audit_history:
            return {"enabled": self.conflict_residual_audit, "count": 0, "history": []}
        entries = self.conflict_residual_audit_history
        keys = ["current_only_progress", "uniform_progress", "residual_progress", "uniform_memory_gain", "residual_memory_gain", "repair_progress", "repair_memory_gain", "residual_keeps_budget", "residual_beats_uniform_progress", "residual_keeps_memory_gain", "repair_keeps_budget", "repair_improves_memory_over_residual", "repair_pareto", "memory_conflict_coefficient"]
        return {"enabled": self.conflict_residual_audit, "every": self.conflict_residual_audit_every, "rho": self.counterfactual_rho, "count": len(entries), **{k: sum(e[k] for e in entries) / len(entries) for k in keys}, "history": entries}

    def _aggregate_slow_fast_state_audit(self) -> Dict[str, object]:
        if not self.slow_fast_state_audit_history:
            return {
                "enabled": self.slow_fast_state_audit,
                "count": 0,
                "history": [],
            }
        entries = self.slow_fast_state_audit_history
        keys = [
            "slow_state_updates",
            "fast_current_ce",
            "slow_current_ce",
            "fusion_current_ce",
            "fusion_current_ce_delta",
            "fast_memory_ce",
            "slow_memory_ce",
            "fusion_memory_ce",
            "slow_memory_ce_improvement",
            "fusion_memory_ce_improvement",
            "fusion_current_accuracy_delta",
            "fusion_memory_accuracy_delta",
            "fusion_mean_accuracy_delta",
            "safe_and_memory_useful",
        ]
        return {
            "enabled": self.slow_fast_state_audit,
            "every": self.slow_fast_state_audit_every,
            "ema_decay": self.slow_fast_ema_decay,
            "count": len(entries),
            **{
                key: sum(entry[key] for entry in entries) / len(entries)
                for key in keys
            },
            "history": entries,
        }

    def _aggregate_counterfactual_audit(self) -> Dict[str, object]:
        """Summarize the non-mutating separability diagnostic."""
        if not self.counterfactual_audit_history:
            return {
                "enabled": self.counterfactual_audit,
                "count": 0,
                "per_experience": {},
            }
        keys = [
            "current_validation_ce_priority_minus_uniform",
            "memory_ce_priority_minus_uniform",
            "safe_and_useful",
        ]
        entries = self.counterfactual_audit_history
        aggregate: Dict[str, object] = {
            "enabled": self.counterfactual_audit,
            "every": self.counterfactual_audit_every,
            "probe_size": self.counterfactual_probe_size,
            "count": len(entries),
            **{key: sum(item[key] for item in entries) / len(entries) for key in keys},
        }
        per_experience: Dict[str, Dict[str, float]] = {}
        for exp_id in sorted({int(item["experience"]) for item in entries}):
            group = [item for item in entries if int(item["experience"]) == exp_id]
            per_experience[str(exp_id)] = {
                "count": float(len(group)),
                **{key: sum(item[key] for item in group) / len(group) for key in keys},
            }
        aggregate["per_experience"] = per_experience
        aggregate["history"] = entries
        return aggregate

    def _aggregate_counterfactual_alpha_audit(self) -> Dict[str, object]:
        """Summarize the current-only replay-strength feasibility measurement."""
        if not self.counterfactual_alpha_audit_history:
            return {
                "enabled": self.counterfactual_alpha_audit,
                "count": 0,
                "per_alpha": {},
            }
        keys = [
            "current_only_progress",
            "mixed_progress",
            "progress_deficit",
            "memory_ce_improvement_vs_current_only",
            "keeps_progress_budget",
            "safe_and_memory_useful",
        ]
        entries = self.counterfactual_alpha_audit_history
        aggregate: Dict[str, object] = {
            "enabled": self.counterfactual_alpha_audit,
            "every": self.counterfactual_audit_every,
            "probe_size": self.counterfactual_probe_size,
            "rho": self.counterfactual_rho,
            "alpha_values": list(self.counterfactual_alpha_values),
            "count": len(entries),
            **{key: sum(entry[key] for entry in entries) / len(entries) for key in keys},
        }
        per_alpha: Dict[str, Dict[str, float]] = {}
        for alpha in self.counterfactual_alpha_values:
            alpha_entries = [entry for entry in entries if entry["alpha"] == alpha]
            if alpha_entries:
                per_alpha[str(alpha)] = {
                    key: sum(entry[key] for entry in alpha_entries) / len(alpha_entries)
                    for key in keys
                }
        aggregate["per_alpha"] = per_alpha
        aggregate["history"] = entries
        return aggregate

    def _aggregate_plasticity_budget_control(self) -> Dict[str, object]:
        """Summarize applied replay-strength decisions for reproducibility."""
        if not self.plasticity_budget_history:
            return {
                "enabled": self.plasticity_budget_control,
                "count": 0,
                "history": [],
            }
        entries = self.plasticity_budget_history
        keys = [
            "current_only_progress",
            "selected_alpha",
            "selected_mixed_progress",
            "selected_memory_ce_improvement",
            "positive_current_progress",
            "safe_candidate_count",
            "evaluated_candidate_count",
        ]
        return {
            "enabled": self.plasticity_budget_control,
            "every": self.plasticity_budget_every,
            "rho": self.counterfactual_rho,
            "alpha_values": list(self.counterfactual_alpha_values),
            "count": len(entries),
            **{key: sum(entry[key] for entry in entries) / len(entries) for key in keys},
            "nonzero_selection_fraction": sum(
                entry["selected_alpha"] > 0.0 for entry in entries
            )
            / len(entries),
            "history": entries,
        }

    @torch.no_grad()
    def _record_boundary_debt_before_training(self, strategy) -> None:
        """Measure pre-update new-to-old confusion without changing training.

        A record is one (current class, already-seen class) pair.  Its score is
        the probability mass that the pre-update classifier assigns to the old
        class on held-out examples of the current class.  The paired post-update
        outcome is populated in ``after_training_exp`` on the exact same probe.
        """
        self._boundary_debt_pending = []
        if not (
            self.boundary_debt_audit
            or self.boundary_debt_replay
            or self.boundary_repair
        ):
            return
        if self._boundary_debt_loader is None:
            return
        if not self._seen_validation_classes:
            return

        was_training = strategy.model.training
        strategy.model.eval()
        old_classes = sorted(self._seen_validation_classes)
        for batch in self._boundary_debt_loader:
            x, y = batch[0].to(strategy.device), batch[1].long().to(strategy.device)
            task_ids = batch[2].to(strategy.device) if len(batch) >= 3 else None
            probabilities = avalanche_forward(strategy.model, x, task_ids).softmax(dim=1)
            for new_class in y.unique().tolist():
                mask = y == int(new_class)
                for old_class in old_classes:
                    self._boundary_debt_pending.append(
                        {
                            "experience": float(self._boundary_debt_experience_id),
                            "new_class": float(new_class),
                            "old_class": float(old_class),
                            "pre_old_probability": float(
                                probabilities[mask, old_class].mean().item()
                            ),
                            "probe_examples": float(mask.sum().item()),
                        }
                    )
        if was_training:
            strategy.model.train()
        self._set_class_boundary_debt()

    def _set_class_boundary_debt(self) -> None:
        """Collapse current new-to-old pair debts to an old-class priority."""
        by_class: Dict[int, List[float]] = {}
        for record in self._boundary_debt_pending:
            by_class.setdefault(int(record["old_class"]), []).append(
                float(record["pre_old_probability"])
            )
        self.class_boundary_debt = {
            class_id: sum(values) / len(values)
            for class_id, values in by_class.items()
            if values
        }
        by_pair: Dict[tuple[int, int], List[float]] = {}
        for record in self._boundary_debt_pending:
            key = (int(record["new_class"]), int(record["old_class"]))
            by_pair.setdefault(key, []).append(float(record["pre_old_probability"]))
        self.boundary_debt_pairs = {
            pair: sum(values) / len(values) for pair, values in by_pair.items() if values
        }
        if self.boundary_debt_replay and self.class_boundary_debt:
            self.boundary_debt_replay_history.append(
                {
                    "experience": self._boundary_debt_experience_id,
                    "class_boundary_debt": {
                        str(class_id): value
                        for class_id, value in sorted(self.class_boundary_debt.items())
                    },
                    "temperature": self.boundary_debt_temperature,
                    "mix": self.boundary_debt_mix,
                    "replay_group_mass_preserved": True,
                }
            )

    @torch.no_grad()
    def _record_boundary_debt_after_training(self, strategy) -> None:
        """Attach post-update cross-task probability and error to audit pairs."""
        if not (
            self.boundary_debt_audit
            or self.boundary_debt_replay
            or self.boundary_repair
        ):
            return
        if self._boundary_debt_loader is None:
            return

        was_training = strategy.model.training
        strategy.model.eval()
        grouped: Dict[int, Dict[str, torch.Tensor]] = {}
        current_classes: set[int] = set()
        for batch in self._boundary_debt_loader:
            x, y = batch[0].to(strategy.device), batch[1].long().to(strategy.device)
            task_ids = batch[2].to(strategy.device) if len(batch) >= 3 else None
            probabilities = avalanche_forward(strategy.model, x, task_ids).softmax(dim=1)
            predictions = probabilities.argmax(dim=1)
            for new_class in y.unique().tolist():
                new_class = int(new_class)
                current_classes.add(new_class)
                mask = y == new_class
                grouped[new_class] = {
                    "probabilities": probabilities[mask],
                    "predictions": predictions[mask],
                }

        for record in self._boundary_debt_pending:
            new_class = int(record["new_class"])
            old_class = int(record["old_class"])
            values = grouped.get(new_class)
            if values is None:
                continue
            record["post_old_probability"] = float(
                values["probabilities"][:, old_class].mean().item()
            )
            record["post_cross_error"] = float(
                (values["predictions"] == old_class).float().mean().item()
            )
            self.boundary_debt_audit_history.append(record)

        self._seen_validation_classes.update(current_classes)
        self._boundary_debt_pending = []
        if was_training:
            strategy.model.train()

    @torch.no_grad()
    def _collect_class_examples(self, dataset, classes: set[int]) -> Dict[int, torch.Tensor]:
        """Collect a fixed number of train samples per class for head repair."""
        collected: Dict[int, List[torch.Tensor]] = {class_id: [] for class_id in classes}
        if dataset is None or not classes:
            return {}
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
        for batch in loader:
            x, y = batch[0], batch[1].long()
            for class_id in classes:
                remaining = self.boundary_repair_samples_per_class - sum(
                    part.shape[0] for part in collected[class_id]
                )
                if remaining <= 0:
                    continue
                index = (y == class_id).nonzero(as_tuple=True)[0][:remaining]
                if index.numel():
                    collected[class_id].append(x[index].detach().cpu())
            if all(
                sum(part.shape[0] for part in values)
                >= self.boundary_repair_samples_per_class
                for values in collected.values()
            ):
                break
        return {
            class_id: torch.cat(values, dim=0)
            for class_id, values in collected.items()
            if values
        }

    def _replay_memory_dataset(self, strategy):
        for plugin in getattr(strategy, "plugins", []):
            policy = getattr(plugin, "storage_policy", None)
            if policy is not None and hasattr(policy, "buffer"):
                return policy.buffer
        return None

    def _run_boundary_repair(self, strategy) -> None:
        """Repair only high-debt class-pair margins after Uniform main training.

        The feature extractor is frozen. Each candidate head update is reverted
        when it increases CE on a fixed current-training reference batch by
        more than the configured tolerance. Thus Boundary-Debt decides *which
        pairwise boundary to repair*, not the global old/new replay mass.
        """
        if not (self.boundary_repair or self.boundary_feature_repair):
            return
        ranked_pairs = sorted(
            self.boundary_debt_pairs.items(), key=lambda item: item[1], reverse=True
        )[: self.boundary_repair_pairs]
        if not ranked_pairs:
            return
        # Protect every new class in the current experience, not only the
        # classes that occur in the selected high-debt pairs.
        current_classes = {
            new_class for new_class, _ in self.boundary_debt_pairs.keys()
        }
        old_classes = {old_class for (_, old_class), _ in ranked_pairs}
        current_examples = self._collect_class_examples(
            strategy.experience.dataset, current_classes
        )
        old_examples = self._collect_class_examples(
            self._replay_memory_dataset(strategy), old_classes
        )
        ranked_pairs = [
            item
            for item in ranked_pairs
            if item[0][0] in current_examples and item[0][1] in old_examples
        ]
        if not ranked_pairs:
            return

        full_model = self.boundary_feature_repair
        if full_model:
            repair_params = [
                param for param in strategy.model.parameters() if param.requires_grad
            ]
        else:
            try:
                _, head = get_last_fc_layer(strategy.model)
            except ValueError:
                return
            repair_params = [param for param in head.parameters() if param.requires_grad]
        if not repair_params:
            return
        reference_x = torch.cat(
            [current_examples[class_id] for class_id in sorted(current_examples)], dim=0
        ).to(strategy.device)
        reference_y = torch.cat(
            [
                torch.full(
                    (current_examples[class_id].shape[0],), class_id, dtype=torch.long
                )
                for class_id in sorted(current_examples)
            ],
            dim=0,
        ).to(strategy.device)
        was_training = strategy.model.training
        original_requires_grad = [param.requires_grad for param in strategy.model.parameters()]
        for param in strategy.model.parameters():
            param.requires_grad_(False)
        for param in repair_params:
            param.requires_grad_(True)
        strategy.model.eval()
        optimizer = torch.optim.SGD(
            repair_params, lr=float(strategy.optimizer.param_groups[0]["lr"])
        )
        with torch.no_grad():
            reference_ce_before = float(
                F.cross_entropy(strategy.model(reference_x), reference_y).item()
            )
        accepted = 0
        rejected = 0
        attempted = []
        for (new_class, old_class), debt in ranked_pairs:
            new_x = current_examples[new_class].to(strategy.device)
            old_x = old_examples[old_class].to(strategy.device)
            x = torch.cat([new_x, old_x], dim=0)
            y = torch.cat(
                [
                    torch.full((new_x.shape[0],), new_class, device=strategy.device),
                    torch.full((old_x.shape[0],), old_class, device=strategy.device),
                ]
            ).long()
            repair_module = strategy.model if full_model else head
            snapshot = {
                name: value.detach().clone()
                for name, value in repair_module.state_dict().items()
            }
            optimizer.zero_grad(set_to_none=True)
            logits = strategy.model(x)
            ce = F.cross_entropy(logits, y)
            new_margin = logits[: new_x.shape[0], new_class] - logits[: new_x.shape[0], old_class]
            old_margin = logits[new_x.shape[0] :, old_class] - logits[new_x.shape[0] :, new_class]
            margin_loss = torch.cat(
                [
                    F.relu(self.boundary_repair_margin - new_margin),
                    F.relu(self.boundary_repair_margin - old_margin),
                ]
            ).mean()
            (ce + self.boundary_repair_lambda * margin_loss).backward()
            optimizer.step()
            with torch.no_grad():
                reference_ce_after = float(
                    F.cross_entropy(strategy.model(reference_x), reference_y).item()
                )
            accepted_step = (
                reference_ce_after
                <= reference_ce_before + self.boundary_repair_current_ce_tolerance
            )
            if accepted_step:
                accepted += 1
            else:
                repair_module.load_state_dict(snapshot)
                rejected += 1
            attempted.append(
                {
                    "new_class": new_class,
                    "old_class": old_class,
                    "debt": debt,
                    "accepted": accepted_step,
                    "reference_ce_after_candidate": reference_ce_after,
                }
            )
        with torch.no_grad():
            reference_ce_after = float(
                F.cross_entropy(strategy.model(reference_x), reference_y).item()
            )
        for param, value in zip(strategy.model.parameters(), original_requires_grad):
            param.requires_grad_(value)
        if was_training:
            strategy.model.train()
        self.boundary_repair_history.append(
            {
                "experience": self._boundary_debt_experience_id,
                "head_only": not full_model,
                "feature_extractor_updated": full_model,
                "replay_count_changed": False,
                "reference_ce_before": reference_ce_before,
                "reference_ce_after": reference_ce_after,
                "accepted_steps": accepted,
                "rejected_steps": rejected,
                "attempted_pairs": attempted,
            }
        )

    def _aggregate_boundary_debt_audit(self) -> Dict[str, object]:
        if not self.boundary_debt_audit_history:
            return {
                "enabled": self.boundary_debt_audit,
                "count": 0,
                "history": [],
            }
        entries = self.boundary_debt_audit_history
        return {
            "enabled": self.boundary_debt_audit,
            "probe_size": self.boundary_debt_probe_size,
            "count": len(entries),
            "mean_pre_old_probability": sum(
                item["pre_old_probability"] for item in entries
            ) / len(entries),
            "mean_post_cross_error": sum(
                item["post_cross_error"] for item in entries
            ) / len(entries),
            "history": entries,
        }

    def summary(self) -> Dict[str, object]:
        """Return paper-facing custom stats for fairness/interpretability tables."""
        prior = None
        if self.class_prior is not None:
            prior = self.class_prior.detach().cpu().tolist()
        aggregate = self._aggregate_history()
        return {
            "budget_mode": self.budget_mode,
            "allocation": self.allocation,
            "budget_ratio": self.budget_ratio,
            "consequence_mode": self.consequence_mode,
            "allocation_scope": self.allocation_scope,
            "gradient_audit": self._aggregate_gradient_audit(),
            "replay_geometry_audit": self._aggregate_replay_geometry_audit(),
            "conflict_residual_audit": self._aggregate_conflict_residual_audit(),
            "slow_fast_state_audit": self._aggregate_slow_fast_state_audit(),
            "counterfactual_audit": self._aggregate_counterfactual_audit(),
            "counterfactual_alpha_audit": self._aggregate_counterfactual_alpha_audit(),
            "plasticity_budget_control": self._aggregate_plasticity_budget_control(),
            "boundary_debt_audit": self._aggregate_boundary_debt_audit(),
            "boundary_debt_replay": {
                "enabled": self.boundary_debt_replay,
                "temperature": self.boundary_debt_temperature,
                "mix": self.boundary_debt_mix,
                "history": self.boundary_debt_replay_history,
            },
            "boundary_repair": {
                "enabled": self.boundary_repair,
                "feature_repair_enabled": self.boundary_feature_repair,
                "pairs": self.boundary_repair_pairs,
                "samples_per_class": self.boundary_repair_samples_per_class,
                "margin": self.boundary_repair_margin,
                "lambda": self.boundary_repair_lambda,
                "current_ce_tolerance": self.boundary_repair_current_ce_tolerance,
                "history": self.boundary_repair_history,
            },
            "instability_mode": self.instability_mode,
            "fusion_mode": self.fusion_mode,
            "instability_lambda": self.instability_lambda,
            "use_instability": self.use_instability,
            "use_consequence": self.use_consequence,
            "use_uncertainty": self.use_uncertainty,
            "use_disagreement": self.use_disagreement,
            "use_prior": self.use_prior,
            "teacher_forward_calls": self.teacher_forward_calls,
            "teacher_forward_examples": self.teacher_forward_examples,
            "risk_mean": aggregate.get("risk_mean"),
            "risk_max": aggregate.get("risk_max"),
            "active_ratio": aggregate.get("active_ratio"),
            "high_low_analysis": {
                "high_risk_loss": aggregate.get("high_risk_loss"),
                "low_risk_loss": aggregate.get("low_risk_loss"),
                "high_risk_acc": aggregate.get("high_risk_acc"),
                "low_risk_acc": aggregate.get("low_risk_acc"),
                "high_risk_mean": aggregate.get("high_risk_mean"),
                "low_risk_mean": aggregate.get("low_risk_mean"),
            },
            "per_experience": aggregate.get("per_experience", {}),
            "class_prior": prior,
            "class_best_accuracy": self.class_best_accuracy,
            "class_validation_accuracy": self.class_validation_accuracy,
            "consequence_history": self.consequence_history,
            "history": self.history,
        }
