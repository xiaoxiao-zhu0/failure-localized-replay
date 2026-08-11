"""Strategy construction and stream runner for RBCL experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision.transforms import Compose, Lambda, RandomCrop, RandomHorizontalFlip

from avalanche.models import SCRModel
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import (
    ClassBalancedBuffer,
    ExperienceBalancedBuffer,
    ReservoirSamplingBuffer,
)
from avalanche.training.supervised import (
    DER,
    EWC,
    ER_ACE,
    JointTraining,
    LwF,
    MIR,
    Naive,
    SCR,
    SynapticIntelligence,
)

from .budgeting import RiskBudgetingPlugin
from .causal_er_ace import CausalERACE
from .causal_prba import PrequentialRiskBudgetedDualHeadCausalERACE
from .counterfactual_accuracy import CausalHybridAccuracyCorrectionAudit
from .counterfactual_mixture import CausalCounterfactualMixtureERACE
from .global_temporal import GlobalTemporalSteadyERACE
from .semantic_anchor import GlobalSemanticAnchorERACE
from .semantic_representative import SemanticRepresentativeERACE, ClassReplayVulnerabilityAuditERACE, SelfReferencedReplayDeteriorationAuditERACE, PersistentSelfReferencedReplayDeteriorationAuditERACE, PersistentSRRDDebtSwapERACE, SelectivePersistentSRRDSwapERACE, ReplayFeatureDualHeadCalibrationERACE, FixedAlphaDualHeadCalibrationERACE, CanonicalOBCDualHeadERACE, RiskBudgetedDualHeadArbitrationERACE, PrequentialRiskBudgetedDualHeadArbitrationERACE, PrequentialCurrentOnlyArbitrationERACE, PrequentialReplayOnlyArbitrationERACE, PrequentialLastAlphaArbitrationERACE, SRRDConsequencePrequentialArbitrationERACE, ParetoGuardedReplayRepairERACE, MemoryCertifiedReplayRepairERACE, PersistentSRRDLossRedistributionERACE, SupportCalibratedPersistentSRRDLossRedistributionERACE, SupportBalancedPersistentSRRDLossRedistributionERACE, TIRPSemanticRepresentativeERACE, FIRPScoreAuditERACE, ClassFIRPERACE, ClassFIRPDebtSwapERACE, ClassFIRPExposureAuditERACE, TIRPSemanticBoundaryERACE, TIRPDecisionConsolidatedERACE, TIRPProxyRelationERACE, TIRPProxyContrastiveERACE, TIRPSemanticRelationERACE, TIRPSemanticRelationGuardERACE, TIRPPrototypeBoundaryRelationERACE, TIRPMaturityNormalizedRelationERACE, TIRPSparseBudgetedRelationERACE
from .memory_audit import ReplayLabelNoisePlugin
from .retention import ConsequenceAwareExperienceBalancedBuffer
from .clock_bridge import ClockBridgeReplayPlugin
from .stream_clock import StreamClockReplayPlugin
from .utils import save_json


def _dataset_family(benchmark_name: str) -> str:
    """Resolve the normalization family used by semantic replay components."""
    key = benchmark_name.lower()
    if "core50" in key:
        return "core50"
    if "tinyimagenet" in key:
        return "tinyimagenet"
    if "cifar100" in key:
        return "cifar100"
    return "cifar10"


def build_strategy(
    name: str,
    *,
    model,
    evaluator,
    device: torch.device,
    lr: float,
    momentum: float,
    train_mb_size: int,
    train_epochs: int,
    eval_mb_size: int,
    mem_size: int,
    replay_mb_size: Optional[int] = None,
    allocation: str = "soft",
    budget_ratio: float = 1.0,
    distill_lambda: float = 0.5,
    distill_temperature: float = 2.0,
    use_instability: bool = True,
    use_consequence: bool = True,
    use_uncertainty: bool = True,
    use_disagreement: bool = True,
    use_prior: bool = True,
    instability_mode: str = "virtual_replay_loss",
    fusion_mode: str = "c_gated",
    instability_lambda: float = 0.25,
    consequence_mode: str = "loss_ema",
    allocation_scope: str = "global",
    gradient_audit: bool = False,
    gradient_audit_every: int = 100,
    counterfactual_audit: bool = False,
    counterfactual_audit_every: int = 200,
    counterfactual_probe_size: int = 64,
    counterfactual_alpha_audit: bool = False,
    counterfactual_alpha_values: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
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
    memory_label_noise_rate: float = 0.0,
    memory_label_noise_seed: int = 0,
    replay_storage: str = "experience",
    retention_mode: str = "experience_balanced",
    retention_strength: float = 1.0,
    stream_clock_samples: int = 1000,
    stream_clock_min_replay_age_samples: int = 0,
    stream_clock_coverage_floor: int = 0,
    stream_clock_coverage_strength: float = 0.0,
    stream_clock_replay_loss_scale: float = 1.0,
    stream_clock_current_uncertainty_lambda: float = 0.0,
    value_coverage_audit: bool = False,
    value_coverage_audit_every: int = 500,
    value_coverage_rho: float = 0.95,
    paired_update_audit: bool = False,
    paired_update_audit_every: int = 512,
    memory_trace_signature: bool = False,
    memory_trace_audit: bool = False,
    ewc_lambda: float = 0.001,
    si_lambda: float = 0.0001,
    lwf_alpha: float = 1.0,
    eval_every: int = -1,
    benchmark_name: str = "",
    num_classes: int = 0,
    tirp_policy_path: str = "",
    tirp_allow_unapproved_policy: bool = False,
    relation_mechanism_audit: bool = False,
    relation_replay_audit: bool = False,
) -> Tuple[object, Optional[RiskBudgetingPlugin]]:
    """Create a baseline or RBCL strategy under a shared protocol."""
    key = name.lower()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = CrossEntropyLoss()
    rbcl_plugin: Optional[RiskBudgetingPlugin] = None
    if replay_mb_size is not None and replay_mb_size <= 0:
        raise ValueError("replay_mb_size must be positive when provided")
    effective_replay_mb_size = min(
        train_mb_size if replay_mb_size is None else replay_mb_size,
        mem_size,
    )

    common = dict(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=train_mb_size,
        train_epochs=train_epochs,
        eval_mb_size=eval_mb_size,
        device=device,
        evaluator=evaluator,
        eval_every=eval_every,
    )

    def semantic_capacities() -> tuple[int, int]:
        semantic = int(round(0.75 * mem_size))
        return semantic, mem_size - semantic
    paired_update_audit_args = dict(
        paired_update_audit=paired_update_audit,
        paired_update_audit_every=paired_update_audit_every,
    )

    if key == "naive":
        return Naive(**common), None

    if key == "replay":
        plugins = [ReplayPlugin(mem_size=mem_size)]
        return Naive(**common, plugins=plugins), None

    if key == "derpp":
        return (
            DER(
                **common,
                mem_size=mem_size,
                batch_size_mem=min(train_mb_size, mem_size),
                alpha=0.1,
                beta=0.5,
            ),
            None,
        )

    if key == "er_ace":
        return (
            ER_ACE(
                **common,
                mem_size=mem_size,
                batch_size_mem=min(train_mb_size, mem_size),
            ),
            None,
        )

    if key in {
        "causal_er_ace",
        "causal_er_ace_trace_control",
        "causal_er_ace_memory_trace",
    }:
        trace_control = key == "causal_er_ace_trace_control"
        full_trace = key == "causal_er_ace_memory_trace"
        return (
            CausalERACE(
                **common,
                **paired_update_audit_args,
                mem_size=mem_size,
                batch_size_mem=min(train_mb_size, mem_size),
                seed=90_000 + mem_size,
                value_coverage_audit=value_coverage_audit,
                value_coverage_audit_every=value_coverage_audit_every,
                value_coverage_rho=value_coverage_rho,
                memory_trace_signature=(
                    memory_trace_signature or trace_control or full_trace
                ),
                memory_trace_audit=memory_trace_audit or full_trace,
            ),
            None,
        )

    if key in {
        "causal_er_ace_prequential_arbitration_noop",
        "causal_er_ace_prequential_arbitration_1",
    }:
        noop = key == "causal_er_ace_prequential_arbitration_noop"
        return (
            PrequentialRiskBudgetedDualHeadCausalERACE(
                **common,
                **paired_update_audit_args,
                mem_size=mem_size,
                batch_size_mem=effective_replay_mb_size,
                seed=90_000 + mem_size,
                calibration_lr_scale=0.0 if noop else 1.0,
                calibration_label_smoothing=0.1,
                calibration_replay_detailed_audit=False,
                memory_trace_signature=memory_trace_signature or noop,
            ),
            None,
        )

    if key == "scr":
        if not all(
            hasattr(model, attribute)
            for attribute in ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "linear")
        ):
            raise ValueError("SCR requires the SlimResNet18 backbone")
        feature_dim = int(model.linear.in_features)
        model.linear = nn.Identity()
        scr_model = SCRModel(
            feature_extractor=model,
            projection=nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, 128),
            ),
            num_classes=num_classes,
        )
        # SCR's crop must preserve the benchmark input resolution.  The
        # original CIFAR-only path hard-coded 32, which makes the augmented
        # replay tensor 32x32 and cannot be concatenated with Tiny ImageNet's
        # 64x64 current tensor.
        scr_crop_size = 64 if "tinyimagenet" in benchmark_name.lower() else 32
        sample_augmentation = Compose(
            [RandomCrop(scr_crop_size, padding=4), RandomHorizontalFlip()]
        )
        augmentation = Compose(
            [
                Lambda(
                    lambda batch: torch.stack(
                        [sample_augmentation(sample) for sample in batch]
                    )
                )
            ]
        )
        return (
            SCR(
                model=scr_model,
                optimizer=SGD(scr_model.parameters(), lr=lr, momentum=momentum),
                augmentations=augmentation,
                mem_size=mem_size,
                temperature=0.1,
                train_mb_size=train_mb_size,
                batch_size_mem=min(train_mb_size, mem_size),
                train_epochs=train_epochs,
                eval_mb_size=eval_mb_size,
                device=device,
                evaluator=evaluator,
                eval_every=eval_every,
            ),
            None,
        )

    if key == "causal_er_ace_cb":
        return (
            CausalERACE(
                **common,
                mem_size=mem_size,
                batch_size_mem=min(train_mb_size, mem_size),
                seed=90_000 + mem_size,
                memory_policy="class_balanced",
                value_coverage_audit=value_coverage_audit,
                value_coverage_audit_every=value_coverage_audit_every,
                value_coverage_rho=value_coverage_rho,
            ),
            None,
        )

    if key == "causal_er_ace_hybrid":
        return (
            CausalERACE(
                **common,
                mem_size=mem_size,
                batch_size_mem=min(train_mb_size, mem_size),
                seed=90_000 + mem_size,
                memory_policy="hybrid",
                value_coverage_audit=value_coverage_audit,
                value_coverage_audit_every=value_coverage_audit_every,
                value_coverage_rho=value_coverage_rho,
            ),
            None,
        )

    if key == "causal_counterfactual_mixture":
        return (
            CausalCounterfactualMixtureERACE(
                **common,
                mem_size=mem_size,
                batch_size_mem=min(train_mb_size, mem_size),
                seed=90_000 + mem_size,
                controller_every=500,
                controller_rho=0.90,
            ),
            None,
        )

    if key == "causal_hybrid_accuracy_audit":
        return (
            CausalHybridAccuracyCorrectionAudit(
                **common,
                mem_size=mem_size,
                batch_size_mem=min(train_mb_size, mem_size),
                seed=90_000 + mem_size,
                audit_every=500,
            ),
            None,
        )

    if key == "causal_hybrid_accuracy_correction":
        return (
            CausalHybridAccuracyCorrectionAudit(
                **common,
                mem_size=mem_size,
                batch_size_mem=min(train_mb_size, mem_size),
                seed=90_000 + mem_size,
                audit_every=500,
                apply_safe_correction=True,
            ),
            None,
        )

    if key in {
        "causal_hybrid_gain_budget_audit",
        "causal_hybrid_gain_budget_correction",
    }:
        return (
            CausalHybridAccuracyCorrectionAudit(
                **common,
                mem_size=mem_size,
                batch_size_mem=min(train_mb_size, mem_size),
                seed=90_000 + mem_size,
                audit_every=500,
                apply_safe_correction=(
                    key == "causal_hybrid_gain_budget_correction"
                ),
                relative_gain_budget=0.005,
            ),
            None,
        )

    if key == "global_temporal_steady":
        return (
            GlobalTemporalSteadyERACE(
                **common,
                mem_size=mem_size,
                batch_size_mem=min(train_mb_size, mem_size),
                seed=90_000 + mem_size,
                renewal_periods=(512, 4096),
            ),
            None,
        )

    if key == "global_semantic_anchor":
        dataset_family = _dataset_family(benchmark_name)
        return (
            GlobalSemanticAnchorERACE(
                **common,
                mem_size=mem_size,
                batch_size_mem=min(train_mb_size, mem_size),
                seed=90_000 + mem_size,
                dataset_family=dataset_family,
                num_classes=num_classes,
                anchor_lambda=0.25,
                anchor_temperature=2.0,
                anchor_refresh_samples=512,
                anchor_ridge=1.0,
            ),
            None,
        )

    if key == "global_semantic_replay_anchor":
        dataset_family = _dataset_family(benchmark_name)
        return (
            GlobalSemanticAnchorERACE(
                **common,
                mem_size=mem_size,
                batch_size_mem=min(train_mb_size, mem_size),
                seed=90_000 + mem_size,
                dataset_family=dataset_family,
                num_classes=num_classes,
                anchor_lambda=0.25,
                anchor_temperature=2.0,
                anchor_refresh_samples=512,
                anchor_ridge=1.0,
                anchor_scope="replay",
            ),
            None,
        )

    if key == "semantic_anchor_compatibility_audit":
        dataset_family = _dataset_family(benchmark_name)
        return (
            GlobalSemanticAnchorERACE(
                **common, mem_size=mem_size, batch_size_mem=min(train_mb_size, mem_size),
                seed=90_000 + mem_size, dataset_family=dataset_family,
                num_classes=num_classes, anchor_lambda=0.0, anchor_temperature=2.0,
                anchor_refresh_samples=512, anchor_ridge=1.0, anchor_scope="replay",
                anchor_audit_only=True, anchor_audit_every=512,
            ), None,
        )

    if key in {"semantic_proto_coverage", "semantic_proto_hybrid"}:
        family = _dataset_family(benchmark_name)
        return (SemanticRepresentativeERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=(key=="semantic_proto_hybrid")), None)

    if key in {
        "semantic_proto_hybrid_75_25",
        "semantic_proto_hybrid_75_25_trace_control",
        "semantic_proto_hybrid_75_25_memory_trace",
    }:
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        trace_control = key == "semantic_proto_hybrid_75_25_trace_control"
        full_trace = key == "semantic_proto_hybrid_75_25_memory_trace"
        return (SemanticRepresentativeERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, memory_trace_signature=(memory_trace_signature or trace_control or full_trace), memory_trace_audit=(memory_trace_audit or full_trace)), None)

    if key == "semantic_proto_hybrid_75_25_replay_vulnerability_audit":
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (ClassReplayVulnerabilityAuditERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, replay_vulnerability_ema_decay=.99, memory_trace_signature=True), None)

    if key == "semantic_proto_hybrid_75_25_self_ref_deterioration_audit":
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (SelfReferencedReplayDeteriorationAuditERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, self_ref_ema_decay=.99, memory_trace_signature=True), None)

    if key == "semantic_proto_hybrid_75_25_persistent_self_ref_deterioration_audit":
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (PersistentSelfReferencedReplayDeteriorationAuditERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, self_ref_ema_decay=.99, memory_trace_signature=True), None)

    if key in {"persistent_srrd_debt_swap_noop", "persistent_srrd_debt_swap_4"}:
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (PersistentSRRDDebtSwapERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, self_ref_ema_decay=.99, max_swaps_per_batch=4, force_neutral_signal=(key=="persistent_srrd_debt_swap_noop"), memory_trace_signature=(memory_trace_signature or key=="persistent_srrd_debt_swap_noop")), None)
    if key in {"persistent_srrd_selective_swap_noop", "persistent_srrd_selective_swap_1"}:
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (SelectivePersistentSRRDSwapERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=effective_replay_mb_size, seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, self_ref_ema_decay=.99, confidence_z=1.96, force_neutral_signal=(key=="persistent_srrd_selective_swap_noop"), memory_trace_signature=(memory_trace_signature or key=="persistent_srrd_selective_swap_noop")), None)
    if key in {"persistent_srrd_dual_head_calibration_noop", "persistent_srrd_dual_head_calibration_1"}:
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (ReplayFeatureDualHeadCalibrationERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, self_ref_ema_decay=.99, confidence_z=1.96, calibration_lr_scale=(0.0 if key=="persistent_srrd_dual_head_calibration_noop" else 1.0), calibration_label_smoothing=.1, memory_trace_signature=(memory_trace_signature or key=="persistent_srrd_dual_head_calibration_noop")), None)
    if key in {"persistent_srrd_fixed_alpha_025", "persistent_srrd_fixed_alpha_05", "persistent_srrd_fixed_alpha_075"}:
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        alpha = {"persistent_srrd_fixed_alpha_025": 0.25, "persistent_srrd_fixed_alpha_05": 0.5, "persistent_srrd_fixed_alpha_075": 0.75}[key]
        return (FixedAlphaDualHeadCalibrationERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, self_ref_ema_decay=.99, confidence_z=1.96, calibration_lr_scale=1.0, calibration_label_smoothing=.1, calibration_replay_detailed_audit=False, deployment_alpha=alpha, memory_trace_signature=memory_trace_signature), None)
    if key == "persistent_srrd_obc_1":
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (CanonicalOBCDualHeadERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=effective_replay_mb_size, seed=90_000+mem_size, obc_seed=190_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, self_ref_ema_decay=.99, confidence_z=1.96, calibration_lr_scale=1.0, calibration_label_smoothing=.1, calibration_replay_detailed_audit=False, memory_trace_signature=memory_trace_signature), None)
    if key in {"persistent_srrd_risk_budgeted_arbitration_noop", "persistent_srrd_risk_budgeted_arbitration_1"}:
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (RiskBudgetedDualHeadArbitrationERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, self_ref_ema_decay=.99, confidence_z=1.96, calibration_lr_scale=(0.0 if key=="persistent_srrd_risk_budgeted_arbitration_noop" else 1.0), calibration_label_smoothing=.1, memory_trace_signature=(memory_trace_signature or key=="persistent_srrd_risk_budgeted_arbitration_noop")), None)
    if key in {"persistent_srrd_prequential_arbitration_noop", "persistent_srrd_prequential_arbitration_1"}:
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (PrequentialRiskBudgetedDualHeadArbitrationERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=effective_replay_mb_size, seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, self_ref_ema_decay=.99, confidence_z=1.96, calibration_lr_scale=(0.0 if key=="persistent_srrd_prequential_arbitration_noop" else 1.0), calibration_label_smoothing=.1, calibration_replay_detailed_audit=False, memory_trace_signature=(memory_trace_signature or key=="persistent_srrd_prequential_arbitration_noop")), None)
    if key in {"persistent_srrd_prequential_current_only", "persistent_srrd_prequential_replay_only", "persistent_srrd_prequential_last_alpha", "persistent_srrd_prequential_no_smoothing"}:
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        strategy_class = {
            "persistent_srrd_prequential_current_only": PrequentialCurrentOnlyArbitrationERACE,
            "persistent_srrd_prequential_replay_only": PrequentialReplayOnlyArbitrationERACE,
            "persistent_srrd_prequential_last_alpha": PrequentialLastAlphaArbitrationERACE,
            "persistent_srrd_prequential_no_smoothing": PrequentialRiskBudgetedDualHeadArbitrationERACE,
        }[key]
        return (strategy_class(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, self_ref_ema_decay=.99, confidence_z=1.96, calibration_lr_scale=1.0, calibration_label_smoothing=(0.0 if key=="persistent_srrd_prequential_no_smoothing" else .1), calibration_replay_detailed_audit=False, memory_trace_signature=memory_trace_signature), None)
    if key in {"persistent_srrd_consequence_prequential_noop", "persistent_srrd_consequence_prequential_1"}:
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (SRRDConsequencePrequentialArbitrationERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, self_ref_ema_decay=.99, confidence_z=1.96, calibration_lr_scale=(0.0 if key=="persistent_srrd_consequence_prequential_noop" else 1.0), calibration_label_smoothing=.1, calibration_replay_detailed_audit=False, memory_trace_signature=(memory_trace_signature or key=="persistent_srrd_consequence_prequential_noop")), None)
    if key in {"persistent_srrd_pareto_head_repair_noop", "persistent_srrd_pareto_head_repair_1"}:
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (ParetoGuardedReplayRepairERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, self_ref_ema_decay=.99, confidence_z=1.96, repair_step_scale=(0.0 if key=="persistent_srrd_pareto_head_repair_noop" else 1.0), repair_interval=4, repair_current_ce_tolerance=1e-6, repair_min_replay_ce_improvement=1e-6, memory_trace_signature=(memory_trace_signature or key=="persistent_srrd_pareto_head_repair_noop")), None)
    if key in {"persistent_srrd_memory_certified_repair_noop", "persistent_srrd_memory_certified_repair_1"}:
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (MemoryCertifiedReplayRepairERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, self_ref_ema_decay=.99, confidence_z=1.96, repair_step_scale=(0.0 if key=="persistent_srrd_memory_certified_repair_noop" else 1.0), repair_interval=4, repair_current_ce_tolerance=1e-6, repair_min_replay_ce_improvement=1e-6, repair_worst_class_ce_tolerance=1e-6, memory_trace_signature=(memory_trace_signature or key=="persistent_srrd_memory_certified_repair_noop")), None)
    if key in {"persistent_srrd_loss_transfer_noop", "persistent_srrd_loss_transfer_0_5"}:
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (PersistentSRRDLossRedistributionERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, self_ref_ema_decay=.99, confidence_z=1.96, loss_transfer=(0.0 if key=="persistent_srrd_loss_transfer_noop" else .5), memory_trace_signature=(memory_trace_signature or key=="persistent_srrd_loss_transfer_noop")), None)
    if key in {"persistent_srrd_support_transfer_noop", "persistent_srrd_support_transfer_0_5"}:
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (SupportCalibratedPersistentSRRDLossRedistributionERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, self_ref_ema_decay=.99, confidence_z=1.96, loss_transfer=(0.0 if key=="persistent_srrd_support_transfer_noop" else .5), min_distinct_items_per_partition=2, memory_trace_signature=(memory_trace_signature or key=="persistent_srrd_support_transfer_noop")), None)
    if key in {"persistent_srrd_support_balanced_transfer_noop", "persistent_srrd_support_balanced_transfer_0_5"}:
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (SupportBalancedPersistentSRRDLossRedistributionERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, self_ref_ema_decay=.99, confidence_z=1.96, loss_transfer=(0.0 if key=="persistent_srrd_support_balanced_transfer_noop" else .5), min_distinct_items_per_partition=2, memory_trace_signature=(memory_trace_signature or key=="persistent_srrd_support_balanced_transfer_noop")), None)

    if key == "tirp_semantic_hybrid_75_25":
        if not tirp_policy_path:
            raise ValueError("tirp_semantic_hybrid_75_25 requires tirp_policy_path")
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (TIRPSemanticRepresentativeERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, policy_checkpoint=tirp_policy_path, allow_unapproved_policy=tirp_allow_unapproved_policy), None)

    if key == "semantic_proto_hybrid_75_25_firp_score_audit":
        if not tirp_policy_path:
            raise ValueError(f"{key} requires tirp_policy_path")
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (FIRPScoreAuditERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, policy_checkpoint=tirp_policy_path, allow_unapproved_policy=tirp_allow_unapproved_policy, memory_trace_signature=True), None)

    if key == "class_firp_hybrid_25":
        if not tirp_policy_path:
            raise ValueError(f"{key} requires tirp_policy_path")
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (ClassFIRPERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, policy_checkpoint=tirp_policy_path, allow_unapproved_policy=tirp_allow_unapproved_policy, firp_quota_fraction=.25, firp_score_refresh_samples=512), None)

    if key in {"class_firp_debt_swap_noop", "class_firp_debt_swap_4", "class_firp_debt_swap_4_trace_control", "class_firp_debt_swap_exposure_audit"}:
        if not tirp_policy_path:
            raise ValueError(f"{key} requires tirp_policy_path")
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        strategy_class = ClassFIRPExposureAuditERACE if key == "class_firp_debt_swap_exposure_audit" else ClassFIRPDebtSwapERACE
        trace_signature = key in {"class_firp_debt_swap_noop", "class_firp_debt_swap_4_trace_control", "class_firp_debt_swap_exposure_audit"}
        return (strategy_class(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, policy_checkpoint=tirp_policy_path, allow_unapproved_policy=tirp_allow_unapproved_policy, firp_quota_fraction=.25, firp_score_refresh_samples=512, max_swaps_per_batch=4, force_neutral_scores=(key=="class_firp_debt_swap_noop"), memory_trace_signature=(memory_trace_signature or trace_signature)), None)

    if key in {"tirp_proxy_relation_audit", "tirp_proxy_relation"}:
        if not tirp_policy_path:
            raise ValueError(f"{key} requires tirp_policy_path")
        family = _dataset_family(benchmark_name)
        return (TIRPProxyRelationERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=75, reservoir_capacity=25, policy_checkpoint=tirp_policy_path, relation_lambda=0.1, relation_audit_only=(key=="tirp_proxy_relation_audit")), None)

    if key == "tirp_proxy_contrastive":
        if not tirp_policy_path:
            raise ValueError("tirp_proxy_contrastive requires tirp_policy_path")
        family = _dataset_family(benchmark_name)
        return (TIRPProxyContrastiveERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=75, reservoir_capacity=25, policy_checkpoint=tirp_policy_path, proxy_lambda=.05, proxy_temperature=1.0), None)

    if key in {"tirp_semantic_relation_guard", "tirp_semantic_relation_guard_v2", "tirp_semantic_relation_guard_v2_1"}:
        if not tirp_policy_path:
            raise ValueError(f"{key} requires tirp_policy_path")
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        guarded_v2 = key == "tirp_semantic_relation_guard_v2"
        guarded_v2_1 = key == "tirp_semantic_relation_guard_v2_1"
        return (TIRPSemanticRelationGuardERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, policy_checkpoint=tirp_policy_path, allow_unapproved_policy=tirp_allow_unapproved_policy, guard_max_relative_deviation=.25, guard_semantic_confidence=.90 if guarded_v2_1 else (.95 if guarded_v2 else None), guard_min_vulnerable_fraction=.125 if guarded_v2_1 else (.25 if guarded_v2 else 0.0), guard_probability_margin_threshold=.1 if guarded_v2_1 else None), None)

    if key in {"tirp_relation_v3_audit", "tirp_relation_v3"}:
        if not tirp_policy_path:
            raise ValueError(f"{key} requires tirp_policy_path")
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (TIRPPrototypeBoundaryRelationERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, policy_checkpoint=tirp_policy_path, allow_unapproved_policy=tirp_allow_unapproved_policy, relation_lambda=.025, relation_audit_only=(key=="tirp_relation_v3_audit"), ema_decay=.99, min_prototype_count=8, margin_slack=.05, margin_cap=.5), None)

    if key in {"tirp_relation_v4_audit", "tirp_relation_v4"}:
        if not tirp_policy_path:
            raise ValueError(f"{key} requires tirp_policy_path")
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (TIRPMaturityNormalizedRelationERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, policy_checkpoint=tirp_policy_path, allow_unapproved_policy=tirp_allow_unapproved_policy, relation_lambda=.025, relation_audit_only=(key=="tirp_relation_v4_audit"), ema_decay=.99, min_prototype_count=8, margin_cap=.5, relative_slack=.25, margin_scale_floor=.005, maturity_threshold=.5, maturity_ema_decay=.9, maturity_min_updates=8), None)

    if key in {"tirp_relation_v5_audit", "tirp_relation_v5"}:
        if not tirp_policy_path:
            raise ValueError(f"{key} requires tirp_policy_path")
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        return (TIRPSparseBudgetedRelationERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, policy_checkpoint=tirp_policy_path, allow_unapproved_policy=tirp_allow_unapproved_policy, relation_lambda=.025, relation_audit_only=(key=="tirp_relation_v5_audit"), ema_decay=.99, min_prototype_count=8, margin_cap=.5, relative_slack=.25, margin_scale_floor=.005, min_reliable_targets=4, max_active_fraction=.125), None)

    if key in {"tirp_semantic_relation_audit", "tirp_semantic_relation", "tirp_semantic_relation_v2", "tirp_semantic_relation_v2_1", "tirp_semantic_relation_coverage_v2"}:
        if not tirp_policy_path:
            raise ValueError(f"{key} requires tirp_policy_path")
        family = _dataset_family(benchmark_name)
        semantic_capacity, reservoir_capacity = semantic_capacities()
        relation_mode='standardized' if key=='tirp_semantic_relation_v2' else ('centered' if key=='tirp_semantic_relation_v2_1' else 'raw')
        return (TIRPSemanticRelationERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=semantic_capacity, reservoir_capacity=reservoir_capacity, coverage_first=(key=="tirp_semantic_relation_coverage_v2"), policy_checkpoint=tirp_policy_path, allow_unapproved_policy=tirp_allow_unapproved_policy, relation_lambda=.025, relation_audit_only=(key=="tirp_semantic_relation_audit"), relation_mechanism_audit=relation_mechanism_audit, relation_replay_audit=relation_replay_audit, relation_mode=relation_mode), None)

    if key in {"tirp_semantic_boundary_audit", "tirp_semantic_boundary"}:
        if not tirp_policy_path:
            raise ValueError(f"{key} requires tirp_policy_path")
        family = _dataset_family(benchmark_name)
        return (TIRPSemanticBoundaryERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=75, reservoir_capacity=25, policy_checkpoint=tirp_policy_path, boundary_audit_only=(key=="tirp_semantic_boundary_audit")), None)
    if key in {"tirp_decision_audit", "tirp_decision_consolidated"}:
        if not tirp_policy_path:
            raise ValueError(f"{key} requires tirp_policy_path")
        family = _dataset_family(benchmark_name)
        return (TIRPDecisionConsolidatedERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=75, reservoir_capacity=25, policy_checkpoint=tirp_policy_path, decision_lambda=0.1, decision_audit_only=(key=="tirp_decision_audit")), None)
    if key in {"tirp_mature_decision_audit", "tirp_mature_decision_consolidated"}:
        if not tirp_policy_path:
            raise ValueError(f"{key} requires tirp_policy_path")
        family = _dataset_family(benchmark_name)
        return (TIRPDecisionConsolidatedERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=75, reservoir_capacity=25, policy_checkpoint=tirp_policy_path, decision_lambda=0.1, decision_audit_only=(key=="tirp_mature_decision_audit"), decision_mature_age=5000), None)
    if key in {"tirp_certified_decision_audit", "tirp_certified_decision_consolidated"}:
        if not tirp_policy_path:
            raise ValueError(f"{key} requires tirp_policy_path")
        family = _dataset_family(benchmark_name)
        return (TIRPDecisionConsolidatedERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=75, reservoir_capacity=25, policy_checkpoint=tirp_policy_path, decision_lambda=0.1, decision_audit_only=(key=="tirp_certified_decision_audit"), decision_mature_age=5000, decision_require_correct_target=True), None)
    if key in {"tirp_certified_margin_audit", "tirp_certified_margin_consolidated"}:
        if not tirp_policy_path:
            raise ValueError(f"{key} requires tirp_policy_path")
        family = _dataset_family(benchmark_name)
        return (TIRPDecisionConsolidatedERACE(**common, **paired_update_audit_args, mem_size=mem_size, batch_size_mem=min(train_mb_size,mem_size), seed=90_000+mem_size, dataset_family=family, with_reservoir=True, semantic_capacity=75, reservoir_capacity=25, policy_checkpoint=tirp_policy_path, decision_lambda=0.1, decision_audit_only=(key=="tirp_certified_margin_audit"), decision_mature_age=5000, decision_require_correct_target=True, decision_mode='label_anchored_margin'), None)

    if key == "mir":
        return (
            MIR(
                **common,
                mem_size=mem_size,
                subsample=mem_size,
                batch_size_mem=min(train_mb_size, mem_size),
            ),
            None,
        )

    if key in {"stream_clock", "stream_clock_mature", "stream_clock_mature_balanced", "stream_clock_mature_uncertain", "stream_clock_coverage", "stream_clock_mature_coverage", "stream_clock_mature_soft_coverage"}:
        rbcl_plugin = RiskBudgetingPlugin(
            budget_mode="uniform",
            allocation=allocation,
            budget_ratio=budget_ratio,
            distill_lambda=distill_lambda,
            distill_temperature=distill_temperature,
            use_instability=use_instability,
            use_consequence=use_consequence,
            use_uncertainty=use_uncertainty,
            use_disagreement=use_disagreement,
            use_prior=use_prior,
            instability_mode=instability_mode,
            fusion_mode=fusion_mode,
            instability_lambda=instability_lambda,
            consequence_mode=consequence_mode,
            allocation_scope=allocation_scope,
            gradient_audit=gradient_audit,
            gradient_audit_every=gradient_audit_every,
            counterfactual_audit=counterfactual_audit,
            counterfactual_audit_every=counterfactual_audit_every,
            counterfactual_probe_size=counterfactual_probe_size,
            counterfactual_alpha_audit=counterfactual_alpha_audit,
            counterfactual_alpha_values=counterfactual_alpha_values,
            counterfactual_rho=counterfactual_rho,
            replay_geometry_audit=replay_geometry_audit,
            replay_geometry_audit_every=replay_geometry_audit_every,
            conflict_residual_audit=conflict_residual_audit,
            conflict_residual_audit_every=conflict_residual_audit_every,
            slow_fast_state_audit=slow_fast_state_audit,
            slow_fast_state_audit_every=slow_fast_state_audit_every,
            slow_fast_ema_decay=slow_fast_ema_decay,
            fair_compute=True,
        )
        clock_replay = StreamClockReplayPlugin(
            mem_size=mem_size,
            clock_samples=stream_clock_samples,
            seed=70_000 + stream_clock_samples,
            min_replay_age_samples=(
                stream_clock_min_replay_age_samples
                if key in {"stream_clock_mature", "stream_clock_mature_balanced", "stream_clock_mature_uncertain", "stream_clock_mature_coverage", "stream_clock_mature_soft_coverage"}
                else 0
            ),
            coverage_floor=(
                stream_clock_coverage_floor
                if key in {"stream_clock_coverage", "stream_clock_mature_coverage"}
                else 0
            ),
            coverage_strength=(
                stream_clock_coverage_strength
                if key == "stream_clock_mature_soft_coverage"
                else 0.0
            ),
            replay_loss_scale=(
                stream_clock_replay_loss_scale
                if key == "stream_clock_mature_balanced"
                else 1.0
            ),
            current_uncertainty_lambda=(
                stream_clock_current_uncertainty_lambda
                if key == "stream_clock_mature_uncertain"
                else 0.0
            ),
        )
        strategy = Naive(**common, plugins=[clock_replay, rbcl_plugin])
        strategy.stream_clock_plugin = clock_replay
        return strategy, rbcl_plugin

    if key == "stream_clock_bridge":
        rbcl_plugin = RiskBudgetingPlugin(
            budget_mode="uniform",
            allocation=allocation,
            budget_ratio=budget_ratio,
            distill_lambda=distill_lambda,
            distill_temperature=distill_temperature,
            use_instability=use_instability,
            use_consequence=use_consequence,
            use_uncertainty=use_uncertainty,
            use_disagreement=use_disagreement,
            use_prior=use_prior,
            instability_mode=instability_mode,
            fusion_mode=fusion_mode,
            instability_lambda=instability_lambda,
            consequence_mode=consequence_mode,
            allocation_scope=allocation_scope,
            fair_compute=True,
        )
        bridge_replay = ClockBridgeReplayPlugin(mem_size=mem_size)
        strategy = Naive(**common, plugins=[bridge_replay, rbcl_plugin])
        strategy.clock_bridge_plugin = bridge_replay
        return strategy, rbcl_plugin

    if key in {"uniform_budget", "risk_budget", "plasticity_budget", "stream_clock_scheduler", "deferred_stream_clock", "boundary_debt", "boundary_repair", "boundary_feature_repair", "memory_noise"}:
        rbcl_plugin = RiskBudgetingPlugin(
            budget_mode="risk" if key == "risk_budget" else "uniform",
            allocation=allocation,
            budget_ratio=budget_ratio,
            distill_lambda=distill_lambda,
            distill_temperature=distill_temperature,
            use_instability=use_instability,
            use_consequence=use_consequence,
            use_uncertainty=use_uncertainty,
            use_disagreement=use_disagreement,
            use_prior=use_prior,
            instability_mode=instability_mode,
            fusion_mode=fusion_mode,
            instability_lambda=instability_lambda,
            consequence_mode=consequence_mode,
            allocation_scope=allocation_scope,
            gradient_audit=gradient_audit,
            gradient_audit_every=gradient_audit_every,
            counterfactual_audit=counterfactual_audit,
            counterfactual_audit_every=counterfactual_audit_every,
            counterfactual_probe_size=counterfactual_probe_size,
            counterfactual_alpha_audit=counterfactual_alpha_audit,
            counterfactual_alpha_values=counterfactual_alpha_values,
            counterfactual_rho=counterfactual_rho,
            plasticity_budget_control=(
                plasticity_budget_control or key == "plasticity_budget"
            ),
            plasticity_budget_every=plasticity_budget_every,
            replay_geometry_audit=replay_geometry_audit,
            replay_geometry_audit_every=replay_geometry_audit_every,
            conflict_residual_audit=conflict_residual_audit,
            conflict_residual_audit_every=conflict_residual_audit_every,
            slow_fast_state_audit=slow_fast_state_audit,
            slow_fast_state_audit_every=slow_fast_state_audit_every,
            slow_fast_ema_decay=slow_fast_ema_decay,
            boundary_debt_audit=boundary_debt_audit,
            boundary_debt_probe_size=boundary_debt_probe_size,
            boundary_debt_replay=boundary_debt_replay or key == "boundary_debt",
            boundary_debt_temperature=boundary_debt_temperature,
            boundary_debt_mix=boundary_debt_mix,
            boundary_repair=boundary_repair or key == "boundary_repair",
            boundary_feature_repair=(
                boundary_feature_repair or key == "boundary_feature_repair"
            ),
            boundary_repair_pairs=boundary_repair_pairs,
            boundary_repair_samples_per_class=boundary_repair_samples_per_class,
            boundary_repair_margin=boundary_repair_margin,
            boundary_repair_lambda=boundary_repair_lambda,
            boundary_repair_current_ce_tolerance=boundary_repair_current_ce_tolerance,
            fair_compute=True,
        )
        if replay_storage not in {"experience", "reservoir", "class"}:
            raise ValueError("replay_storage must be one of: experience, reservoir, class")
        if retention_mode not in {"experience_balanced", "c_uniform", "c_aware"}:
            raise ValueError(
                "retention_mode must be one of: experience_balanced, "
                "c_uniform, c_aware"
            )
        if retention_strength < 0:
            raise ValueError("retention_strength must be non-negative")
        if replay_storage == "reservoir":
            if retention_mode != "experience_balanced":
                raise ValueError("consequence-aware retention requires replay_storage=experience")
            storage_policy = ReservoirSamplingBuffer(max_size=mem_size)
        elif replay_storage == "class":
            if retention_mode != "experience_balanced":
                raise ValueError("consequence-aware retention requires replay_storage=experience")
            storage_policy = ClassBalancedBuffer(max_size=mem_size, adaptive_size=True)
        elif retention_mode == "experience_balanced":
            storage_policy = ExperienceBalancedBuffer(
                max_size=mem_size, adaptive_size=True
            )
        else:
            storage_policy = ConsequenceAwareExperienceBalancedBuffer(
                max_size=mem_size,
                consequence_provider=lambda: rbcl_plugin.class_prior,
                mode="uniform" if retention_mode == "c_uniform" else "c_aware",
                strength=retention_strength,
            )
        replay_plugin = ReplayPlugin(
            mem_size=mem_size, storage_policy=storage_policy
        )
        plugins = [replay_plugin]
        if key == "memory_noise":
            plugins.append(
                ReplayLabelNoisePlugin(
                    noise_rate=memory_label_noise_rate,
                    seed=memory_label_noise_seed,
                )
            )
        plugins.append(rbcl_plugin)
        strategy = Naive(**common, plugins=plugins)
        strategy.rbcl_retention_policy = storage_policy
        return strategy, rbcl_plugin

    if key == "ewc":
        return EWC(**common, ewc_lambda=ewc_lambda, mode="separate"), None

    if key == "si":
        return SynapticIntelligence(**common, si_lambda=si_lambda), None

    if key == "lwf":
        return LwF(
            **common,
            alpha=lwf_alpha,
            temperature=distill_temperature,
        ), None

    if key == "joint":
        return JointTraining(**common), None

    raise ValueError(
        "Unknown strategy. Supported: naive, replay, derpp, er_ace, causal_er_ace, causal_er_ace_trace_control, causal_er_ace_memory_trace, causal_er_ace_prequential_arbitration_noop, causal_er_ace_prequential_arbitration_1, semantic_proto_hybrid_75_25_trace_control, semantic_proto_hybrid_75_25_memory_trace, scr, causal_er_ace_cb, causal_er_ace_hybrid, causal_counterfactual_mixture, causal_hybrid_accuracy_audit, causal_hybrid_accuracy_correction, causal_hybrid_gain_budget_audit, causal_hybrid_gain_budget_correction, global_temporal_steady, mir, stream_clock, stream_clock_mature, stream_clock_mature_balanced, stream_clock_mature_uncertain, stream_clock_coverage, stream_clock_mature_coverage, stream_clock_mature_soft_coverage, stream_clock_scheduler, deferred_stream_clock, uniform_budget, risk_budget, plasticity_budget, boundary_debt, boundary_repair, boundary_feature_repair, memory_noise, "
        "global_semantic_anchor, global_semantic_replay_anchor, semantic_anchor_compatibility_audit, persistent_srrd_dual_head_calibration_noop, persistent_srrd_dual_head_calibration_1, persistent_srrd_fixed_alpha_025, persistent_srrd_fixed_alpha_05, persistent_srrd_fixed_alpha_075, persistent_srrd_obc_1, persistent_srrd_risk_budgeted_arbitration_noop, persistent_srrd_risk_budgeted_arbitration_1, persistent_srrd_prequential_arbitration_noop, persistent_srrd_prequential_arbitration_1, persistent_srrd_prequential_current_only, persistent_srrd_prequential_replay_only, persistent_srrd_prequential_last_alpha, persistent_srrd_prequential_no_smoothing, persistent_srrd_consequence_prequential_noop, persistent_srrd_consequence_prequential_1, tirp_semantic_relation_coverage_v2, ewc, si, lwf, joint."
    )


def run_stream(
    strategy,
    benchmark,
    *,
    strategy_name: str,
    output_dir: str | Path,
    rbcl_plugin: Optional[RiskBudgetingPlugin] = None,
    num_workers: int = 0,
    validation_stream=None,
    historical_validation_stream=None,
    eval_stride: int = 1,
) -> Dict[str, object]:
    """Train/evaluate on an Avalanche stream and export paper-facing logs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    if eval_stride <= 0:
        raise ValueError("eval_stride must be positive")
    validation_experiences = list(validation_stream) if validation_stream is not None else None
    historical_validation_experiences = (
        list(historical_validation_stream)
        if historical_validation_stream is not None
        else validation_experiences
    )
    # Strong baselines such as DER++, ER-ACE, and MIR own their training
    # objectives and must not receive the RBCL loss-rewriting plugin.  This
    # detached tracker only evaluates per-class accuracy/forgetting so their
    # hard/blurry PSS remains directly comparable to RBCL runs.
    detached_classwise_tracker = None
    if rbcl_plugin is None and historical_validation_experiences is not None:
        detached_classwise_tracker = RiskBudgetingPlugin(
            budget_mode="uniform",
            consequence_mode="raw_forgetting",
            use_instability=False,
            use_consequence=False,
            use_uncertainty=False,
            use_disagreement=False,
            use_prior=False,
            distill_lambda=0.0,
            fair_compute=False,
        )
    deferred_audit = None
    deferred_config = getattr(strategy, "deferred_stream_clock_config", None)
    if deferred_config is not None:
        deferred_audit = {
            **deferred_config,
            "emitted_event_sizes": [],
            "reservoir_peak_size": 0,
        }

    if strategy_name.lower() == "joint":
        train_metrics = strategy.train(benchmark.train_stream, num_workers=num_workers)
        eval_metrics = strategy.eval(benchmark.test_stream, num_workers=num_workers)
        results.append({"train": train_metrics, "eval": eval_metrics})
    else:
        train_experiences = list(benchmark.train_stream)
        for exp_id, train_exp in enumerate(train_experiences):
            print(f"[{strategy_name}] train experience {exp_id}")
            is_eval_checkpoint = (
                (exp_id + 1) % eval_stride == 0
                or exp_id == len(train_experiences) - 1
            )
            if (
                is_eval_checkpoint
                and rbcl_plugin is not None
                and validation_experiences is not None
            ):
                rbcl_plugin.set_current_validation_experience(
                    validation_experiences[exp_id], experience_id=exp_id
                )
            train_metrics = strategy.train(train_exp, num_workers=num_workers)
            if deferred_audit is not None:
                deferred_audit["emitted_event_sizes"].append(len(train_exp.dataset))
                buffer = getattr(strategy.rbcl_retention_policy, "buffer", None)
                if buffer is not None:
                    deferred_audit["reservoir_peak_size"] = max(
                        deferred_audit["reservoir_peak_size"], len(buffer)
                    )
            if (
                is_eval_checkpoint
                and historical_validation_experiences is not None
            ):
                consequence_tracker = (
                    rbcl_plugin
                    if rbcl_plugin is not None
                    else detached_classwise_tracker
                )
                consequence_tracker.update_historical_consequence(
                    strategy,
                    (
                        historical_validation_experiences
                        if historical_validation_stream is not None
                        else historical_validation_experiences[: exp_id + 1]
                    ),
                    batch_size=strategy.eval_mb_size,
                    num_workers=num_workers,
                    experience_id=exp_id,
                )
            eval_metrics = (
                strategy.eval(benchmark.test_stream, num_workers=num_workers)
                if is_eval_checkpoint
                else {}
            )
            results.append(
                {"experience": exp_id, "train": train_metrics, "eval": eval_metrics}
            )

    payload: Dict[str, object] = {
        "strategy": strategy_name,
        "last_metrics": strategy.evaluator.get_last_metrics(),
        "experience_results": results,
    }
    if rbcl_plugin is not None:
        payload["budget_stats"] = rbcl_plugin.summary()
    if detached_classwise_tracker is not None:
        payload["classwise_audit"] = {
            "consequence_history": detached_classwise_tracker.consequence_history,
            "training_objective_modified": False,
        }
    for plugin in getattr(strategy, "plugins", []):
        if isinstance(plugin, ReplayLabelNoisePlugin):
            payload["memory_noise_audit"] = plugin.summary()
    retention_policy = getattr(strategy, "rbcl_retention_policy", None)
    if retention_policy is not None and hasattr(retention_policy, "summary"):
        payload["retention_stats"] = retention_policy.summary()
    stream_clock_plugin = getattr(strategy, "stream_clock_plugin", None)
    if stream_clock_plugin is not None:
        payload["stream_clock_stats"] = stream_clock_plugin.summary()
    clock_bridge_plugin = getattr(strategy, "clock_bridge_plugin", None)
    if clock_bridge_plugin is not None:
        payload["clock_bridge_stats"] = clock_bridge_plugin.summary()
    if hasattr(strategy, "rbcl_summary"):
        payload["strategy_audit"] = strategy.rbcl_summary()
    if getattr(strategy, "paired_update_audit", False) and hasattr(
        strategy, "export_paired_update_audit"
    ):
        payload["paired_update_audit_file"] = strategy.export_paired_update_audit(
            output_dir / "paired_update_audit.pt"
        )
    if deferred_audit is not None:
        payload["deferred_stream_clock_audit"] = deferred_audit

    save_json(output_dir / "summary.json", payload)
    return payload

