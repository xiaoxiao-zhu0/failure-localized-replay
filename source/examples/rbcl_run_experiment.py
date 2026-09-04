"""Full comparison entry point for RBCL experiments.

Paper mapping:
This script runs the baseline table: Fine-tune, ER, Uniform Budget, Risk
Budget, and optional regularization/distillation upper-bound baselines.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rbcl import (
    apply_deferred_sample_clock,
    apply_sample_clock,
    build_benchmark,
    build_evaluator,
    build_model,
    build_strategy,
    run_stream,
)
from avalanche.benchmarks.scenarios.validation_scenario import (
    benchmark_with_validation_stream,
)
from rbcl.utils import configure_determinism, ensure_dir, resolve_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run RBCL continual learning comparisons.")
    parser.add_argument("--benchmark", default="split_mnist")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--strategies", default="naive,replay,uniform_budget,risk_budget")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=-1)
    parser.add_argument("--n_experiences", type=int, default=5)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--train_mb_size", type=int, default=64)
    parser.add_argument(
        "--replay_mb_size",
        type=int,
        default=None,
        help=(
            "Replay batch size for RBCL/ER-ACE-family strategies. "
            "Defaults to train_mb_size and is capped by mem_size."
        ),
    )
    parser.add_argument("--eval_mb_size", type=int, default=128)
    parser.add_argument("--mem_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--allocation", default="soft", choices=["soft", "topk"])
    parser.add_argument("--budget_ratio", type=float, default=1.0)
    parser.add_argument("--distill_lambda", type=float, default=0.5)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument("--no_instability", action="store_true")
    parser.add_argument("--no_consequence", action="store_true")
    parser.add_argument("--no_uncertainty", action="store_true")
    parser.add_argument("--no_disagreement", action="store_true")
    parser.add_argument("--no_prior", action="store_true")
    parser.add_argument(
        "--instability_mode",
        default="virtual_replay_loss",
        choices=["entropy_kl", "virtual_replay_loss"],
    )
    parser.add_argument(
        "--fusion_mode", default="c_gated", choices=["product", "c_gated"]
    )
    parser.add_argument("--instability_lambda", type=float, default=0.25)
    parser.add_argument(
        "--allocation_scope", default="global", choices=["global", "replay_group"]
    )
    parser.add_argument(
        "--gradient_audit",
        action="store_true",
        help="Record non-mutating current/replay classifier-head gradient conflicts.",
    )
    parser.add_argument("--gradient_audit_every", type=int, default=100)
    parser.add_argument(
        "--replay_geometry_audit",
        action="store_true",
        help="Record non-mutating sample-level current/replay gradient geometry.",
    )
    parser.add_argument("--replay_geometry_audit_every", type=int, default=500)
    parser.add_argument("--conflict_residual_audit", action="store_true")
    parser.add_argument("--conflict_residual_audit_every", type=int, default=500)
    parser.add_argument(
        "--value_coverage_audit",
        action="store_true",
        help=(
            "Record non-mutating value-times-coverage replay routing "
            "counterfactuals for causal ER-ACE."
        ),
    )
    parser.add_argument("--value_coverage_audit_every", type=int, default=500)
    parser.add_argument("--value_coverage_rho", type=float, default=0.95)
    parser.add_argument(
        "--paired_update_audit",
        action="store_true",
        help="Record non-mutating classifier-head gradients for the D35 paired update audit.",
    )
    parser.add_argument("--paired_update_audit_every", type=int, default=512)
    parser.add_argument(
        "--memory_trace_signature",
        action="store_true",
        help="Hash replay selections, final memory, and final model without changing training.",
    )
    parser.add_argument(
        "--memory_trace_audit",
        action="store_true",
        help="Record non-mutating memory occupancy, replay exposure, and age traces.",
    )
    parser.add_argument("--slow_fast_state_audit", action="store_true")
    parser.add_argument("--slow_fast_state_audit_every", type=int, default=500)
    parser.add_argument("--slow_fast_ema_decay", type=float, default=0.99)
    parser.add_argument(
        "--counterfactual_audit",
        action="store_true",
        help="Record non-mutating Uniform-vs-priority one-step validation counterfactuals.",
    )
    parser.add_argument("--counterfactual_audit_every", type=int, default=200)
    parser.add_argument("--counterfactual_probe_size", type=int, default=64)
    parser.add_argument(
        "--counterfactual_alpha_audit",
        action="store_true",
        help=(
            "Record non-mutating current-only versus replay-strength "
            "counterfactuals; actual training remains unchanged."
        ),
    )
    parser.add_argument(
        "--counterfactual_alpha_values",
        default="0,0.25,0.5,0.75,1",
        help="Comma-separated replay strengths in [0, 1]; must include 0.",
    )
    parser.add_argument(
        "--counterfactual_rho",
        type=float,
        default=0.95,
        help="Minimum fraction of current-only progress required by the audit.",
    )
    parser.add_argument(
        "--plasticity_budget_every",
        type=int,
        default=500,
        help=(
            "Fixed update interval for applied replay-plasticity decisions; "
            "used by strategy=plasticity_budget."
        ),
    )
    parser.add_argument(
        "--boundary_debt_audit",
        action="store_true",
        help=(
            "Record non-mutating pre-update new-to-old confusion and paired "
            "post-update cross-task errors on training-held-out data."
        ),
    )
    parser.add_argument("--boundary_debt_probe_size", type=int, default=256)
    parser.add_argument("--boundary_debt_temperature", type=float, default=0.25)
    parser.add_argument("--boundary_debt_mix", type=float, default=0.5)
    parser.add_argument("--boundary_repair_pairs", type=int, default=2)
    parser.add_argument("--boundary_repair_samples_per_class", type=int, default=32)
    parser.add_argument("--boundary_repair_margin", type=float, default=0.5)
    parser.add_argument("--boundary_repair_lambda", type=float, default=0.5)
    parser.add_argument("--boundary_repair_current_ce_tolerance", type=float, default=0.01)
    parser.add_argument("--memory_label_noise_rate", type=float, default=0.0)
    parser.add_argument(
        "--replay_storage",
        default="experience",
        choices=["experience", "reservoir", "class"],
    )
    parser.add_argument(
        "--consequence_mode",
        default="loss_ema",
        choices=[
            "loss_ema",
            "historical_forgetting",
            "raw_forgetting",
            "validation_error",
        ],
    )
    parser.add_argument(
        "--retention_mode",
        default="experience_balanced",
        choices=["experience_balanced", "c_uniform", "c_aware"],
    )
    parser.add_argument("--retention_strength", type=float, default=1.0)
    parser.add_argument(
        "--stream_clock_samples",
        type=int,
        default=1000,
        help="Current samples between task-ID-free Stream-Clock memory updates.",
    )
    parser.add_argument(
        "--stream_clock_defer_samples",
        type=int,
        default=50,
        help="Tail samples delayed by one fixed clock for deferred Stream-Clock.",
    )
    parser.add_argument(
        "--stream_clock_eval_stride",
        type=int,
        default=10,
        help="Evaluate every N fixed-clock events; final event is always evaluated.",
    )
    parser.add_argument(
        "--stream_clock_min_replay_age_samples",
        type=int,
        default=0,
        help="Fixed arrival-sample age required before clocked memory is replayable.",
    )
    parser.add_argument(
        "--stream_clock_coverage_floor",
        type=int,
        default=0,
        help="Per-label memory copies protected by coverage-aware Stream-Clock variants.",
    )
    parser.add_argument(
        "--stream_clock_coverage_strength",
        type=float,
        default=0.0,
        help="Bounded soft coverage pressure in [0, 1] for Stream-Clock replay.",
    )
    parser.add_argument(
        "--stream_clock_replay_loss_scale",
        type=float,
        default=1.0,
        help="Loss mass multiplier for mature replay samples in clocked replay.",
    )
    parser.add_argument(
        "--stream_clock_current_uncertainty_lambda",
        type=float,
        default=0.0,
        help="Prediction-entropy plasticity multiplier for current clocked samples.",
    )
    parser.add_argument("--validation_fraction", type=float, default=0.0)
    parser.add_argument(
        "--tirp_policy_path", default="",
        help="Frozen D32 policy checkpoint required by tirp_semantic_hybrid_75_25.",
    )
    parser.add_argument(
        "--tirp_allow_unapproved_policy",
        action="store_true",
        help="Allow an explicitly exploratory-only TIRP checkpoint for screening.",
    )
    parser.add_argument(
        "--relation_mechanism_audit",
        action="store_true",
        help="Record non-mutating Relation loss, gradient, pair, and age diagnostics.",
    )
    parser.add_argument(
        "--relation_replay_audit",
        action="store_true",
        help="Record replay-specific non-mutating Relation gradient and margin diagnostics.",
    )
    parser.add_argument(
        "--historical_reference",
        default="event_validation",
        choices=["event_validation", "test_stream"],
        help=(
            "Reference stream for reported historical accuracy/forgetting. "
            "Use test_stream only for a pre-specified internal screening run that must "
            "compare different training-clock granularities."
        ),
    )
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--output_dir", default="results/rbcl/full_comparison")
    parser.add_argument("--full_metrics", action="store_true")
    parser.add_argument("--enable_fwt", action="store_true")
    parser.add_argument("--eval_every", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic PyTorch/CUDA kernels for paired causal controls.",
    )
    parser.add_argument(
        "--save_final_model",
        action="store_true",
        help="Save the final model state for a post-training, non-mutating diagnostic.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.replay_mb_size is not None and args.replay_mb_size <= 0:
        raise ValueError("replay_mb_size must be positive when provided")
    configure_determinism(args.deterministic)
    strategy_names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if "deferred_stream_clock" in strategy_names:
        if not 0 < args.stream_clock_defer_samples < args.stream_clock_samples:
            raise ValueError(
                "stream_clock_defer_samples must be in (0, stream_clock_samples)"
            )
        if args.stream_clock_defer_samples >= args.mem_size:
            raise ValueError("stream_clock_defer_samples must be smaller than mem_size")
    if args.historical_reference == "test_stream" and args.validation_fraction != 0.0:
        raise ValueError(
            "historical_reference=test_stream requires validation_fraction=0 "
            "so all compared methods train on the same sample stream"
        )
    args.counterfactual_alpha_values = tuple(
        float(value.strip())
        for value in args.counterfactual_alpha_values.split(",")
        if value.strip()
    )
    if "plasticity_budget" in strategy_names and args.validation_fraction <= 0.0:
        raise ValueError(
            "strategy=plasticity_budget requires validation_fraction > 0 for "
            "a train-held-out current-progress probe"
        )
    root_out = ensure_dir(Path(args.output_dir) / args.benchmark / f"seed_{args.seed}")
    device = resolve_device(args.cuda)
    all_results = {}
    if args.enable_fwt and args.eval_every < 0:
        args.eval_every = 0

    for strategy_name in strategy_names:
        print(f"=== Running {strategy_name} on {args.benchmark} ===")
        set_seed(args.seed)

        # Rebuild benchmark/model per method to keep baselines independent.
        use_stream_clock_scheduler = strategy_name in {
            "stream_clock_scheduler",
            "stream_clock_bridge",
            "deferred_stream_clock",
        }
        benchmark = build_benchmark(
            args.benchmark,
            n_experiences=args.n_experiences,
            seed=args.seed,
            dataset_root=args.dataset_root,
            validation_fraction=(
                0.0
                if use_stream_clock_scheduler
                or args.historical_reference == "test_stream"
                else args.validation_fraction
            ),
        )
        historical_validation_stream = (
            list(benchmark.test_stream)
            if args.historical_reference == "test_stream"
            else None
        )
        if use_stream_clock_scheduler:
            # Split off a fixed classwise held-out stream before adapting the
            # training stream. Otherwise micro-clock methods would measure
            # forgetting over 200 windows while ER uses 10 class segments.
            if (
                args.validation_fraction > 0
                and args.historical_reference != "test_stream"
            ):
                benchmark = benchmark_with_validation_stream(
                    benchmark,
                    validation_size=args.validation_fraction,
                    shuffle=True,
                    seed=args.seed,
                )
                historical_validation_stream = list(benchmark.valid_stream)
            if strategy_name == "deferred_stream_clock":
                benchmark = apply_deferred_sample_clock(
                    benchmark,
                    clock_samples=args.stream_clock_samples,
                    defer_samples=args.stream_clock_defer_samples,
                )
            else:
                benchmark = apply_sample_clock(
                    benchmark, clock_samples=args.stream_clock_samples
                )
        num_classes = getattr(benchmark, "n_classes", None)
        if num_classes is None:
            # benchmark_with_validation_stream returns a generic CLScenario.
            # Recover the class count from the training datasets without using
            # the test stream.
            num_classes = (
                max(
                    int(max(experience.dataset.targets))
                    for experience in benchmark.train_stream
                )
                + 1
            )
        model = build_model(
            args.model,
            num_classes=num_classes,
            benchmark_name=args.benchmark,
        )
        out_dir = ensure_dir(root_out / strategy_name)
        evaluator = build_evaluator(
            out_dir / "avalanche_logs",
            quiet=args.quiet,
            full_metrics=args.full_metrics,
            include_fwt=args.enable_fwt,
            include_train_epoch_accuracy=(strategy_name != "scr"),
        )

        strategy, rbcl_plugin = build_strategy(
            strategy_name,
            model=model,
            evaluator=evaluator,
            device=device,
            lr=args.lr,
            momentum=args.momentum,
            train_mb_size=args.train_mb_size,
            replay_mb_size=args.replay_mb_size,
            train_epochs=args.train_epochs,
            eval_mb_size=args.eval_mb_size,
            mem_size=(
                args.mem_size - args.stream_clock_defer_samples
                if strategy_name == "deferred_stream_clock"
                else args.mem_size
            ),
            allocation=args.allocation,
            budget_ratio=args.budget_ratio,
            distill_lambda=args.distill_lambda,
            distill_temperature=args.distill_temperature,
            use_instability=not args.no_instability,
            use_consequence=not args.no_consequence,
            use_uncertainty=not args.no_uncertainty,
            use_disagreement=not args.no_disagreement,
            use_prior=not args.no_prior,
            instability_mode=args.instability_mode,
            fusion_mode=args.fusion_mode,
            instability_lambda=args.instability_lambda,
            consequence_mode=args.consequence_mode,
            allocation_scope=args.allocation_scope,
            gradient_audit=args.gradient_audit,
            gradient_audit_every=args.gradient_audit_every,
            replay_geometry_audit=args.replay_geometry_audit,
            replay_geometry_audit_every=args.replay_geometry_audit_every,
            conflict_residual_audit=args.conflict_residual_audit,
            conflict_residual_audit_every=args.conflict_residual_audit_every,
            slow_fast_state_audit=args.slow_fast_state_audit,
            slow_fast_state_audit_every=args.slow_fast_state_audit_every,
            slow_fast_ema_decay=args.slow_fast_ema_decay,
            counterfactual_audit=args.counterfactual_audit,
            counterfactual_audit_every=args.counterfactual_audit_every,
            counterfactual_probe_size=args.counterfactual_probe_size,
            counterfactual_alpha_audit=args.counterfactual_alpha_audit,
            counterfactual_alpha_values=args.counterfactual_alpha_values,
            counterfactual_rho=args.counterfactual_rho,
            plasticity_budget_control=(strategy_name == "plasticity_budget"),
            plasticity_budget_every=args.plasticity_budget_every,
            boundary_debt_audit=args.boundary_debt_audit,
            boundary_debt_probe_size=args.boundary_debt_probe_size,
            boundary_debt_temperature=args.boundary_debt_temperature,
            boundary_debt_mix=args.boundary_debt_mix,
            boundary_repair_pairs=args.boundary_repair_pairs,
            boundary_repair_samples_per_class=args.boundary_repair_samples_per_class,
            boundary_repair_margin=args.boundary_repair_margin,
            boundary_repair_lambda=args.boundary_repair_lambda,
            boundary_repair_current_ce_tolerance=args.boundary_repair_current_ce_tolerance,
            memory_label_noise_rate=args.memory_label_noise_rate,
            memory_label_noise_seed=args.seed + 70_000,
            replay_storage=args.replay_storage,
            retention_mode=args.retention_mode,
            retention_strength=args.retention_strength,
            stream_clock_samples=args.stream_clock_samples,
            stream_clock_min_replay_age_samples=args.stream_clock_min_replay_age_samples,
            stream_clock_coverage_floor=args.stream_clock_coverage_floor,
            stream_clock_coverage_strength=args.stream_clock_coverage_strength,
            stream_clock_replay_loss_scale=args.stream_clock_replay_loss_scale,
            stream_clock_current_uncertainty_lambda=args.stream_clock_current_uncertainty_lambda,
            tirp_policy_path=args.tirp_policy_path,
            tirp_allow_unapproved_policy=args.tirp_allow_unapproved_policy,
            relation_mechanism_audit=args.relation_mechanism_audit,
            relation_replay_audit=args.relation_replay_audit,
            value_coverage_audit=args.value_coverage_audit,
            value_coverage_audit_every=args.value_coverage_audit_every,
            value_coverage_rho=args.value_coverage_rho,
            paired_update_audit=args.paired_update_audit,
            paired_update_audit_every=args.paired_update_audit_every,
            memory_trace_signature=args.memory_trace_signature,
            memory_trace_audit=args.memory_trace_audit,
            eval_every=args.eval_every,
            benchmark_name=args.benchmark,
            num_classes=num_classes,
        )
        if strategy_name == "deferred_stream_clock":
            strategy.deferred_stream_clock_config = {
                "total_online_storage_cap": args.mem_size,
                "deferred_queue_capacity": args.stream_clock_defer_samples,
                "reservoir_capacity": (
                    args.mem_size - args.stream_clock_defer_samples
                ),
                "clock_samples": args.stream_clock_samples,
                "uses_task_id": False,
                "uses_labels_for_boundary": False,
                "first_train_count_per_raw_sample": 1,
            }

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started_at = time.perf_counter()
        strategy_result = run_stream(
            strategy,
            benchmark,
            strategy_name=strategy_name,
            output_dir=out_dir,
            rbcl_plugin=rbcl_plugin,
            validation_stream=(
                benchmark.valid_stream
                if args.validation_fraction > 0 and not use_stream_clock_scheduler
                else None
            ),
            historical_validation_stream=historical_validation_stream,
            eval_stride=(
                args.stream_clock_eval_stride if use_stream_clock_scheduler else 1
            ),
        )
        strategy_result["run_metadata"] = {
            "benchmark": args.benchmark,
            "model": args.model,
            "strategy": strategy_name,
            "seed": args.seed,
            "n_experiences": args.n_experiences,
            "train_epochs": args.train_epochs,
            "train_mb_size": args.train_mb_size,
            "replay_mb_size": min(
                args.replay_mb_size
                if args.replay_mb_size is not None
                else args.train_mb_size,
                args.mem_size,
            ),
            "eval_mb_size": args.eval_mb_size,
            "mem_size": args.mem_size,
            "lr": args.lr,
            "momentum": args.momentum,
            "validation_fraction": args.validation_fraction,
            "historical_reference": args.historical_reference,
            "deterministic": args.deterministic,
            "elapsed_seconds": time.perf_counter() - started_at,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        }
        save_json(out_dir / "summary.json", strategy_result)
        all_results[strategy_name] = strategy_result
        if args.save_final_model:
            torch.save(
                {
                    "model_name": args.model,
                    "benchmark": args.benchmark,
                    "num_classes": num_classes,
                    "state_dict": strategy.model.state_dict(),
                },
                out_dir / "final_model_state.pt",
            )

    save_json(root_out / "all_results.json", all_results)
    print(f"Saved all results to: {root_out / 'all_results.json'}")


if __name__ == "__main__":
    main()
