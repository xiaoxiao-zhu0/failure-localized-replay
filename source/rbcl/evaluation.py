"""Evaluation setup for RBCL experiments."""

from __future__ import annotations

from pathlib import Path
from typing import List

from avalanche.evaluation.metrics import (
    MAC_metrics,
    accuracy_metrics,
    bwt_metrics,
    cpu_usage_metrics,
    forgetting_metrics,
    forward_transfer_metrics,
    loss_metrics,
    ram_usage_metrics,
    timing_metrics,
)
from avalanche.logging import CSVLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def build_evaluator(
    log_dir: str | Path,
    *,
    quiet: bool = False,
    full_metrics: bool = False,
    include_fwt: bool = False,
    include_train_epoch_accuracy: bool = True,
) -> EvaluationPlugin:
    """Create an Avalanche evaluator matching the paper metric section."""
    loggers: List[object] = [CSVLogger(str(log_dir))]
    if not quiet:
        loggers.append(InteractiveLogger())

    metrics = [
        accuracy_metrics(
            epoch=include_train_epoch_accuracy, experience=True, stream=True
        ),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        timing_metrics(epoch=True, experience=True, stream=True),
    ]

    # FWT in this Avalanche version requires strategy.eval_every > -1. Keep it
    # opt-in so the minimal demo can run with the standard post-experience eval.
    if include_fwt:
        metrics.append(forward_transfer_metrics(experience=True, stream=True))

    # Full metrics are useful for paper tables, but can slow down a demo run.
    if full_metrics:
        metrics.extend(
            [
                cpu_usage_metrics(experience=True, stream=True),
                ram_usage_metrics(experience=True, stream=True),
                MAC_metrics(experience=True),
            ]
        )

    return EvaluationPlugin(*metrics, loggers=loggers, collect_all=True)
