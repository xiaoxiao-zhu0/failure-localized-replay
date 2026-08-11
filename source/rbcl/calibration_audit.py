"""Audit-only prediction statistics for dual-head calibration.

The functions in this module never select samples, update parameters, or feed
values back into a continual-learning strategy.  They aggregate already
computed logits and labels into JSON-serializable evidence.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def empty_dual_head_audit(bin_count: int = 15) -> dict:
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    return {
        "bin_count": int(bin_count),
        "sample_count": 0,
        "disagreement_count": 0,
        "absolute_logit_delta_sum": 0.0,
        "training": _empty_head_audit(bin_count),
        "deployment": _empty_head_audit(bin_count),
    }


def _empty_head_audit(bin_count: int) -> dict:
    return {
        "sample_count": 0,
        "correct_count": 0,
        "nll_sum": 0.0,
        "brier_sum": 0.0,
        "confidence_sum": 0.0,
        "target_logit_sum": 0.0,
        "margin_sum": 0.0,
        "predicted_old_count": 0,
        "predicted_new_count": 0,
        "predicted_other_count": 0,
        "old_new_logit_gap_sum": 0.0,
        "old_new_logit_gap_count": 0,
        "target_groups": {
            "old": _empty_group_audit(),
            "new": _empty_group_audit(),
        },
        "ece_bins": [
            {"count": 0, "confidence_sum": 0.0, "correct_sum": 0.0}
            for _ in range(bin_count)
        ],
        "per_class": {},
    }


def _empty_group_audit() -> dict:
    return {
        "count": 0,
        "correct_count": 0,
        "nll_sum": 0.0,
        "brier_sum": 0.0,
        "confidence_sum": 0.0,
        "target_logit_sum": 0.0,
        "margin_sum": 0.0,
    }


@torch.no_grad()
def update_dual_head_audit(
    audit: dict,
    training_logits: torch.Tensor,
    deployment_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    old_classes: Iterable[int] = (),
    new_classes: Iterable[int] = (),
) -> None:
    if training_logits.shape != deployment_logits.shape:
        raise ValueError("training and deployment logits must have equal shape")
    if training_logits.ndim != 2 or labels.ndim != 1:
        raise ValueError("expected [batch, classes] logits and [batch] labels")
    if training_logits.shape[0] != labels.numel():
        raise ValueError("logit and label batch sizes differ")
    if labels.numel() == 0:
        return
    if int(labels.min()) < 0 or int(labels.max()) >= training_logits.shape[1]:
        raise ValueError("labels are outside the classifier output range")

    old = sorted({int(value) for value in old_classes})
    new = sorted({int(value) for value in new_classes})
    if set(old).intersection(new):
        raise ValueError("old and new class partitions must be disjoint")

    # Audit values never feed back into the method. Move each small batch to
    # CPU once so the detailed ECE/group/per-class accounting below does not
    # introduce dozens of device synchronizations per training/eval batch.
    if training_logits.device.type != "cpu":
        paired_logits = torch.stack(
            [training_logits.detach(), deployment_logits.detach()], dim=0
        ).cpu()
        training_logits, deployment_logits = paired_logits.unbind(dim=0)
        labels = labels.detach().cpu()

    training_predictions = training_logits.argmax(dim=1)
    deployment_predictions = deployment_logits.argmax(dim=1)
    count = int(labels.numel())
    audit["sample_count"] += count
    audit["disagreement_count"] += int(
        training_predictions.ne(deployment_predictions).sum().cpu()
    )
    audit["absolute_logit_delta_sum"] += float(
        (training_logits - deployment_logits).abs().mean(dim=1).sum().cpu()
    )
    _update_head_audit(audit["training"], training_logits, labels, old, new)
    _update_head_audit(audit["deployment"], deployment_logits, labels, old, new)


@torch.no_grad()
def _update_head_audit(
    audit: dict,
    logits: torch.Tensor,
    labels: torch.Tensor,
    old_classes: list[int],
    new_classes: list[int],
) -> None:
    probabilities = logits.softmax(dim=1)
    predictions = probabilities.argmax(dim=1)
    confidence = probabilities.max(dim=1).values
    losses = F.cross_entropy(logits, labels, reduction="none")
    one_hot = F.one_hot(labels, num_classes=logits.shape[1]).to(probabilities)
    brier = (probabilities - one_hot).square().sum(dim=1)
    rows = torch.arange(labels.numel(), device=labels.device)
    target_logits = logits[rows, labels]
    rivals = logits.clone()
    rivals[rows, labels] = -torch.inf
    margins = target_logits - rivals.max(dim=1).values
    correct = predictions.eq(labels)

    count = int(labels.numel())
    audit["sample_count"] += count
    audit["correct_count"] += int(correct.sum().cpu())
    audit["nll_sum"] += float(losses.sum().cpu())
    audit["brier_sum"] += float(brier.sum().cpu())
    audit["confidence_sum"] += float(confidence.sum().cpu())
    audit["target_logit_sum"] += float(target_logits.sum().cpu())
    audit["margin_sum"] += float(margins.sum().cpu())

    old_tensor = torch.tensor(old_classes, dtype=torch.long, device=labels.device)
    new_tensor = torch.tensor(new_classes, dtype=torch.long, device=labels.device)
    predicted_old = _isin(predictions, old_tensor)
    predicted_new = _isin(predictions, new_tensor)
    audit["predicted_old_count"] += int(predicted_old.sum().cpu())
    audit["predicted_new_count"] += int(predicted_new.sum().cpu())
    audit["predicted_other_count"] += int(
        (~predicted_old & ~predicted_new).sum().cpu()
    )

    target_old = _isin(labels, old_tensor)
    target_new = _isin(labels, new_tensor)
    _update_group(
        audit["target_groups"]["old"],
        target_old,
        correct,
        losses,
        brier,
        confidence,
        target_logits,
        margins,
    )
    _update_group(
        audit["target_groups"]["new"],
        target_new,
        correct,
        losses,
        brier,
        confidence,
        target_logits,
        margins,
    )

    if old_tensor.numel() and new_tensor.numel():
        gap = logits[:, new_tensor].max(dim=1).values - logits[:, old_tensor].max(
            dim=1
        ).values
        audit["old_new_logit_gap_sum"] += float(gap.sum().cpu())
        audit["old_new_logit_gap_count"] += count

    bin_count = len(audit["ece_bins"])
    indices = torch.clamp((confidence * bin_count).long(), max=bin_count - 1)
    for index in range(bin_count):
        mask = indices.eq(index)
        if not bool(mask.any()):
            continue
        row = audit["ece_bins"][index]
        row["count"] += int(mask.sum().cpu())
        row["confidence_sum"] += float(confidence[mask].sum().cpu())
        row["correct_sum"] += float(correct[mask].float().sum().cpu())

    for label in labels.unique(sorted=True).tolist():
        label = int(label)
        mask = labels.eq(label)
        row = audit["per_class"].setdefault(str(label), _empty_group_audit())
        _update_group(
            row,
            mask,
            correct,
            losses,
            brier,
            confidence,
            target_logits,
            margins,
        )


def _isin(values: torch.Tensor, choices: torch.Tensor) -> torch.Tensor:
    if choices.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    return values[:, None].eq(choices[None, :]).any(dim=1)


def _update_group(
    audit: dict,
    mask: torch.Tensor,
    correct: torch.Tensor,
    losses: torch.Tensor,
    brier: torch.Tensor,
    confidence: torch.Tensor,
    target_logits: torch.Tensor,
    margins: torch.Tensor,
) -> None:
    count = int(mask.sum().cpu())
    if count == 0:
        return
    audit["count"] += count
    audit["correct_count"] += int(correct[mask].sum().cpu())
    audit["nll_sum"] += float(losses[mask].sum().cpu())
    audit["brier_sum"] += float(brier[mask].sum().cpu())
    audit["confidence_sum"] += float(confidence[mask].sum().cpu())
    audit["target_logit_sum"] += float(target_logits[mask].sum().cpu())
    audit["margin_sum"] += float(margins[mask].sum().cpu())


def summarize_dual_head_audit(audit: dict) -> dict:
    count = int(audit["sample_count"])
    return {
        "sample_count": count,
        "prediction_disagreement_rate": (
            audit["disagreement_count"] / count if count else None
        ),
        "mean_absolute_logit_delta": (
            audit["absolute_logit_delta_sum"] / count if count else None
        ),
        "training": _summarize_head_audit(audit["training"]),
        "deployment": _summarize_head_audit(audit["deployment"]),
    }


def _summarize_head_audit(audit: dict) -> dict:
    count = int(audit["sample_count"])
    ece = 0.0
    if count:
        for row in audit["ece_bins"]:
            bin_count = int(row["count"])
            if not bin_count:
                continue
            accuracy = row["correct_sum"] / bin_count
            confidence = row["confidence_sum"] / bin_count
            ece += (bin_count / count) * abs(accuracy - confidence)
    return {
        "sample_count": count,
        "accuracy": audit["correct_count"] / count if count else None,
        "nll": audit["nll_sum"] / count if count else None,
        "brier_score": audit["brier_sum"] / count if count else None,
        "mean_confidence": audit["confidence_sum"] / count if count else None,
        "mean_target_logit": audit["target_logit_sum"] / count if count else None,
        "mean_margin": audit["margin_sum"] / count if count else None,
        "ece_15bin": ece if count else None,
        "predicted_old_fraction": (
            audit["predicted_old_count"] / count if count else None
        ),
        "predicted_new_fraction": (
            audit["predicted_new_count"] / count if count else None
        ),
        "predicted_other_fraction": (
            audit["predicted_other_count"] / count if count else None
        ),
        "new_to_old_prediction_ratio": (
            audit["predicted_new_count"] / audit["predicted_old_count"]
            if audit["predicted_old_count"]
            else None
        ),
        "mean_new_minus_old_max_logit": (
            audit["old_new_logit_gap_sum"] / audit["old_new_logit_gap_count"]
            if audit["old_new_logit_gap_count"]
            else None
        ),
        "target_groups": {
            name: _summarize_group(row)
            for name, row in audit["target_groups"].items()
        },
        "per_class": {
            label: _summarize_group(row)
            for label, row in sorted(
                audit["per_class"].items(), key=lambda item: int(item[0])
            )
        },
    }


def _summarize_group(audit: dict) -> dict:
    count = int(audit["count"])
    return {
        "count": count,
        "accuracy": audit["correct_count"] / count if count else None,
        "nll": audit["nll_sum"] / count if count else None,
        "brier_score": audit["brier_sum"] / count if count else None,
        "mean_confidence": audit["confidence_sum"] / count if count else None,
        "mean_target_logit": audit["target_logit_sum"] / count if count else None,
        "mean_margin": audit["margin_sum"] / count if count else None,
    }
