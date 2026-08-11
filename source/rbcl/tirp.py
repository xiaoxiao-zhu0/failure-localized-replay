"""Taskification-invariant replay policy utilities.

The functions in this module are shared by offline policy training/auditing
and online deployment.  They intentionally depend only on frozen semantic
features and causal memory metadata.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F


def normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """Return entropy normalized to one for a uniform distribution."""
    probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
    if probabilities.numel() <= 1:
        return probabilities.new_tensor(1.0)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return entropy / math.log(probabilities.numel())


def effective_sample_size_ratio(probabilities: torch.Tensor) -> torch.Tensor:
    """Return effective sample size divided by the number of candidates."""
    probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
    ess = 1.0 / probabilities.square().sum().clamp_min(1e-12)
    return ess / max(1, probabilities.numel())


def enforce_entropy_floor(
    probabilities: torch.Tensor,
    floor_ratio: float = 0.95,
    iterations: int = 32,
) -> torch.Tensor:
    """Mix with uniform just enough to satisfy a normalized entropy floor.

    The bisection coefficient is treated as a projection constant. Gradients
    still flow through the policy probabilities after the projection.
    """
    if not 0.0 <= floor_ratio <= 1.0:
        raise ValueError("floor_ratio must be in [0, 1]")
    probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
    if probabilities.numel() <= 1 or float(normalized_entropy(probabilities)) >= floor_ratio:
        return probabilities
    uniform = torch.full_like(probabilities, 1.0 / probabilities.numel())
    low, high = 0.0, 1.0
    detached = probabilities.detach()
    detached_uniform = uniform.detach()
    for _ in range(iterations):
        middle = (low + high) / 2.0
        candidate = (1.0 - middle) * detached + middle * detached_uniform
        if float(normalized_entropy(candidate)) >= floor_ratio:
            high = middle
        else:
            low = middle
    projected = (1.0 - high) * probabilities + high * uniform
    return projected / projected.sum().clamp_min(1e-12)


def constrained_probabilities(
    logits: torch.Tensor,
    entropy_floor_ratio: float = 0.95,
) -> torch.Tensor:
    return enforce_entropy_floor(torch.softmax(logits, dim=0), entropy_floor_ratio)


def expected_uniform_class_coverage(labels: torch.Tensor, count: int) -> float:
    """Expected distinct-class count under uniform sampling without replacement."""
    if labels.device.type != "cpu":
        labels = labels.detach().cpu()
    population = labels.numel()
    count = min(max(0, int(count)), population)
    if count == 0 or population == 0:
        return 0.0
    denominator = math.comb(population, count)
    expectation = 0.0
    for label in labels.unique():
        class_size = int((labels == label).sum())
        missed = (
            math.comb(population - class_size, count) / denominator
            if population - class_size >= count
            else 0.0
        )
        expectation += 1.0 - missed
    return expectation


def sample_without_replacement(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    count: int,
    min_class_coverage_ratio: float = 1.0,
    generator: torch.Generator | None = None,
    min_class_coverage: int | None = None,
    priority_mask: torch.Tensor | None = None,
    min_priority_count: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted sampling over choices that keep hard floors feasible.

    Constraints are lazy: a candidate is excluded only when selecting it would
    make the class or priority floor impossible to satisfy with the remaining
    replay slots. The returned log probability is the exact probability under
    this constrained sequential policy and therefore supports REINFORCE.
    """
    if probabilities.ndim != 1 or labels.ndim != 1:
        raise ValueError("probabilities and labels must be one-dimensional")
    if probabilities.numel() != labels.numel():
        raise ValueError("probabilities and labels must have equal length")
    if not 0.0 <= min_class_coverage_ratio <= 1.0:
        raise ValueError("min_class_coverage_ratio must be in [0, 1]")
    if priority_mask is not None and priority_mask.shape != labels.shape:
        raise ValueError("priority_mask and labels must have equal shape")
    if probabilities.device.type != "cpu":
        if generator is not None:
            raise ValueError("explicit generators require CPU sampling tensors")
        original_device = probabilities.device
        indices, log_probability = sample_without_replacement(
            probabilities.cpu(),
            labels.detach().cpu(),
            count,
            min_class_coverage_ratio,
            min_class_coverage=min_class_coverage,
            priority_mask=(
                priority_mask.detach().cpu() if priority_mask is not None else None
            ),
            min_priority_count=min_priority_count,
        )
        return indices.to(original_device), log_probability.to(original_device)
    count = min(max(0, int(count)), probabilities.numel())
    if count == 0:
        empty = torch.empty(0, dtype=torch.long, device=probabilities.device)
        return empty, probabilities.sum() * 0.0

    probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
    available = torch.ones(probabilities.numel(), dtype=torch.bool, device=probabilities.device)
    unique_classes = int(labels.unique().numel())
    maximum_coverage = min(count, unique_classes)
    target_coverage = (
        min(maximum_coverage, max(0, int(min_class_coverage)))
        if min_class_coverage is not None
        else min(
            maximum_coverage,
            int(math.ceil(maximum_coverage * min_class_coverage_ratio)),
        )
    )
    priority_mask = (
        priority_mask.bool()
        if priority_mask is not None
        else torch.zeros_like(labels, dtype=torch.bool)
    )
    target_priority = min(
        count,
        int(priority_mask.sum()),
        max(0, int(min_priority_count)),
    )
    class_space = int(labels.max()) + 1
    represented_mask = torch.zeros(
        class_space, dtype=torch.bool, device=labels.device
    )
    represented_count = 0
    selected: list[torch.Tensor] = []
    selected_priority = 0
    log_probability = probabilities.sum() * 0.0

    def choose(eligible: torch.Tensor) -> torch.Tensor:
        nonlocal log_probability, represented_count, selected_priority
        candidate_indices = torch.nonzero(eligible, as_tuple=False).squeeze(1)
        candidate_probabilities = probabilities[candidate_indices]
        if float(candidate_probabilities.sum().detach()) <= 1e-12:
            candidate_probabilities = torch.ones_like(candidate_probabilities)
        candidate_probabilities = candidate_probabilities / candidate_probabilities.sum()
        local_index = torch.multinomial(
            candidate_probabilities.detach(), 1, generator=generator
        ).squeeze(0)
        index = candidate_indices[local_index]
        log_probability = log_probability + candidate_probabilities[local_index].clamp_min(1e-12).log()
        selected.append(index)
        available[index] = False
        label = int(labels[index])
        if not bool(represented_mask[label]):
            represented_mask[label] = True
            represented_count += 1
        selected_priority += int(priority_mask[index])
        return index

    for _ in range(count):
        slots_after = count - len(selected) - 1
        available_indices = torch.nonzero(available, as_tuple=False).squeeze(1)
        available_labels = labels[available_indices]
        class_counts = torch.bincount(
            available_labels, minlength=class_space
        )
        priority_class_counts = torch.bincount(
            labels[available & priority_mask], minlength=class_space
        )
        unseen_class_mask = class_counts.gt(0) & ~represented_mask
        unseen_priority_class_mask = (
            priority_class_counts.gt(0) & ~represented_mask
        )
        unseen_count = int(unseen_class_mask.sum())
        unseen_priority_count = int(unseen_priority_class_mask.sum())
        available_priority = int(priority_mask[available].sum())
        class_gain = (~represented_mask[available_labels]).long()
        priority_gain = priority_mask[available_indices].long()
        needed_classes = (
            target_coverage - represented_count - class_gain
        ).clamp_min(0)
        needed_priority = (
            target_priority - selected_priority - priority_gain
        ).clamp_min(0)
        unseen_classes_after = unseen_count - class_gain
        priority_after = available_priority - priority_gain
        overlap_after = unseen_priority_count - (
            class_gain.bool()
            & priority_class_counts[available_labels].gt(0)
        ).long()
        maximum_overlap = torch.minimum(
            torch.minimum(needed_classes, needed_priority), overlap_after
        )
        minimum_slots = needed_classes + needed_priority - maximum_overlap
        feasible_candidates = (
            needed_classes.le(slots_after)
            & needed_classes.le(unseen_classes_after)
            & needed_priority.le(slots_after)
            & needed_priority.le(priority_after)
            & minimum_slots.le(slots_after)
        )
        feasible = torch.zeros_like(available)
        feasible[available_indices] = feasible_candidates
        if not bool(feasible.any()):
            raise RuntimeError("replay coverage constraints became infeasible")
        choose(feasible)

    if represented_count < target_coverage or selected_priority < target_priority:
        raise RuntimeError("replay sampler failed to satisfy hard coverage floors")

    return torch.stack(selected), log_probability / count


def rbf_bandwidth(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Median-heuristic bandwidth computed from complete memory states."""
    combined = F.normalize(torch.cat((first.float(), second.float()), dim=0), dim=1)
    squared = torch.cdist(combined, combined).square()
    positive = squared.detach()[squared.detach() > 0]
    return positive.median() if positive.numel() else squared.new_tensor(1.0)


def rbf_mmd(
    first: torch.Tensor,
    second: torch.Tensor,
    bandwidth: torch.Tensor | None = None,
) -> torch.Tensor:
    """Biased RBF maximum mean discrepancy between full sample sets."""
    first = F.normalize(first.float(), dim=1)
    second = F.normalize(second.float(), dim=1)
    combined = torch.cat((first, second), dim=0)
    squared = torch.cdist(combined, combined).square()
    if bandwidth is None:
        bandwidth = rbf_bandwidth(first, second)
    bandwidth = bandwidth.clamp_min(1e-6)
    kernel = torch.exp(-squared / (2.0 * bandwidth))
    size = first.shape[0]
    return (kernel[:size, :size].mean() + kernel[size:, size:].mean()
            - 2.0 * kernel[:size, size:].mean()).clamp_min(0.0)


def class_conditional_mmd(
    first: torch.Tensor,
    first_labels: torch.Tensor,
    second: torch.Tensor,
    second_labels: torch.Tensor,
    bandwidth: torch.Tensor | None = None,
) -> torch.Tensor:
    shared = sorted(set(first_labels.tolist()) & set(second_labels.tolist()))
    if not shared:
        return rbf_mmd(first, second, bandwidth)
    losses = [
        rbf_mmd(
            first[first_labels == label],
            second[second_labels == label],
            bandwidth,
        )
        for label in shared
    ]
    return torch.stack(losses).mean()


def class_mass_tv(
    first_labels: torch.Tensor,
    second_labels: torch.Tensor,
) -> torch.Tensor:
    maximum = max(
        int(first_labels.max()) if first_labels.numel() else 0,
        int(second_labels.max()) if second_labels.numel() else 0,
    ) + 1
    first = torch.bincount(first_labels, minlength=maximum).float()
    second = torch.bincount(second_labels, minlength=maximum).float()
    first /= first.sum().clamp_min(1.0)
    second /= second.sum().clamp_min(1.0)
    return 0.5 * (first - second).abs().sum()


def age_wasserstein(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """One-dimensional Wasserstein distance for equal-size replay batches."""
    if first.numel() != second.numel():
        raise ValueError("age_wasserstein expects equal-size replay batches")
    if not first.numel():
        return first.new_tensor(0.0)
    return (first.float().sort().values - second.float().sort().values).abs().mean()


def semantic_diversity(features: torch.Tensor) -> torch.Tensor:
    if features.shape[0] <= 1:
        return features.new_tensor(0.0)
    features = F.normalize(features.float(), dim=1)
    distance = 1.0 - features @ features.T
    mask = ~torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    return distance[mask].mean()


def boundary_coverage(quality: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """Fraction of selected items in the memory's least-typical quartile."""
    if not selected.numel():
        return quality.new_tensor(0.0)
    threshold = torch.quantile(quality.float(), 0.25)
    return (quality[selected].float() <= threshold).float().mean()


def replay_batch_metrics(
    first: Mapping[str, torch.Tensor | float | int],
    second: Mapping[str, torch.Tensor | float | int],
    first_indices: torch.Tensor,
    second_indices: torch.Tensor,
    bandwidth: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    first_z = first["z"][first_indices]
    second_z = second["z"][second_indices]
    first_y = first["y"][first_indices]
    second_y = second["y"][second_indices]
    first_age = first["age"][first_indices] / max(1.0, float(first["clock"]))
    second_age = second["age"][second_indices] / max(1.0, float(second["clock"]))
    return {
        "semantic_mmd": rbf_mmd(first_z, second_z, bandwidth),
        "class_conditional_mmd": class_conditional_mmd(
            first_z, first_y, second_z, second_y, bandwidth
        ),
        "class_mass_tv": class_mass_tv(first_y, second_y),
        "age_wasserstein": age_wasserstein(first_age, second_age),
        "first_diversity": semantic_diversity(first_z),
        "second_diversity": semantic_diversity(second_z),
        "first_boundary_coverage": boundary_coverage(first["quality"], first_indices),
        "second_boundary_coverage": boundary_coverage(second["quality"], second_indices),
        "first_class_coverage": first_y.unique().numel() / max(1, first_indices.numel()),
        "second_class_coverage": second_y.unique().numel() / max(1, second_indices.numel()),
    }
