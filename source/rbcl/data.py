"""Dataset construction for RBCL experiments.

This module maps the paper's "Datasets" section to Avalanche native
benchmarks. The custom part is only the uniform wrapper used by our scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from avalanche.benchmarks.classic import (
    CORe50,
    PermutedMNIST,
    RotatedMNIST,
    SplitCIFAR10,
    SplitCIFAR100,
    SplitMNIST,
    SplitTinyImageNet,
)
from avalanche.benchmarks.scenarios.validation_scenario import (
    benchmark_with_validation_stream,
)
from avalanche.benchmarks.scenarios import benchmark_from_datasets


def _equal_exposure_blurry_train_datasets(
    train_sources, *, seed: int, seed_offset: int
):
    """Delay 20% of each non-final experience without dropping or repeating data."""
    if len(train_sources) < 2:
        raise ValueError("equal-exposure blurry streams require at least two experiences")

    own_parts = []
    deferred_parts = []
    for index, dataset in enumerate(train_sources):
        generator = torch.Generator().manual_seed(seed_offset + 17 * seed + index)
        permutation = torch.randperm(len(dataset), generator=generator).tolist()
        if index == len(train_sources) - 1:
            own_parts.append(dataset.subset(permutation))
            deferred_parts.append(None)
            continue
        own_count = int(round(0.8 * len(dataset)))
        own_parts.append(dataset.subset(permutation[:own_count]))
        deferred_parts.append(dataset.subset(permutation[own_count:]))

    train_datasets = [own_parts[0]]
    for index in range(1, len(train_sources)):
        train_datasets.append(deferred_parts[index - 1].concat(own_parts[index]))
    return train_datasets


def apply_sample_clock(benchmark, *, clock_samples: int):
    """Ignore incoming experience ends and emit fixed-size sample windows.

    This is an online-compatible adapter: it only concatenates the arrival
    order of the training stream and emits a window when ``clock_samples``
    items have accumulated. It never inspects labels or task IDs. The final
    partial window is retained rather than dropped.
    """
    if clock_samples <= 0:
        raise ValueError("clock_samples must be positive")
    train_datasets = [experience.dataset for experience in benchmark.train_stream]
    if not train_datasets:
        raise ValueError("Cannot apply a sample clock to an empty train stream")
    flat = train_datasets[0]
    for dataset in train_datasets[1:]:
        flat = flat.concat(dataset)
    clocked_train = [
        flat.subset(range(start, min(start + clock_samples, len(flat))))
        for start in range(0, len(flat), clock_samples)
    ]
    return benchmark_from_datasets(
        train=clocked_train,
        test=[experience.dataset for experience in benchmark.test_stream],
    )


def apply_deferred_sample_clock(
    benchmark, *, clock_samples: int, defer_samples: int
):
    """Emit a task-ID-free sample clock with a bounded one-clock delay.

    Samples arrive in their original order. The last ``defer_samples`` items
    of each raw clock are kept locally and first trained together with the
    beginning of the next clock. This creates a fixed, boundary-independent
    transition exposure without reading labels or incoming experience IDs.
    Every raw sample is emitted for its first training exactly once; the final
    delayed tail is released at stream end.
    """
    if clock_samples <= 0:
        raise ValueError("clock_samples must be positive")
    if not 0 < defer_samples < clock_samples:
        raise ValueError("defer_samples must be in (0, clock_samples)")

    train_datasets = [experience.dataset for experience in benchmark.train_stream]
    if not train_datasets:
        raise ValueError("Cannot apply a deferred clock to an empty train stream")
    flat = train_datasets[0]
    for dataset in train_datasets[1:]:
        flat = flat.concat(dataset)

    raw_windows = [
        flat.subset(range(start, min(start + clock_samples, len(flat))))
        for start in range(0, len(flat), clock_samples)
    ]
    emitted = []
    delayed = None
    for index, window in enumerate(raw_windows):
        is_last = index == len(raw_windows) - 1
        if is_last:
            emitted.append(window if delayed is None else delayed.concat(window))
            continue

        split = len(window) - defer_samples
        if split <= 0:
            raise ValueError("A raw clock window must exceed defer_samples")
        ready = window.subset(range(split))
        emitted.append(ready if delayed is None else delayed.concat(ready))
        delayed = window.subset(range(split, len(window)))

    return benchmark_from_datasets(
        train=emitted,
        test=[experience.dataset for experience in benchmark.test_stream],
    )


def build_benchmark(
    name: str,
    *,
    n_experiences: int,
    seed: int,
    dataset_root: Optional[str] = None,
    return_task_id: bool = False,
    validation_fraction: float = 0.0,
):
    """Build an Avalanche benchmark from a short experiment name.

    Paper mapping:
    - SplitMNIST is the minimal demo / sanity-check benchmark.
    - SplitCIFAR100, SplitTinyImageNet, and CORe50 are the main-paper targets.
    """
    root = Path(dataset_root) if dataset_root else None
    key = name.lower().replace("-", "_")

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")

    if key == "split_mnist":
        benchmark = SplitMNIST(
            n_experiences=n_experiences,
            return_task_id=return_task_id,
            seed=seed,
            dataset_root=root,
        )
    elif key == "permuted_mnist":
        benchmark = PermutedMNIST(
            n_experiences=n_experiences,
            seed=seed,
            dataset_root=root,
        )
    elif key == "rotated_mnist":
        benchmark = RotatedMNIST(
            n_experiences=n_experiences,
            seed=seed,
            dataset_root=root,
        )
    elif key == "split_cifar10":
        benchmark = SplitCIFAR10(
            n_experiences=n_experiences,
            return_task_id=return_task_id,
            seed=seed,
            dataset_root=root,
        )
    elif key == "blurry_split_cifar10":
        if n_experiences != 10:
            raise ValueError("blurry_split_cifar10 requires n_experiences=10")
        # Construct a fixed-length stream in which each new class keeps 80% of
        # its samples and the immediately preceding class occupies the other
        # 20%. The underlying class order remains identical to SplitCIFAR10.
        base = SplitCIFAR10(
            n_experiences=10,
            return_task_id=return_task_id,
            seed=seed,
            dataset_root=root,
        )
        train_sources = [experience.dataset for experience in base.train_stream]
        train_datasets = []
        for index, dataset in enumerate(train_sources):
            generator = torch.Generator().manual_seed(30_000 + 17 * seed + index)
            permutation = torch.randperm(len(dataset), generator=generator)
            if index == 0:
                train_datasets.append(dataset.subset(permutation.tolist()))
                continue
            current_count = int(round(0.8 * len(dataset)))
            current = dataset.subset(permutation[:current_count].tolist())
            previous_dataset = train_sources[index - 1]
            previous_generator = torch.Generator().manual_seed(
                40_000 + 17 * seed + index
            )
            previous_indices = torch.randperm(
                len(previous_dataset), generator=previous_generator
            )[: len(dataset) - current_count]
            train_datasets.append(current.concat(previous_dataset.subset(previous_indices.tolist())))
        benchmark = benchmark_from_datasets(
            train=train_datasets,
            test=[experience.dataset for experience in base.test_stream],
        )
    elif key == "equal_exposure_blurry_cifar10":
        if n_experiences != 10:
            raise ValueError("equal_exposure_blurry_cifar10 requires n_experiences=10")
        # A controlled boundary-form intervention.  Unlike
        # ``blurry_split_cifar10``, this stream neither duplicates nor drops a
        # training example: classes 0--8 contribute 80% to their own segment
        # and defer the remaining 20% to the next one.  The last class stays
        # fully in the final segment, which receives the deferred 20% from
        # class 8.  Thus both the aggregate sample exposure and per-class
        # sample counts exactly match SplitCIFAR10.
        base = SplitCIFAR10(
            n_experiences=10,
            return_task_id=return_task_id,
            seed=seed,
            dataset_root=root,
        )
        train_sources = [experience.dataset for experience in base.train_stream]
        train_datasets = _equal_exposure_blurry_train_datasets(
            train_sources, seed=seed, seed_offset=50_000
        )
        benchmark = benchmark_from_datasets(
            train=train_datasets,
            test=[experience.dataset for experience in base.test_stream],
        )
    elif key in {
        "equal_exposure_blurry_cifar100",
        "equal_exposure_blurry_tinyimagenet",
    }:
        # Dataset-general version of the controlled boundary-form intervention
        # above. Every original sample occurs once: 80% of each non-final
        # source experience stays in place and its remaining 20% is deferred
        # into the next experience. This changes only boundary form, not total
        # exposure, class counts, or the underlying class order.
        base = (
            SplitCIFAR100(
                n_experiences=n_experiences,
                return_task_id=return_task_id,
                seed=seed,
                dataset_root=root,
            )
            if key == "equal_exposure_blurry_cifar100"
            else SplitTinyImageNet(
                n_experiences=n_experiences,
                return_task_id=return_task_id,
                seed=seed,
                dataset_root=root,
            )
        )
        train_sources = [experience.dataset for experience in base.train_stream]
        if len(train_sources) < 2:
            raise ValueError(f"{key} requires at least two experiences")
        train_datasets = _equal_exposure_blurry_train_datasets(
            train_sources, seed=seed, seed_offset=60_000
        )
        benchmark = benchmark_from_datasets(
            train=train_datasets,
            test=[experience.dataset for experience in base.test_stream],
        )
    elif key in {"sample_clock_hard_cifar10", "sample_clock_blurry_cifar10"}:
        if n_experiences != 10:
            raise ValueError(f"{key} requires n_experiences=10")
        # Causal clock intervention. Both variants use the exact same ordered
        # sample stream and are cut every 5,000 samples. The blurry version is
        # first represented with its 80%/20% adjacent segments, then flattened
        # before the fixed sample-clock cut. Hence any prior hard/blurry
        # difference caused by *where an experience ends* is removed without
        # changing a sample's occurrence count, label, or temporal order.
        base = SplitCIFAR10(
            n_experiences=10,
            return_task_id=return_task_id,
            seed=seed,
            dataset_root=root,
        )
        train_sources = [experience.dataset for experience in base.train_stream]
        ordered_sources = []
        blurry_segments = []
        deferred = None
        for index, dataset in enumerate(train_sources):
            generator = torch.Generator().manual_seed(50_000 + 17 * seed + index)
            permutation = torch.randperm(len(dataset), generator=generator).tolist()
            ordered = dataset.subset(permutation)
            ordered_sources.append(ordered)
            if index == 0:
                blurry_segments.append(ordered.subset(range(int(round(0.8 * len(ordered))))))
            elif index == len(train_sources) - 1:
                blurry_segments.append(deferred.concat(ordered))
            else:
                own_count = int(round(0.8 * len(ordered)))
                blurry_segments.append(deferred.concat(ordered.subset(range(own_count))))
            if index < len(train_sources) - 1:
                own_count = int(round(0.8 * len(ordered)))
                deferred = ordered.subset(range(own_count, len(ordered)))

        def flatten(datasets):
            merged = datasets[0]
            for source in datasets[1:]:
                merged = merged.concat(source)
            return merged

        flat = flatten(
            ordered_sources if key == "sample_clock_hard_cifar10" else blurry_segments
        )
        if len(flat) % n_experiences != 0:
            raise ValueError("Fixed sample-clock chunks must have equal size")
        chunk_size = len(flat) // n_experiences
        train_datasets = [
            flat.subset(range(start, start + chunk_size))
            for start in range(0, len(flat), chunk_size)
        ]
        benchmark = benchmark_from_datasets(
            train=train_datasets,
            test=[experience.dataset for experience in base.test_stream],
        )
    elif key == "split_cifar100":
        benchmark = SplitCIFAR100(
            n_experiences=n_experiences,
            return_task_id=return_task_id,
            seed=seed,
            dataset_root=root,
        )
    elif key == "split_tinyimagenet":
        benchmark = SplitTinyImageNet(
            n_experiences=n_experiences,
            return_task_id=return_task_id,
            seed=seed,
            dataset_root=root,
        )
    elif key in {
        "core50",
        "core50_mini",
        "equal_exposure_blurry_core50",
        "equal_exposure_blurry_core50_mini",
    }:
        if n_experiences != 9:
            raise ValueError(f"{key} requires n_experiences=9 for CORe50 NC")
        benchmark = CORe50(
            scenario="nc",
            run=seed % 10,
            object_lvl=True,
            mini=key.endswith("_mini"),
            dataset_root=root,
        )
        if key.startswith("equal_exposure_blurry_"):
            benchmark = benchmark_from_datasets(
                train=_equal_exposure_blurry_train_datasets(
                    [experience.dataset for experience in benchmark.train_stream],
                    seed=seed,
                    seed_offset=70_000,
                ),
                test=[experience.dataset for experience in benchmark.test_stream],
            )
    else:
        raise ValueError(
            f"Unknown benchmark '{name}'. Supported: split_mnist, permuted_mnist, "
            "rotated_mnist, split_cifar10, blurry_split_cifar10, "
            "equal_exposure_blurry_cifar10, sample_clock_hard_cifar10, "
            "sample_clock_blurry_cifar10, split_cifar100, "
            "equal_exposure_blurry_cifar100, split_tinyimagenet, "
            "equal_exposure_blurry_tinyimagenet, core50, core50_mini, "
            "equal_exposure_blurry_core50, equal_exposure_blurry_core50_mini."
        )

    if validation_fraction > 0:
        benchmark = benchmark_with_validation_stream(
            benchmark,
            validation_size=validation_fraction,
            shuffle=True,
            seed=seed,
        )
    return benchmark
