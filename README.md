# Failure-Localized Replay: Reproducibility Artifact

This repository accompanies the Pattern Recognition manuscript
"Failure-Localized Replay: Preservation, Exposure, and Prequential Decision
Arbitration for Online Continual Learning."

Version 1.2.0 freezes the code, configurations, analyzer outputs, figures,
complete LPR runs, exact training-environment record, and integrity hashes used
by the current manuscript draft. Public benchmark data are not included.

## Repository and archived version

- GitHub: https://github.com/xiaoxiao-zhu0/failure-localized-replay
- Zenodo v1.2.0 DOI: https://doi.org/10.5281/zenodo.22306368
- Zenodo concept DOI: https://doi.org/10.5281/zenodo.21892632

The version DOI identifies the immutable v1.2.0 artifact. The concept DOI
resolves to the latest published version of the artifact.

## Evidence included

- Split CIFAR-100: primary parent-path, interaction, memory-budget, runtime,
  closest-method, and design-control analyses.
- Tiny ImageNet: the completed D134 normalization-corrected five-seed hard and
  equal-exposure blurry study.
- CORe50: the completed D135 three-seed matched hard/blurry external-validity
  study, including the OBC comparison.
- D136: three-seed random-arbitration and Wilson-gate controls.
- LPR: complete five-seed hard/blurry comparison with all 10 run directories,
  logs, CSV outputs, execution manifest, audit fields, and frozen analysis.

The compact JSON files in `results/analysis/` retain the per-seed values,
aggregates, comparisons, and protocol or parent-path audit outcomes used in the
manuscript. They are the frozen evidence layer for the reported tables.

## Repository layout

- `source/avalanche/`: Avalanche source snapshot used by the experiments.
- `source/rbcl/`: replay, memory, arbitration, audit, evaluation, and LPR code.
- `source/examples/rbcl_run_experiment.py`: experiment entry point.
- `source/scripts/`: analyzers, validation scripts, and matched runners.
- `configs/`: frozen protocol configurations.
- `protocols/`: stream, metric, audit, and resource definitions.
- `results/analysis/`: machine-readable frozen analyzer outputs.
- `results/runs/lpr_baseline_pattern_recognition/`: complete LPR runs and logs.
- `figures/`: publication figures used by the current manuscript.
- `environment/`: exact recovered training environment, full `pip freeze`,
  portable dependency specification, and source hashes.

## Running from the source snapshot

Create an environment from `environment/requirements.txt`, obtain the public
datasets separately, and run commands from the `source/` directory so that the
bundled Avalanche snapshot and `rbcl` package are on `PYTHONPATH`.

Example:

```bash
cd source
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python examples/rbcl_run_experiment.py --help
```

The shell runners accept `PYTHON`, `DATASET_ROOT`, `GPU` or `GPUS`, and output
root overrides. Hardware-specific paths are not embedded in the artifact.

## Main protocol boundary

The main CIFAR-100 protocol uses ten experiences, five epochs per experience,
current and replay minibatches of 64, memory 100, SGD with learning rate 0.05
and momentum 0.9, and paired hard/equal-exposure blurry streams. D134 uses the
same optimization interface on Tiny ImageNet. D135 uses nine CORe50 experiences
and a dataset-adapted learning rate of 0.01. D136 uses seeds 252-254 and the
frozen CIFAR-100 resource interface. LPR uses seeds 218-222 on both paired
streams with the same main optimization protocol.

The artifact supports the bounded claims in the manuscript. It does not claim
universal dominance or formal retention guarantees.

## Integrity

`SHA256SUMS.txt` lists every release file except the checksum file itself. The
GitHub release archive and the Zenodo upload are generated from the same frozen
directory.

## License and citation

Code is distributed under the MIT License. Cite the versioned Zenodo record for
the artifact used in an analysis. Citation metadata are provided in
`CITATION.cff` and `.zenodo.json`.
