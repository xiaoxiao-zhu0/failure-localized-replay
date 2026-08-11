# Failure-Localized Replay: Reproducibility Artifact

This artifact accompanies the Pattern Recognition manuscript
"Failure-Localized Replay: Preservation, Exposure, and Prequential Decision
Arbitration for Online Continual Learning."

The artifact contains the source code for the replay components, the frozen
analysis outputs used in the manuscript, experiment protocols, configuration
values, and publication figures. Public datasets are not redistributed. The
experiments use CIFAR-100 and Tiny ImageNet obtained from their official or
standard public sources.

## Repository and DOI

The GitHub repository URL and the permanent Zenodo DOI must be inserted here
after the public release is created:

- GitHub: `https://github.com/xiaoxiao-zhu0/failure-localized-replay`
- Zenodo: `https://doi.org/10.5281/zenodo/REPLACE_WITH_RECORD_ID`

Do not submit the manuscript until both identifiers resolve publicly.

## Contents

- `source/avalanche/`: Avalanche source snapshot used by the experiments.
- `source/rbcl/`: replay, memory, arbitration, audit, and evaluation code.
- `source/examples/rbcl_run_experiment.py`: experiment entry point.
- `source/scripts/`: frozen analyzers and validation scripts for the reported studies.
- `configs/`: exact main settings and seed assignments.
- `protocols/`: dataset, stream, metric, audit, and resource protocols.
- `results/analysis/`: compact frozen analyzer outputs cited by the paper.
- `results/raw_summaries/`: optional per-run summaries for Zenodo; these are not
  intended to be committed to the GitHub source repository.
- `figures/`: figures used in the manuscript.
- `environment/`: Python dependency specification.

## Main experimental settings

- Dataset protocols: ten-experience online class-incremental Split CIFAR-100,
  hard and equal-exposure blurry streams; Tiny ImageNet boundary study.
- Optimizer: SGD, learning rate 0.05, momentum 0.9, no weight decay.
- Training: five epochs per experience, current minibatch 64, replay minibatch 64.
- Memory: 100 for the main study; 50 and 200 for memory sensitivity.
- Hybrid memory: 75% semantic coverage and 25% reservoir sampling.
- Layer 2: EMA decay 0.99, Wilson z=1.96, residual threshold 0.5, at most one swap.
- Main factorial seeds: 260, 261, 262, 263, 264.
- Evaluation: prequential test-then-train order, final accuracy, mean and worst
  forgetting, BWT, and PSS.

## Verifying frozen statistics

The D133 paired statistics are already included in
`results/analysis/d133_paired_statistics.json`. The analyzer can be rerun with:

```text
python source/scripts/analyze_d133_paired_statistics.py ^
  --input results/analysis/d133_layer2_prba_interaction.json ^
  --output results/analysis/d133_paired_statistics.recomputed.json ^
  --figure figures/d133_paired_effects.recomputed.svg
```

The command uses the five paired effects and reproduces the exhaustive
bootstrap percentile intervals, exact sign-flip p-values, Cohen's d_z, and
direction counts reported in the paper.

## Re-running training

Training requires a CUDA-capable PyTorch environment and local copies of the
public datasets. The shell runners in the original project document the
server execution commands, but hardware-specific paths must be replaced with
local paths. Re-running training is not required to verify the frozen paper
statistics.

## Reproducibility boundary

The artifact does not claim that five seeds establish a universal population
effect. The paper reports the small-sample limitation, the failed forgetting
magnitude gate, the Tiny ImageNet accuracy boundary miss, OBC's more
retention-aggressive operating point, and PRBA's serial runtime overhead.

## License and citation

Code is distributed under the license in `LICENSE`. Please cite the Zenodo
record after its DOI is assigned. The DOI record should preserve the exact
release associated with the submitted manuscript.
