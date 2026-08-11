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


The public repository and its versioned Zenodo archive are:


- GitHub: `https://github.com/xiaoxiao-zhu0/failure-localized-replay`
- Zenodo v1.0.0: `https://doi.org/10.5281/zenodo.21892633`


The DOI resolves to the exact v1.0.0 artifact associated with the manuscript.


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
