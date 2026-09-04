# Version 1.2.0

This release completes the LPR evidence and environment audit for the current
Pattern Recognition manuscript.

## Added

- All 10 completed LPR hard/blurry CIFAR-100 run directories for seeds 218-222.
- LPR logs, per-run outputs, execution manifest, JSON/Markdown analysis, and
  resource audit.
- Exact recovered server environment and complete `pip freeze`.
- Server-matched SHA-256 hashes for the LPR implementation, trainer, and runner.
- Frozen LPR protocol configuration.

## Updated

- LPR source and runner aligned byte-for-byte with the completed server runs.
- Dataset, metric, audit, resource, citation, and Zenodo metadata.
- Data Availability wording and release-wide SHA-256 manifest.

## Boundary

D135 and D136 remain three-seed descriptive studies. LPR runtime is descriptive
because two runs were executed concurrently. Public benchmark datasets and
temporary training artifacts are excluded. The historical training commit is
unavailable because the server experiment directory contained no `.git`
metadata; the source snapshot and hashes are frozen instead.
