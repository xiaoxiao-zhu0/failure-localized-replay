# LPR CIFAR-100 Baseline Summary

- Status: completed, 10/10 runs, all protocol and LPR audit checks passed.
- Seeds: 218--222; hard and equal-exposure blurry Split CIFAR-100.
- Runtime is descriptive because two runs were executed concurrently.

| Method | Acquisition | Accuracy | Mean AF | Worst AF | BWT | PSS |
|---|---:|---:|---:|---:|---:|---:|
| LPR | 62.534 +/- 2.195 | 10.539 +/- 0.824 | 51.995 +/- 1.985 | 55.406 +/- 2.955 | -51.995 +/- 1.985 | 11.756 +/- 2.087 |

## Differences Relative to Existing Methods

Positive accuracy/BWT deltas favor LPR. Positive AF/PSS improvements mean that LPR has lower forgetting or lower stream sensitivity.

| Reference | Accuracy delta | Mean AF improvement | Worst AF improvement | BWT delta | PSS improvement |
|---|---:|---:|---:|---:|---:|
| C | -9.401 | -16.132 | -18.100 | -16.132 | -6.062 |
| PRBA | -8.861 | -19.553 | -21.224 | -19.553 | -5.000 |
| OBC | -7.895 | -21.372 | -22.466 | -21.372 | -4.190 |
| DER++ | -1.615 | +1.517 | +2.692 | +1.495 | +2.328 |
| MIR | +0.532 | -3.994 | -5.926 | -4.111 | -4.800 |

## Resource Audit

- Mean elapsed time per run: 738.7 s.
- Mean peak CUDA memory: 694.0 MiB.
- Mean preconditioner updates: 119.5.
- Mean preconditioned updates: 3572.5.
- Mean extra replay-memory scans: 119.5.

The exact numerical evidence and per-seed checks are retained in the JSON report.
