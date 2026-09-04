# D134 unified corrected Tiny ImageNet matrix

Protocol checks pass: True
Parent-path checks pass: True

## Absolute results

| Method | Accuracy (%) | Mean AF (%) | Worst AF (%) | PSS (pp) |
|:--|--:|--:|--:|--:|
| CE-ACE | 11.911 +/- 0.442 | 34.248 +/- 0.658 | 35.576 +/- 1.026 | 6.566 +/- 1.197 |
| Layer 2 / C | 12.867 +/- 0.448 | 37.218 +/- 0.531 | 38.258 +/- 0.887 | 5.598 +/- 1.897 |
| PRBA | 11.870 +/- 0.421 | 29.873 +/- 0.666 | 30.316 +/- 0.776 | 3.770 +/- 1.060 |
| OBC | 9.713 +/- 0.458 | 25.733 +/- 0.547 | 26.398 +/- 0.827 | 3.028 +/- 1.276 |

## Paired PRBA contrasts

| Contrast | Accuracy delta (pp) | Mean AF improvement (pp) | Worst AF improvement (pp) | PSS delta (pp) |
|:--|--:|--:|--:|--:|
| PRBA - Layer 2 / C | -0.997 | +7.345 | +7.942 | -1.828 |
| PRBA - OBC | +2.157 | -4.140 | -3.918 | +0.742 |
| PRBA - CE-ACE | -0.041 | +4.375 | +5.260 | -2.796 |

The JSON file contains per-seed values, Student-t and bootstrap 95% intervals, exact sign-flip p-values, protocol checks, and parent-path audits.
