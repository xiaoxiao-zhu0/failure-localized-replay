# D135 matched CORe50 OBC comparison

Protocol checks pass: True

| Method | Mean accuracy (%) | Mean forgetting (%) | PSS (pp) |
|:--|--:|--:|--:|
| A: CE-ACE | 20.900 +/- 0.613 | 28.039 +/- 1.613 | 4.849 +/- 2.250 |
| B: Layer 1 | 23.355 +/- 0.433 | 25.295 +/- 2.955 | 5.367 +/- 1.359 |
| C: Layer 2 | 23.765 +/- 1.655 | 24.827 +/- 1.973 | 7.151 +/- 0.969 |
| D: PRBA | 23.729 +/- 1.916 | 23.685 +/- 1.881 | 5.258 +/- 1.362 |
| OBC | 23.748 +/- 1.802 | 22.439 +/- 1.931 | 5.522 +/- 1.546 |

| Contrast | Accuracy delta (pp) | Forgetting improvement (pp) | PSS delta (pp) |
|:--|--:|--:|--:|
| OBC - D: PRBA | +0.019 | +1.246 | +0.265 |
| OBC - C: Layer 2 | -0.016 | +2.389 | -1.628 |
| OBC - A: CE-ACE | +2.848 | +5.600 | +0.673 |
