# D135 matched CORe50 hard/blurry cross-stream confirmation

Protocol checks pass: True

| Method | Mean accuracy (%) | Mean forgetting (%) | PSS (pp) |
|:--|--:|--:|--:|
| A: causal_er_ace | 20.900 +/- 0.613 | 28.039 +/- 1.613 | 4.849 +/- 2.250 |
| B: semantic_proto_hybrid_75_25 | 23.355 +/- 0.433 | 25.295 +/- 2.955 | 5.367 +/- 1.359 |
| C: persistent_srrd_selective_swap_1 | 23.765 +/- 1.655 | 24.827 +/- 1.973 | 7.151 +/- 0.969 |
| D: persistent_srrd_prequential_arbitration_1 | 23.729 +/- 1.916 | 23.685 +/- 1.881 | 5.258 +/- 1.362 |

| Contrast | Accuracy delta (pp) | Forgetting improvement (pp) | PSS delta (pp) |
|:--|--:|--:|--:|
| B-A | +2.455 | +2.744 | +0.518 |
| C-B | +0.409 | +0.467 | +1.783 |
| D-C | -0.035 | +1.142 | -1.893 |
| D-A | +2.829 | +4.354 | +0.409 |
