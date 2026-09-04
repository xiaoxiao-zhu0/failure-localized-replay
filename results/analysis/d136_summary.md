# D136 Design-Choice Controls

| Variant | Accuracy | Mean AF | Worst AF | PSS |
|---|---:|---:|---:|---:|
| Random arbitration | 19.872 +/- 0.348 | 36.197 +/- 0.321 | 37.357 +/- 0.398 | 6.837 +/- 0.942 |
| Layer 2 without Wilson gate | 20.125 +/- 1.030 | 38.270 +/- 0.590 | 38.807 +/- 1.032 | 6.063 +/- 2.422 |
| PRBA without Wilson gate | 19.252 +/- 0.974 | 35.268 +/- 0.545 | 36.070 +/- 1.236 | 6.500 +/- 2.249 |

All values are mean +/- sample SD over seeds 252--254 after combining the paired hard and blurry streams by the registered metric definitions.

## Audit

- Random arbitration: PASS
- Layer 2 without Wilson gate: PASS
- PRBA without Wilson gate: PASS
