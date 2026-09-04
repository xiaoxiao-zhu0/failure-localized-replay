# D134 Tiny ImageNet uncertainty analysis

Seeds: 218, 219, 220, 221, 222. All statistics use paired seed identities.

## Absolute results

| Method | Metric | Mean | SD | 95% Student-t CI | Per-seed values |
|---|---|---:|---:|---|---|
| persistent_srrd_selective_swap_1 | mean_accuracy | 12.8670 | 0.4479 | [12.3109, 13.4231] | 13.4750, 12.9250, 13.0750, 12.3850, 12.4750 |
| persistent_srrd_selective_swap_1 | worst_forgetting | 38.2580 | 0.8866 | [37.1572, 39.3588] | 39.4800, 37.3000, 38.3700, 38.6400, 37.5000 |
| persistent_srrd_selective_swap_1 | pss | 0.0560 | 0.0190 | [0.0324, 0.0795] | 0.0808, 0.0528, 0.0648, 0.0524, 0.0291 |
| persistent_srrd_selective_swap_1 | hard_accuracy | 11.1080 | 0.2337 | [10.8178, 11.3982] | 11.2300, 11.1300, 11.2700, 10.7000, 11.2100 |
| persistent_srrd_selective_swap_1 | blurry_accuracy | 14.6260 | 0.7688 | [13.6714, 15.5806] | 15.7200, 14.7200, 14.8800, 14.0700, 13.7400 |
| persistent_srrd_prequential_arbitration_1 | mean_accuracy | 11.8700 | 0.4214 | [11.3467, 12.3933] | 12.5750, 11.8600, 11.4650, 11.7850, 11.6650 |
| persistent_srrd_prequential_arbitration_1 | worst_forgetting | 30.3160 | 0.7763 | [29.3521, 31.2799] | 31.3100, 29.1300, 30.3500, 30.3600, 30.4300 |
| persistent_srrd_prequential_arbitration_1 | pss | 0.0377 | 0.0106 | [0.0245, 0.0509] | 0.0465, 0.0370, 0.0491, 0.0330, 0.0229 |
| persistent_srrd_prequential_arbitration_1 | hard_accuracy | 10.4280 | 0.4377 | [9.8845, 10.9715] | 10.8600, 10.1900, 9.8500, 10.3800, 10.8600 |
| persistent_srrd_prequential_arbitration_1 | blurry_accuracy | 13.3120 | 0.6674 | [12.4833, 14.1407] | 14.2900, 13.5300, 13.0800, 13.1900, 12.4700 |

## Full minus Layer 2 paired effects

| Effect | Mean | SD | 95% Student-t CI | Bootstrap 95% CI | Direction | Exact sign-flip p | Per-seed values |
|---|---:|---:|---|---|---|---:|---|
| accuracy_delta_pp | -0.9970 | 0.3816 | [-1.4708, -0.5232] | [-1.3260, -0.7350] | 0/5 positive | 0.0625 | -0.9000, -1.0650, -1.6100, -0.6000, -0.8100 |
| mean_forgetting_improvement_pp | 7.3450 | 0.2435 | [7.0426, 7.6474] | [7.1360, 7.5230] | 5/5 positive | 0.0625 | 6.9850, 7.5050, 7.4250, 7.5900, 7.2200 |
| worst_forgetting_improvement_pp | 7.9420 | 0.4962 | [7.3259, 8.5581] | [7.5020, 8.2140] | 5/5 positive | 0.0625 | 8.1700, 8.1700, 8.0200, 8.2800, 7.0700 |
| pss_relative_change_percent | -30.9864 | 8.7803 | [-41.8886, -20.0842] | [-37.7742, -24.1985] | 0/5 positive | 0.0625 | -42.4505, -29.9242, -24.2284, -37.0229, -21.3058 |
| hard_accuracy_delta_pp | -0.6800 | 0.4873 | [-1.2850, -0.0750] | [-1.1100, -0.3420] | 0/5 positive | 0.0625 | -0.3700, -0.9400, -1.4200, -0.3200, -0.3500 |
| blurry_accuracy_delta_pp | -1.3140 | 0.3374 | [-1.7329, -0.8951] | [-1.5880, -1.0520] | 0/5 positive | 0.0625 | -1.4300, -1.1900, -1.8000, -0.8800, -1.2700 |

Interpretation boundary: five seeds support uncertainty and directional reporting, not broad universal significance claims. The primary paper text should report the per-seed effects and intervals, while the full JSON preserves exact values.
