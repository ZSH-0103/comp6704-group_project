# Quadratic results

Batch: `20261029T085102Z_a0a3a0d1`; protocol hash: `855f7779ef86`.

Exact gradients; raw iterates; 500 updates; seeds [1, 2, 3] vary starting direction. Target: relative gap <= 1e-06.

| Condition | Method | SGD lr | Final relative gap, mean ± sample SD | Reached | Mean target updates (reached only) |
| --- | --- | ---: | ---: | ---: | ---: |
| k1_a1 | Tuned SGD | 1 | 0 ± 0 | 3/3 | 1.0 |
| k1_a1 | DoG | adaptive | 3.751e-50 ± 6.13e-60 | 3/3 | 125.0 |
| k10_a1 | Tuned SGD | 0.18 | 4.377e-90 ± 3.05e-90 | 3/3 | 27.3 |
| k10_a1 | DoG | adaptive | 4.841e-13 ± 5.34e-13 | 3/3 | 256.7 |
| k100_a1 | Tuned SGD | 0.0195 | 4.05e-13 ± 2.01e-13 | 3/3 | 179.3 |
| k100_a1 | DoG | adaptive | 0.0003706 ± 0.000136 | 0/3 | unreached |
| k10_a0.1 | Tuned SGD | 1.8 | 4.377e-90 ± 3.05e-90 | 3/3 | 27.3 |
| k10_a0.1 | DoG | adaptive | 4.841e-13 ± 5.34e-13 | 3/3 | 256.7 |
| k10_a10 | Tuned SGD | 0.018 | 4.377e-90 ± 3.05e-90 | 3/3 | 27.3 |
| k10_a10 | DoG | adaptive | 4.841e-13 ± 5.34e-13 | 3/3 | 256.7 |

Comparison failures: 0/30. Tuning failures: 5/45.
Tuning cost: 45 SGD trials; 0.356 measured update/evaluation seconds. DoG defaults were not searched.
Timing excludes problem setup and file writes; tiny CPU timings should not be used as a speed ranking.

See quadratic_findings.md for interpretation and limitations. No neural-network/main results are included.
