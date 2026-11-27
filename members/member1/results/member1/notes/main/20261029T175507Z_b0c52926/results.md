# Member 1 formal results

Batch: `20261029T175507Z_b0c52926`. Three formal seeds 1/2/3; frozen settings; final averaged-model test evaluation.

| Task | Optimizer | Test metric | Mean ± sample SD | Completed | Validation CE |
| --- | --- | --- | --- | --- | --- |
| fashion_mnist | sgd | accuracy_pct | 84.14667 ± 2.66740 | 3/3 | 0.39695 ± 0.12958 |
| fashion_mnist | dog | accuracy_pct | 85.46333 ± 0.33501 | 3/3 | 0.32456 ± 0.00148 |
| cifar10 | sgd | accuracy_pct | 58.44667 ± 0.64594 | 3/3 | 1.28057 ± 0.00774 |
| cifar10 | dog | accuracy_pct | 57.92667 ± 0.78360 | 3/3 | 1.24342 ± 0.01616 |
| tiny_shakespeare | sgd | CE_nats_per_character | 1.88516 ± 0.01118 | 3/3 | 1.77671 ± 0.00374 |
| tiny_shakespeare | dog | CE_nats_per_character | 1.89892 ± 0.00692 | 3/3 | 1.72509 ± 0.00801 |

## Target attainment (conditional means)

| Task | Optimizer | Target CE | Reached | Mean updates among reached | Mean seconds among reached |
| --- | --- | ---: | --- | ---: | ---: |
| fashion_mnist | sgd | 0.5 | 3/3 | 300 | 1.3020439583318268 |
| fashion_mnist | dog | 0.5 | 3/3 | 433.3333333333333 | 1.9931403753337993 |
| cifar10 | sgd | 1.5 | 3/3 | 666.6666666666666 | 22.077576138333825 |
| cifar10 | dog | 1.5 | 3/3 | 1433.3333333333333 | 46.565409666664586 |
| tiny_shakespeare | sgd | 2.0 | 3/3 | 1100 | 12.135090750334106 |
| tiny_shakespeare | dog | 2.0 | 3/3 | 566.6666666666666 | 6.401804777667469 |

Unreached runs retain null target times/steps and are counted above. Failed runs remain in main_run_inventory.csv; means use completed runs only. Training CE uses the fixed training subset. Test CE is nats/character for text; image test accuracy is percent. Mean ± sample SD is descriptive, not a significance test.

The mechanism diagnostic is separate: see quadratic_findings.md and tables/diagnostic_results.csv. The main figure contains formal task curves only; the quadratic plots remain companion figures.
