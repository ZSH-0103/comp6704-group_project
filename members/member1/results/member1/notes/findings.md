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

## Hypothesis and interpretation

The study asks whether global DoG can reduce external learning-rate search while achieving comparable task quality to SGD selected from a declared five-candidate grid. Formal results use independent seeds 1/2/3; tuning seed 11 is excluded. No test result was used to change settings.

- fashion_mnist: DoG minus SGD mean test metric = +1.31667 accuracy_pct. Higher is better for image accuracy; lower is better for text CE. Report SD and all three seeds alongside this difference; this is not a significance test.
- cifar10: DoG minus SGD mean test metric = -0.52000 accuracy_pct. Higher is better for image accuracy; lower is better for text CE. Report SD and all three seeds alongside this difference; this is not a significance test.
- tiny_shakespeare: DoG minus SGD mean test metric = +0.01376 CE_nats_per_character. Higher is better for image accuracy; lower is better for text CE. Report SD and all three seeds alongside this difference; this is not a significance test.

## Costs, failures and mechanism

Completed 18/18 formal runs. SGD tuning trial cost is recorded in cost_stability.csv and retained separately from formal training time. DoG used fixed defaults with zero search trials. Target-time means are conditional on reaching the threshold; the reached counts must accompany them.
The quadratic experiment is separate evidence about conditioning and scaling; see quadratic_findings.md. It uses exact gradients and raw iterates, whereas the neural study uses minibatches and averaged evaluation parameters. Its theoretical implications cannot be transferred automatically.

## Limitations

Only three formal seeds; small CPU models and subsets; fixed learning rates for SGD; five-candidate search with accepted upper-bound winners on Fashion/text; final-budget evaluation instead of best-checkpoint selection; no broad claim of optimizer optimality. Training CE is measured on a fixed subset, validation is complete, and time-to-target has 100-update evaluation resolution. Absolute timing comparisons are within this Mac only.

## Shared-table handoff

- tab:methods: plain SGD and independently implemented ordinary global DoG; DoG keeps an initial parameter snapshot and scalar accumulator, not per-coordinate preconditioning. Both updates have O(d) work; DoG's persistent initial snapshot is O(d).
- tab:protocol: frozen settings in docs/FROZEN_SETTINGS_MEMBER1.md and configs/member1.json; SGD rates {'fashion_mnist': {'lr': 0.3}, 'cifar10': {'lr': 0.1}, 'tiny_shakespeare': {'lr': 1.0}}; DoG {'reps_rel': 1e-06, 'eps': 1e-08}; averaging gamma=8, final-budget checkpoint.
- tab:diagnostic-plan: independent quadratic conditioning/scale sweeps; settings, measurements and limitations in quadratic_findings.md.
- tab:hardware: use environment.json from the main raw directories; unavailable fields stay null. Version and thread settings are measured, not inferred.
- tab:synthesis: use the task-specific mean ± SD above and completion/attainment counts; do not combine accuracy and text CE into a single unnormalized score.
- tab:contributions: Member 1 independently implemented SGD/DoG, fixed-gradient checks, quadratic diagnostics, shared CPU pipeline, tuning, formal-run orchestration and exports. Report integration/presentation work remains to be completed.

## Abstract/conclusion contribution

The frozen three-task CPU comparison separates formal quality across three seeds from validation search cost; the main and cost tables quantify DoG against a five-candidate SGD baseline. Interpret any task-specific advantage with the observed SD and the study's limited model/data scope.
