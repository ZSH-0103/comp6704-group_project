# SGD / DoG 500-update CPU pilots

Batch: `20261029T134226Z_6bdb672b`. Shared protocol: `shared-v1.0`.
All six runs completed. Raw logs/configuration/environment/summary remain in the run directories referenced by `pilot/<batch>/manifest.json`; source snapshots, pairing checks, audit and CSV summary are saved beside that manifest.

## Exact shared settings used

- CPU float32, one thread, deterministic algorithms, seed 11, split seed 6704.
- 500 updates; evaluations at 0, 100, 200, 300, 400 and 500.
- Each image task: 10,000 training / 2,000 validation; batch 64; fixed training-CE subset 512; full validation evaluation.
- Text: all 13,942 training / 1,742 validation windows; batch 16; length/stride 64; fixed training-CE subset 256; full validation evaluation.
- Approved split hashes checked; matching initial model/split/architecture/sample-order hashes within each pair; initial losses match exactly.
- Polynomial averaging gamma=8 for both algorithms; raw validation curves additionally recorded. No test evaluation.
- Independent member implementations: SGD lr=0.01 (fixed pilot baseline, not tuned); DoG reps_rel=1e-6 and eps=1e-8. No momentum, weight decay or external schedule.

## Observed results

| Task | Optimizer | Final validation CE | Validation accuracy | Last 100 updates: validation CE reduction | Measured seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| Fashion-MNIST | SGD | 1.27501 | 59.65% | 15.39% | 2.11 |
| Fashion-MNIST | DoG | 0.45858 | 83.50% | 8.07% | 2.22 |
| CIFAR-10 | SGD | 2.14094 | 26.15% | 3.40% | 17.69 |
| CIFAR-10 | DoG | 2.29107 | 10.20% | 0.37% | 18.64 |
| Tiny Shakespeare | SGD | 3.55346 | not applicable | 3.24% | 5.65 |
| Tiny Shakespeare | DoG | 2.00964 | not applicable | 2.85% | 5.77 |

These are selected averaged-model metrics. Image CE is nats/image; text CE is nats/character. The late reduction is `(CE_400 - CE_500) / CE_400`, not a projection of future improvement. Curves contain only six measurements, with one seed and no uncertainty estimates.

Fashion-MNIST: both curves are still falling. SGD's pilot learning rate is small relative to DoG's observed dynamic step size, so lower DoG loss does not establish superiority over tuned SGD. DoG's validation improvement also has not stopped.

CIFAR-10: neither method is near a demonstrated plateau. DoG remains close to chance at 500, but its raw step size grows from about 0.000108 at update 100 to 0.01999 at update 500. Its tiny default initial movement and subsequent adaptation are consistent with delayed progress. This is an observation, not proof of the cause or evidence that more steps alone will solve it. Keep the agreed preprocessing/model; investigate tuning and learning progress before attributing the issue to the dataset or changing shared settings.

Text: 500 updates cover only about 0.57 shuffled epochs of the training windows. Both methods improve even in the final interval. A longer budget is justified, while the quality of a tuned SGD baseline remains to be established.

## Recommended formal budgets (proposal, not a shared freeze)

| Task | Suggested updates | Evaluation interval | Approximate shuffled epochs | Linear runtime estimate per run |
| --- | ---: | ---: | ---: | ---: |
| Fashion-MNIST | 2,000 | 100 | 12.7 | 9 s |
| CIFAR-10 | 3,000 | 100 | 19.1 | 109 s |
| Tiny Shakespeare | 5,000 | 100 | 5.7 | 57 s |

These are practical candidate budgets, not empirically proven convergence points. They extend the pilot by 4x, 6x and 10x, respectively, accounting for continued improvement and different training-set coverage. Keep one budget per task for every optimizer and retain the shared final-budget checkpoint rule; do not give an individual method extra steps because it performs poorly.

The estimates scale each task's mean measured pilot time by proposed updates/500. They include initial and scheduled train/validation evaluation, exclude downloads/preparation, final test evaluation and file/setup overhead, and are measurements on this Mac only. With evaluation every 100 steps, the 18 Member 1 main runs project to roughly 17.5 minutes of training/evaluation; reserve around 20–40 minutes for planning, plus separate tuning, final tests, data preparation, reruns and any slower conditions. Other members must measure their own costs.

Before freezing: tune SGD fairly under the agreed five-candidate cap, inspect CIFAR learning progress at the proposed budget during tuning, and obtain the other optimizer pairs' pilot reports. If curves are still falling materially or failures occur, revise the common budget before main runs and record the change. One untuned seed cannot establish an optimal budget or a final optimizer ranking.

Suggested initial SGD search candidates, to agree and record before tuning: Fashion/CIFAR {0.003, 0.01, 0.03, 0.1, 0.3}; text {0.01, 0.03, 0.1, 0.3, 1}. These searches have not been run. No DoG setting was changed in these pilots.

Do not choose target validation-loss thresholds merely to make one optimizer look favorable. Confirm one task-specific threshold using the agreed pilot/tuning evidence before formal main runs, or explicitly omit time-to-target until it has a defensible definition. Current target fields remain null.

`pilot_budget_recommendations.json` records this proposal. The canonical shared configuration and distributed setup remain unchanged and unfrozen.

## Reproduction and plots

```sh
.venv/bin/python -m src.members.member1_pilot
.venv/bin/python -m src.members.member1_pilot_plots
```

The first command performs six new sequential pilots using the canonical shared configuration. The second redraws the latest saved batch without training. Figures are `results/member1/figures/pilot_loss_curves.{png,pdf}` and `pilot_loss_vs_time.{png,pdf}`. Per-run configuration records shared code hashes and the optimizer source hash. The CSV is `results/member1/notes/pilot_summary.csv`.
