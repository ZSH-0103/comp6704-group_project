# SGD validation tuning and matched-budget DoG comparison

Evidence batch: `20261029T135207Z_dab064d3`; manifest under `notes/tuning/<batch>/`.
All 15 SGD trials and 3 fixed-default DoG runs completed, with no failures. These are tune-phase runs on seed 11; formal main seeds 1, 2, 3 remain unused.

## Search and fairness

The search was declared in `configs/member1_tuning.json` before execution. Each image task tested SGD learning rates {0.003, 0.01, 0.03, 0.1, 0.3}; text tested {0.01, 0.03, 0.1, 0.3, 1}. This respects the shared limit of five candidates per optimizer/task. The winner minimizes final-budget polynomial-averaged validation CE; intermediate minima and raw validation scores do not determine selection. Ties choose the smaller learning rate.

Each run used the approved shared data/model configuration, one CPU thread, float32, split seed 6704 and initialization/order seed 11. Budgets were Fashion 2,000, CIFAR 3,000 and text 5,000 updates with full validation every 100 updates. The authorized candidate budgets are recorded as tuning overrides, leaving the canonical shared main configuration unfrozen. Both methods use polynomial averaging gamma=8, raw curves retained, no momentum/weight decay/scheduler. DoG keeps reps_rel=1e-6 and eps=1e-8 without tuning.

Every candidate within a task has the same initialization, sample-order, model and approved split hashes. The post-run audit checks these, complete budgets, finite losses, exact selection and disabled test evaluation. All runs retain config/environment/metrics/summary. Source snapshots, the original tuning declaration, and the selection audit are in the batch directory.

## Selected results

| Task | Budget | Selected SGD lr | SGD final validation CE | DoG final validation CE | SGD / DoG validation accuracy |
| --- | ---: | ---: | ---: | ---: | --- |
| Fashion-MNIST | 2,000 | 0.3 | 0.32519 | 0.32468 | 87.35% / 88.15% |
| CIFAR-10 | 3,000 | 0.1 | 1.24880 | 1.20547 | 58.55% / 58.55% |
| Tiny Shakespeare | 5,000 | 1.0 | 1.77732 | 1.71823 | not applicable |

CE is nats/image or nats/character. These are single-seed validation scores, not test scores or means across main seeds. Selected SGD curves reuse their winning search runs; they are not extra independent repetitions. Reusing the selection seed favors SGD on this validation comparison. No significance claim or universal optimizer ranking is supported.

Fashion: tuning changes the earlier picture substantially. SGD and DoG end with very similar validation CE; DoG is lower by only about 0.00051 on this seed. Both still improve by about 3.7–3.9% over the last 20% of updates. The best SGD candidate is at the upper search boundary, so 0.3 is the best tested rate, not a demonstrated global optimum.

CIFAR: lr=0.1 is an interior winner; lr=0.3 is worse at the final budget. Selected SGD achieves its observed validation minimum 1.22238 at update 2,400, then ends at 1.24880 while training CE continues falling. This is consistent with emerging overfitting but is one validation trajectory, not proof of the cause. DoG ends at 1.20547 and keeps improving by 3.44% over the last 20%. Their accuracy is equal on this split even though CE differs. We retain final-budget scoring; we do not replace the selected SGD final score with its earlier minimum.

Text: tuned SGD reaches 1.77732, far better than the weak pilot baseline. DoG reaches 1.71823. Final-20%-window improvements are 1.43% for SGD and 0.80% for DoG. The 5,000-update budget captures much of the visible improvement, while further training may still help. SGD lr=1 is the upper boundary winner.

## Recommended freeze proposal

Keep candidate budgets 2,000 / 3,000 / 5,000 and evaluation interval 100 for all optimizers within each respective task, pending the other members' pilot costs. These provide useful training within a small CPU budget, not a guarantee of full convergence. On CIFAR especially, extending training indefinitely is not automatically beneficial for validation quality.

Recommend accepting the best-tested SGD settings under the agreed five-candidate resource cap, transparently disclosing the Fashion/text boundary winners. If the group wants a stronger learning-rate-optimality claim, agree a search extension first and record all additional trials; do not silently exceed the cap or change the claim to “optimally tuned SGD.” Any formal budget change requires checking/reselecting learning rates at the new final-budget criterion before main runs.

DoG retains its current defaults and averaging policy. Member 1's configuration is now `TUNED_AWAITING_GROUP_FREEZE`; shared and distributed settings remain unfrozen. Proposed selected settings are also saved in `notes/selected_hyperparameters_proposal.json`.

## What to confirm with teammates

1. Accept the common budgets and final-budget evaluation rule, after checking other pairs' cost and convergence. Retain existing shared data/models and approved manifests unless a coordinated revision is justified.
2. Accept the five-candidate search result with boundary disclosure, or explicitly authorize additional candidates before freezing. My recommendation is to keep the current resource cap for this course project and state the limitation.
3. Agree task-specific common target validation CE thresholds if time-to-target will be reported. Plausible proposals based on the observed curves are Fashion 0.5, CIFAR 1.5 and text 2.0; the other pairs should review them. These are unadopted proposals, not used in current runs. Alternatively agree to omit time-to-target and update the reporting contract consistently. Do not leave an undefined metric in the final report.
4. Members 2–4 must complete their evaluation-policy declarations; Member 2 verifies Schedule-Free parameter transitions. SGD/DoG keep gamma=8 averaged final weights with raw validation as supplementary evidence.
5. Confirm one source/config/manifest release and a handoff deadline for other members' CSVs, figures, settings, machine details, conclusions/limitations, contributions and BibTeX. Member 1 integrates these rows without rerunning their algorithms.

After confirmation, Member 1 still needs main batch execution and final exports, 18 independent main runs, test evaluation after freezing, aggregate statistics, Sections 2.2/4.1 and the shared report integration, followed by presentation and delivery checks. No tune runs belong in main_results.csv.

## Cost and reproduction

SGD's 15 measured search runs cost approximately 754 seconds of training/evaluation on this Mac, separate from setup and DoG comparison runs. The cost is preserved per trial in `notes/tuning_log.csv`; the selected trajectory's runtime alone is not the cost of tuning. Different Macs should not be ranked by absolute time.

```sh
.venv/bin/python scripts/run_member.py --member 1 --phase tune
.venv/bin/python -m src.members.member1_tuning_plots
```

The first command executes a new complete search batch; the second redraws the latest saved logs. Figures are `figures/tuned_sgd_vs_dog.{png,pdf}`, `sgd_validation_search.{png,pdf}` and `tuned_loss_vs_time.{png,pdf}`. `notes/tuned_comparison.csv` contains final metrics, costs, late progress and boundary flags. The full checklist is `docs/MEMBER1_FREEZE_CHECKLIST.md`.

## 可发送给组员的确认信息

SGD 的验证集调参已完成：每任务固定 5 个候选，seed=11，未评估测试集。按 Fashion/CIFAR/文本 2,000/3,000/5,000 步的最终平均参数验证 CE 选择，最佳学习率分别是 0.3/0.1/1.0。SGD 与默认 DoG 的最终验证 CE 分别为 0.3252/0.3247、1.2488/1.2055、1.7773/1.7182。这些还不是正式多种子结果。

建议先确认：①其他三对优化器能否采用相同任务预算、每 100 步评估；②Fashion 和文本的最佳 SGD 值在搜索上界，是否接受“5 次候选内最佳”的资源限制并说明局限（我建议接受），还是统一同意扩展搜索；③是否采用共同 validation CE 达标阈值 0.5/1.5/2.0，或明确不报告 time-to-target；④Member 2/3/4 补齐评估策略，尤其 Schedule-Free 的参数切换；⑤统一代码/配置/划分版本和最终结果交接时间。确认后再冻结共享及个人设置，开始 seeds 1/2/3 的 main。CIFAR SGD 的验证损失后期略有回升，但仍按已同意的最终预算模型评分，不临时改为 best checkpoint。
