# Experiment interfaces and output contract

## Current status

This is a partially implemented project. The plan phase is available for all members.
Member 1 has working check/smoke/tune/main/diagnostic/export phases, with measured evidence under results/member1. All 18 Member 1 formal runs and exports are complete. Shared data/model/training modules are implemented. Other members' phase handlers remain unfinished. Unsupported phases fail explicitly. Smoke and quadratic results
must not be described as the completed three-task main experiment.

## Execution phases

| Phase | Required behavior | Output |
| --- | --- | --- |
| check | Compare updates on fixed gradients or a known small problem; test save/load and SF train/eval where relevant | notes/implementation_check.md |
| smoke | A few CPU updates on each task; validate pipeline, shapes, finite losses, logging and paired initialization | raw/smoke_*; exclude from main aggregates |
| tune | Method-specific validation search on a separate tuning seed; save every trial and selection rule | notes/tuning_log.csv and frozen config |
| main | 3 tasks × 2 optimizers × 3 seeds; fixed shared budget; test once after settings are frozen | 18 main run directories |
| diagnostic | Member-specific controlled problem or ablation with separate config IDs | raw/diagnostic_* |
| export | Aggregate measured logs, draw figures, populate result CSVs; reject empty or incomplete inputs | tables/*.csv, figures/*.pdf, notes/findings.md |

Never label smoke or tuning runs as main results. Do not overwrite an existing run directory.
A run ID includes member, phase, task, optimizer, seed and config hash.

## Per-run output files

Under `results/memberN/raw/<run_id>/`:

- `config.json`: complete resolved configuration, phase, seed, dataset/split/model/initialization hashes, optimizer implementation version, schedule and budget.
- `environment.json`: Python, dependency versions, CPU model, macOS, RAM, thread settings, git commit when available.
- `metrics.jsonl`: one record per evaluation; `step`, `elapsed_s`, `train_ce`, `val_ce`, `val_accuracy_pct` (null for text), `learning_rate` (if defined). Training CE must be evaluated consistently, not an unlabelled moving minibatch average.
- `summary.json`: `run_id`, `phase`, `task`, `optimizer`, `seed`, `config_hash`, `status`, `final_train_ce`, `final_val_ce`, `test_accuracy_pct` (images only), `test_ce`, `elapsed_s`, `target_val_ce`, `target_step`, `target_time_s`, `target_reached`, `optimizer_state_bytes`, `error_message`.

Status is `completed`, `diverged`, `timeout`, or `error`. Missing numeric data are null, never zero.
Metrics and summary must be written even when a run fails, as far as possible. JSON must not contain NaN or Infinity.
Time includes training and initial/scheduled evaluation, excludes download, dataset preparation and final test evaluation. Final test cost is separately recorded as test_elapsed_s.
For SF-SGD, evaluate and resume at the intended parameter points. Test is not read during tuning.

## Aggregate CSV files and report mapping

| Output in results/memberN/tables | Columns | Report destination |
| --- | --- | --- |
| main_results.csv | task, optimizer, n_expected, n_completed, train_ce_mean, train_ce_sd, val_ce_mean, val_ce_sd, test_metric_name, test_metric_mean, test_metric_sd, elapsed_s_mean, elapsed_s_sd | `tab:mN-main` / M<N>-MAIN |
| diagnostic_results.csv | diagnostic_id, condition, optimizer, metric_name, metric_unit, n, mean, sd, config_hash | `tab:mN-diag` / M<N>-DIAG |
| cost_stability.csv | task, optimizer, n_expected, n_completed, n_target_reached, target_val_ce, target_steps_mean_reached, target_time_s_mean_reached, optimizer_state_bytes, tuning_trials, tuning_elapsed_s | Results paragraph and relevant figure; final overview evidence |

The MAIN CSV has exactly six task/optimizer rows when the primary matrix is complete. Aggregate only
main runs with the selected hyperparameters. Use sample SD (ddof=1); with fewer than two observations
report SD as missing. Report completion count and individual failures, not just successful-run means.
Conditional means of target time must be labelled as conditional on reaching the target; also report
n_target_reached. Do not compare conditional means as if unsuccessful runs disappeared.
Each diagnostic CSV uses long format so every factor level and optimizer has an auditable row.

`notes/tuning_log.csv` columns: trial_id, task, optimizer, seed, hyperparameters_json,
validation_ce, status, elapsed_s, selected. Use validation only.

`notes/findings.md` contains: hypothesis, table/figure evidence, failures, mechanism explanation,
limitations, a row for shared synthesis, and one short abstract/conclusion contribution.

## Figures

Produce `results/memberN/figures/memberN_overview.pdf` plus the plotting code.
Use a multi-panel layout: three-task loss versus updates, loss versus elapsed time, and a compact
mechanism panel. If this is too dense, create separate PDFs and update the report figure block.
State n, SD bands, units, evaluation point and conditions; do not imply that a confidence interval
is an SD band. Preserve machine-readable underlying data. Copy the selected report figure to
`../tex/figures/memberN_overview.pdf` locally, or `report/figures/` in the extracted repository.

## Frozen protocol

The shared task protocol and Member 1 configuration are now frozen; Members 2–4 still need their own evaluation declarations and member freezes. For future protocol revisions, freeze sample counts, split files, initialization, models, batch sizes,
update budgets, evaluation intervals and target thresholds after a CPU pilot. Method-specific tuning
must be documented. Mark both shared and member JSON `FROZEN` only when the contents are actually
complete. Coordinate shared changes once; implement independent optimizer modules and diagnostics in parallel.
