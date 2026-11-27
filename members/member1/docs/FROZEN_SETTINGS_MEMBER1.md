# Frozen shared task protocol and Member 1 settings

Protocol version: `shared-v1.1-frozen`. Authorized after the user confirmed group agreement on budgets, the five-candidate SGD search cap and common validation-loss thresholds. The configuration and code snapshots in each formal batch identify the exact executed version.

| Task | Updates | Eval interval | Batch | Target validation CE | Selected SGD lr |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fashion-MNIST | 2,000 | 100 | 64 | 0.5 | 0.3 |
| CIFAR-10 | 3,000 | 100 | 64 | 1.5 | 0.1 |
| Tiny Shakespeare | 5,000 | 100 | 16 | 2.0 | 1.0 |

DoG: reps_rel=1e-6, eps=1e-8, multiplier 1, no tuning. Both methods use polynomial averaging gamma=8 and final-budget averaged weights for the primary evaluation. Raw validation metrics are supplementary. No momentum, weight decay, dropout, BatchNorm or external scheduler. Fashion/text upper-bound SGD winners are accepted as best within the documented five-candidate search, without a global-optimality claim.

Formal seeds are 1/2/3; tuning seed 11 is excluded. Initial weights and batch order match within each seed/task pair. Training CE uses the fixed training subset; validation covers the complete configured split. Frozen data: image train/validation 10,000/2,000 per task; text 13,942/1,742 windows with chronological 80/10/10 splitting and length/stride 64. Final tests use all 10,000 official test images per task or all chronological text test windows. All source and approved split hashes are stored in each run.

The final test set is evaluated once per newly executed completed main run. Main-run recovery reuses an existing summary and refuses ambiguous interrupted evidence rather than repeating a possible test evaluation. Test results cannot change the frozen settings or select a new checkpoint.

`elapsed_s` includes initialization-time evaluation, training and scheduled train/validation evaluation, and excludes preparation and final test evaluation. `test_elapsed_s` records final test cost separately. Target attainment is the first scheduled evaluation at or below the frozen threshold; its resolution is 100 updates, and unreached step/time values remain null.

The freeze covers common data/models/budgets/metrics and Member 1's settings. Members 2–4 must still finalize their evaluation policies, tuned hyperparameters and personal FROZEN configurations. This does not claim their experiments are complete. `shared_team_setup_v1/` is the historical pilot handoff; use the canonical `configs/shared_protocol.json` for the new freeze.

Commands from the repository root:

```sh
.venv/bin/python scripts/run_member.py --member 1 --phase main
.venv/bin/python scripts/run_member.py --member 1 --phase export
```

Main records a manifest after each run and resumes only the same frozen batch. Re-running after batch completion returns without retraining or repeated tests. A deliberate new formal batch requires an explicit new experiment plan; do not delete the latest pointer to hide past results.

Export uses only the current main manifest, rejects duplicate/missing or mixed-phase runs, checks frozen settings and raw evidence, and includes completion/attainment counts. CSVs use sample SD (ddof=1). Failed runs stay in the inventory; their stale pre-failure measurements are excluded from completed-result means. Conditional target means must be interpreted together with reached counts.

Outputs: `tables/main_results.csv`, `tables/cost_stability.csv`, `tables/main_run_inventory.csv`, `figures/member1_overview.png/.pdf`, `notes/main_results.md`, and batch-specific audits/source snapshots under `notes/main/`. Existing quadratic results remain separate. The overview PDF is copied to `report/figures/` for the report.
