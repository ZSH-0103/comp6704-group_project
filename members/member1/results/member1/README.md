# Member 1 outputs

The quadratic diagnostic, six real-data CPU smoke runs, and six 500-update CPU pilots have been completed.
Validation tuning and all 18 formal neural-network main experiments are complete; exports are generated. Literature notes,
optimizer checks, quadratic evidence and `smoke_results.md` are in `notes/`.

Read `notes/tuning_findings.md` for selected SGD settings, DoG comparisons and the
remaining group decisions. `notes/tuning_latest.json` locates all 18 tune-phase runs.
The 18 independent main runs on seeds 1/2/3 are recorded separately by notes/main_latest.json. Read notes/main_results.md and notes/findings.md; tables/main_results.csv and cost_stability.csv contain the formal aggregates.

See `notes/pilot_findings.md` for the agreed-protocol pilots and formal-budget
recommendations. Curves are `figures/pilot_loss_curves.png` and `.pdf`;
`notes/pilot_latest.json` locates the full batch evidence. Formal budgets are now frozen as documented in docs/FROZEN_SETTINGS_MEMBER1.md.

Start with `notes/literature.md`, then `notes/implementation_spec.md`.
`notes/implementation_check.md` summarizes the latest checks; timestamped evidence is
preserved in `notes/checks/`. For measured quadratic results, read
`notes/quadratic_findings.md` and `notes/quadratic_results.md`. Plots are in `figures/`;
the immutable experiment batch, source snapshots and run manifest are in `notes/quadratic/`.

- `raw/`: per-run configuration, metrics.jsonl, summary.json, environment.json.
- `tables/`: main_results.csv, diagnostic_results.csv, cost_stability.csv.
- `figures/`: member1_overview.pdf and its editable plotting source.
- `notes/`: literature.md, implementation_check.md, findings.md, tuning_log.csv.

Table labels: `tab:m1-main` and `tab:m1-diag`; shared tables also need your row.
Use the exact schemas and units in docs/EXPERIMENT_CONTRACT.md.
