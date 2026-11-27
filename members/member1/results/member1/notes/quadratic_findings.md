# Quadratic experiment: results and interpretation

Batch: `20261029T085102Z_a0a3a0d1`. Protocol hash: `855f7779ef86`.
Evidence: `quadratic/20261029T085102Z_a0a3a0d1/manifest.json`, `results.md`, `audit.json`, and the per-run files referenced by the manifest.

## What was measured

We minimized f(x) = (a/2) xᵀHx, with known optimum x*=0 and f*=0. H has 20 eigenvalues geometrically spaced from 1 to κ and fixed orthogonal eigenvectors (rotation seed 6704). The condition sweep changes κ with a=1; the scale sweep changes a with κ=10. Initial vectors have norm 1; seeds 1, 2 and 3 vary their direction and are paired between methods and across conditions. Gradients are exact, so this compares full-gradient SGD updates with DoG, not stochastic training.

The preliminary CPU pilot used 50 updates at κ=10, a=1 and seed 11. Both optimizers ran successfully. The diagnostic configuration was then frozen with the already proposed 500-update budget, target relative gap 1e-6, float64 CPU execution, one thread, raw iterates and no averaging. The neural-network protocol remains DRAFT.

There are five distinct conditions: the reference (κ=10, a=1) is shared by the two sweeps. Each condition has six comparison runs: two methods × three starting points, for 30 comparisons total.

## SGD tuning and fairness

For each condition, nine constant learning rates were tested on the separate starting-point seed 11. Candidates were c/L for c in {0.05, 0.1, 0.25, 0.5, 1, 1.5, 1.8, 1.95, 2.1}, where L=aκ. Choose the completed run with minimum final relative gap; break ties by smaller learning rate. No seed-1/2/3 comparison result was used for selection.

This is a deliberately strong baseline: the grid uses the known largest eigenvalue of this synthetic problem. It is not a comparison with equal tuning compute or unknown curvature. SGD used 45 tuning runs; DoG used fixed implementation defaults (relative radius 1e-6 and epsilon 1e-8) without a search. All five c=2.1 trials crossed the predeclared divergence threshold; their logs and failure statuses are retained. They were not assigned zero losses or silently deleted.

There is no train/validation/test dataset for this synthetic problem. The separate tuning initialization serves as the selection instance; final comparisons use the three predeclared held-out initialization seeds. This procedure is distinct from the future neural-network validation protocol.

## Main observations

At scale 1, mean updates to relative gap <=1e-6 were:

| Condition number | Tuned SGD | DoG |
| --- | ---: | ---: |
| 1 | 1.0 (3/3 reached) | 125.0 (3/3 reached) |
| 10 | 27.3 (3/3 reached) | 256.7 (3/3 reached) |
| 100 | 179.3 (3/3 reached) | Not reached (0/3 within 500) |

At κ=100, DoG's final relative gap was 3.706e-4 ± 1.36e-4 (sample SD), versus 4.050e-13 ± 2.01e-13 for tuned SGD. All 30 comparison runs completed; failing to reach a target is different from numerical divergence.

At κ=1, the objective is spherical and SGD's selected η=1 cancels the gradient in one step in exact arithmetic. This is an especially favorable case for the tuned reference and should not be generalized to arbitrary tasks.

In the scale sweep, selected SGD rates were 1.8, 0.18 and 0.018 for a=0.1, 1 and 10. The corresponding target times in updates stayed at 27.3 on average. DoG required 256.7 updates on average at every scale without changing its settings. Across paired DoG runs, the largest absolute difference between relative-gap trajectories across scales was approximately 1.004e-8. This is approximate, not exact, scale invariance: fixed epsilon and floating-point arithmetic break the ideal cancellation.

## How to read the plots

- `figures/quadratic_conditioning.png` / `.pdf`: upper panels show relative gaps, lower panels show ηL. The upper dotted line is the target; the lower dotted line is the constant-step quadratic stability boundary ηL=2. Higher κ slows both methods. DoG starts with a very small step because of its fixed initial-radius default; its step subsequently grows but remains substantially smaller than the selected SGD step. This is consistent with slower progress along low-curvature directions. It does not establish that every possible DoG initialization would behave this way.
- `figures/quadratic_scale.png` / `.pdf`: upper curves almost coincide across scales, while lower-panel step sizes shift inversely with scale. Note that lower panels have different tick values. Multiplying f by a multiplies its gradient by a; approximately dividing the step size by a preserves movement in parameter space. DoG makes this adjustment internally, while SGD was retuned separately using the curvature-scaled grid.
- `figures/quadratic_tuning.png` / `.pdf`: the separate tuning sweep, with stars identifying selected settings. Overlapping scale curves are expected. Full numerical values, including divergent trials, are in `notes/quadratic_tuning_log.csv`.

Plots show arithmetic means with ±1 sample SD over completed runs; these are not confidence intervals. Thin objective curves show individual starts. The log display clips gaps and lower band edges at 1e-16. A band touching the display floor can arise from mean minus SD <=0, not from an observed exact-zero loss. The data retain original values; plots should not be used to rank extremely small values below the display floor.

## Cost, reproducibility and limits

`tables/diagnostic_results.csv` contains the long-format mechanism metrics. `tables/quadratic_cost_stability.csv` separately reports completion, attainment, conditional time/steps, persistent tensor-state bytes and tuning cost. Conditional means exclude unreached runs and are always accompanied by reached counts. DoG stores a 20-element float64 initial snapshot (160 tensor bytes); plain SGD has no persistent tensor buffer. This excludes model/gradient tensors, Python objects and temporary reductions.

Every run has configuration, environment, step logs and summary. Source snapshots and selected settings are saved with the batch. CPU/RAM fields were unavailable to the sandbox and are null; timings are very short and exclude setup/file writing, so they are not a reliable speed ranking. The original optimizer checks passed previously; four new checks passed for spectrum/gradient/pairing, scaling, closed-form SGD/target logging and retained divergence. A post-run audit checked 75 unique directories, finite JSON and paired configuration hashes.

Limitations: only three starting directions, a single dimension and spectrum family, exact gradients, a known zero optimum, raw iterates, fixed DoG initial radius and a strong curvature-aware SGD search. No conclusions about noisy gradients, iterate averaging or neural-network performance follow yet.

Suggested report sentence: On controlled exact-gradient quadratics, default global DoG preserved nearly identical relative convergence under objective scaling without retuning, but tuned SGD reached the target sooner; at κ=100, DoG did not reach the predeclared target within 500 updates for any of the three initializations.

## Commands

From the repository root:

```sh
# Re-run the four experiment checks.
.venv/bin/python -m src.members.member1_quadratic_checks

# Run a new batch: new run directories, fresh tuning, tables and plots.
.venv/bin/python scripts/run_member.py --member 1 --phase diagnostic

# Redraw from the latest saved evidence without training again.
.venv/bin/python -m src.members.member1_quadratic_plots
```

The environment snapshot is `notes/requirements-quadratic-lock.txt`. This findings note refers to the batch named above; inspect the new batch's generated results before updating its interpretation after any future configuration change. Do not place these values in the three-task main-results table.
