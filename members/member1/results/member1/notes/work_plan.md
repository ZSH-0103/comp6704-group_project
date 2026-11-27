# Member 1 work plan

Prepared 29 October 2026. This is a proposed execution plan, not completed research or frozen experiment settings. Paths below are relative to `comp6704-optimizer-study/`.

## 1. Scope and current state

Your individual study is **SGD versus ordinary, global DoG**: literature, independent implementation, correctness checks, three common tasks, conditioning/scale diagnostics, analysis, report Sections 2.2 and 4.1, and presentation.

Your shared responsibilities are title/member information, Introduction, training protocol, Section 4.5 synthesis, contributions, and bibliography integration. Collect other members' results; you do not own their optimizer implementations or experimental runs.

Verified locally:

- `src/members/member1.py` is a stub raising `NotImplementedError`.
- `src/common/README.md` describes data/model/training interfaces; their Python implementations are absent.
- `configs/member1.json` has unset hyperparameters and diagnostic levels. The shared configuration has unset splits, batch sizes, update budgets and thresholds.
- `results/member1/` had no measured results at inspection.
- The plan command works and reports 18 main runs and `DRAFT_REQUIRES_CPU_PILOT`.
- This folder is not currently a Git checkout. Coordinate the team's actual repository before adopting the suggested member branch.

Corrected locations: report in `report/`, prompts in `docs/prompts/`, assignment in `docs/COMP6704_Optimization_Group_Project_2026_Fall-1.pdf`, code/configuration/results directly under this project root. Ignore the old user's absolute paths.

## 2. Proposed milestones

The dates assume your group's Week 12 allocation is correct: presentation 17 November 2026 and submission 22 November at 23:59 Hong Kong time. The assignment confirms the week/date mapping and Sunday deadline rule, but does not identify your group's allocation. Verify Group ID and assigned week in Canvas.

| Target | Work | Completion criterion |
| --- | --- | --- |
| Oct 29–30 | Read DoG, write research questions, agree shared ownership and evaluation policy | Literature notes, implementation specification, list of shared decisions |
| Oct 31–Nov 2 | Implement SGD/DoG; check updates; build small quadratic diagnostic; contribute shared trainer | Correctness evidence and reproducible diagnostic pilot |
| Nov 3–4 | Run CPU smoke tests on all three common tasks; agree budgets | Data/model manifests available; paired runs work; complete shared settings frozen after team agreement |
| Nov 5–7 | Validation-only tuning; complete diagnostic design | Every trial logged; selected member settings frozen; diagnostic settings fixed |
| Nov 8–10 | Run 18 main experiments and final diagnostics | Complete run inventory including failures and per-run evidence |
| Nov 11–12 | Export tables/figures; write Sections 2.2 and 4.1 | Auditable CSVs, plots, findings and bibliography handoff |
| Nov 13–14 | Integrate shared report sections and team results | Protocol/synthesis/contributions complete; references consistent |
| Nov 15–16 | Prepare and rehearse presentation | Approximately 7–8 minutes, timed with the group; Q&A preparation |
| Nov 17 | Present, if Week 12 allocation confirmed | Record feedback and any corrections needed |
| Nov 18–21 | Correct, reproduce, compile and inspect; package with Member 4 | Report <=15 pages, final report+slides PDF <40 MB, reproducible repository |
| Nov 22 | Submit before 23:59, if Week 12 allocation confirmed | One agreed uploader verifies successful Canvas submission |

These are work targets, not runtime estimates. Measure pilot runtimes before committing to the full compute schedule. Finish the main scientific story before the presentation.

## 3. Start with a small, answerable research question

Primary question: How does DoG compare with properly tuned SGD in convergence, final quality, stability and total tuning cost across the three common tasks?

Mechanism question: How do objective scaling and condition number affect those differences on a controlled quadratic?

Write hypotheses before interpreting outcomes. A useful candidate is that DoG reduces dependence on choosing an external learning rate, while a scalar adaptive step size may still struggle with ill-conditioning. Treat this as a hypothesis to test, not a promised result.

Read the [DoG paper](https://proceedings.mlr.press/v202/ivgi23a.html), including the algorithm and theorem assumptions, and inspect the [official implementation](https://github.com/formll/dog). Its README recommends iterate averaging. Decide and document whether predictions use raw or averaged parameters before tuning; if averaging is used, document the rule, extra state and evaluation/restoration behavior. Do not silently compare different evaluation rules or conflate ordinary DoG, layerwise DoG and the theoretical variant.

Create `results/member1/notes/literature.md` covering motivation, exact update order, distance initialization/floor, gradient accumulation, numerical safeguards, averaging, assumptions, original experimental setting and the limits of this project's smaller reproduction. Record the exact reference-code version used for checking. Keep new bibliography entries in `notes/references_member1.bib` until integration.

## 4. Implementation and correctness

Use `src/members/member1.py` as the phase dispatcher. Suggested new helper files are `member1_optimizers.py`, `member1_diagnostics.py` and `member1_export.py` in the same directory. These names are proposals; they do not exist yet.

Independently implement both plain SGD and ordinary DoG. Use built-in SGD and official DoG only as references for verification, with matching settings. Check:

- One-step and multi-step updates against hand calculations or a small independent reference.
- DoG's distance from its saved initial parameters, accumulation order and single global norm across tensors.
- Zero gradients, tiny norms, finite outputs, and the chosen floor/numerical behavior.
- Save/load continuation matches uninterrupted updates, including averaging state if applicable.
- Equal initial weights and minibatch sequences within each paired experiment.

Record actual tolerances, observed errors, reference versions and pass/fail outcomes in `notes/implementation_check.md`. Passing the planner is not implementation validation.

## 5. Shared work to settle early

| With | Required agreement or handoff |
| --- | --- |
| Member 3 | Data loading, checksums, split manifests, vocabulary/text boundaries, exact network shapes and initialization hashes |
| Member 2 | Optimizer lifecycle hooks in the shared trainer, especially evaluation transitions for Schedule-Free SGD |
| Member 4 | Metric definitions, target-loss thresholds, timing boundaries, environment capture and state-memory accounting |
| All members | Batch sizes, update/evaluation budgets, tuning limits, checkpoint/evaluation selection rule and frozen configuration version |

Member 1 maintains training order and budgets. Propose taking the lead on `src/common/train.py`, with metric and lifecycle contributions from Members 4 and 2; confirm implementation ownership so the trainer does not remain unassigned. Member 3 maintains shared data/model specifications and implementation. Keep optimizer-specific code separate from the common loop.

Use the CPU pilot to estimate each task's run time. Compute your main-run estimate by summing the predicted times for both optimizers across all three tasks and multiplying by three seeds. Add tuning, diagnostics and recovery time separately. If the workload is too large, reduce shared budgets by team agreement before freezing, never privately after seeing test results.

## 6. Main experiments and tuning

The matrix is Fashion-MNIST/MLP, CIFAR-10 subset/CNN and Tiny Shakespeare/character GRU, each with SGD and DoG and seeds 1, 2, 3: **18 main runs**.

Use the proposed tuning seed 11 and validation CE for selection. Start with a small logarithmic SGD learning-rate search, such as five candidates after a pilot establishes a sensible range. Predefine any range-expansion rule and record every trial and its cost. Choose settings per task. For DoG, explicitly document fixed defaults and any searched choices; do not invent a conventional learning-rate search merely to fill the current JSON field. Record whether no such search is applicable.

Freeze the selection rule, training/evaluation protocol and complete shared configuration before formal runs, then freeze the selected personal hyperparameters. Use test data only for the agreed final evaluation. Formal comparisons must use the same task budget, initial parameters and data order within pairs. Equal learning rates across algorithms are not required.

Every run must produce `config.json`, `environment.json`, `metrics.jsonl` and `summary.json` under a unique `raw/<run_id>/`. Preserve diverged/failed runs. Aggregate only the selected main configurations, use sample SD, record actual successful n, and represent unavailable numbers as null rather than zero. Compare absolute timing within your own machine; do not rank different members' Macs by raw time.

## 7. Controlled quadratic experiment

A proposed starting design is `f(x) = (a/2) (x-x*)^T H (x-x*)`, with positive-definite H and known optimum. Hold dimension, eigenvectors, initialization, update budget and gradient-oracle definition fixed. Choose and document the spectrum construction, for example fixing the smallest eigenvalue to 1 and varying the largest to kappa.

- Conditioning sweep: kappa in {1, 10, 100}, with a=1.
- Scale sweep: a in {0.1, 1, 10}, with kappa=10.
- Run these as two one-factor sweeps, sharing the reference condition. Do not begin with a large interaction grid.
- Start with exact gradients for implementation debugging. If adding stochastic gradients, specify the noise distribution and whether noise scales with a. For a pure objective-scaling test, scale the whole oracle by a so the signal-to-noise ratio is preserved; use the same underlying noise draws within paired runs.
- Tune SGD reasonably per condition for the primary comparison and disclose the cost. An additional fixed-learning-rate sensitivity curve is useful, but label it separately from tuned SGD.
- Record absolute objective gap and normalized gap `gap_t/gap_0`, target-reaching counts, updates to target and step-size trajectories. Use a predeclared relative-gap threshold for cross-scale comparisons so scaling the objective does not alone change task difficulty.

Pilot first, then freeze levels, oracle, seeds, budget, tuning rules and target. Record distinct conditions and configuration hashes. Do not repeat identical deterministic runs and present them as independent stochastic evidence; use declared varying initializations/noise seeds for replicated uncertainty estimates. Negative results at high condition numbers are valid findings.

## 8. Deliverables and writing

Your results directory must include:

- `tables/main_results.csv`: six task/optimizer rows from the 18 main runs.
- `tables/diagnostic_results.csv`: long-format mechanism results.
- `tables/cost_stability.csv`: failures, target attainment, state size and tuning costs.
- `notes/tuning_log.csv`, `notes/implementation_check.md`, `notes/literature.md`, `notes/findings.md`, and your bibliography entries.
- `figures/member1_overview.pdf` plus plotting source and underlying data; copy the chosen figure to `report/figures/`.

For your own Sections 2.2 and 4.1, target approximately 2–2.5 pages combined, subject to the team's final page budget. Write the method and assumptions early. Write results after export: hypothesis, numerical evidence, mechanism interpretation, costs, failures and limitations. Report image accuracy in percent and text CE in nats/character.

Edit these files under `report/sections/`:

- `section2_methodology_member1.tex` and `section4_experiments_member1.tex`: your study.
- `frontmatter_member1.tex` and `section1_introduction_member1.tex`: identity and common motivation.
- `section3_protocol_member1.tex`: agreed protocol and collected optimizer settings.
- `section4_synthesis_member1.tex`: each member's evidence-backed conclusion, without averaging incompatible task metrics.
- `contributions_member1.tex`: actual implementations and contributions.

Ask each teammate to deliver by November 12: selected settings, result tables/figures, machine information, method/theory/cost summary, one supported conclusion with a limitation, contribution statement and BibTeX. Supply your own rows to Members 2 and 4 for their shared tables. Merge bibliography into `report/references.bib` without overwriting others' entries.

Prepare your presentation around motivation, the DoG mechanism, experimental fairness, the strongest main/diagnostic evidence and limitations. Rehearse explanations of parameter-free terminology, convex versus neural-network guarantees, SGD tuning fairness, averaging and the three-seed limitation.

## 9. Commands and completion gates

Run from the project root. Only `plan` has been verified in this planning task:

```sh
python3 scripts/run_member.py --member 1 --phase plan
```

After implementing the handlers:

```sh
python3 scripts/run_member.py --member 1 --phase check
python3 scripts/run_member.py --member 1 --phase smoke
python3 scripts/run_member.py --member 1 --phase tune
python3 scripts/run_member.py --member 1 --phase main
python3 scripts/run_member.py --member 1 --phase diagnostic
python3 scripts/run_member.py --member 1 --phase export
sh report/build_report.sh
python3 scripts/package_project.py --output dist/member1_project_snapshot.zip
```

Quadratic development can begin before the shared neural-network pipeline is ready. The final diagnostic command must use its own frozen specification. Export must reject missing/incomplete inputs, rather than manufacture a full-looking table.

You are done when your implementation is verified; all intended runs are accounted for; conclusions trace to raw evidence; your report files and shared handoffs are complete; plots and tables agree; and your timed presentation is ready. With Member 4, check the final <=15-page report, GitHub link, identities, true Group ID, and the combined report/slides PDF under 40 MB. A ZIP does not replace the Canvas PDF.

## First working session

1. Verify the assigned presentation week and group identity.
2. Agree on shared implementation ownership and an early CPU-pilot target.
3. Start `literature.md` and specify exactly which DoG update and evaluation rule will be implemented.
4. Implement a tiny fixed-gradient SGD/DoG check before attempting the neural-network tasks.

Do not wait for the full shared pipeline to start your independent method work.
