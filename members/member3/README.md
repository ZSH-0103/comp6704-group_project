# Member 3 — RMSProp / momentum-free ADOPT

Complete, independently runnable CPU study for COMP6704. Run commands below from
`members/member3/`, not the repository root. This module uses its own frozen configurations
and preserves the experiment code that generated the included results.

## What is included

- Independent RMSProp and ADOPT with beta1=0 implementations.
- 15 implementation/protocol checks; 6 smoke runs; 18 validation-only tuning runs;
  18 main runs (3 tasks × 2 methods × 3 seeds); 240 controlled diagnostic trajectories.
- Frozen hyperparameters, split indices, initialization/batch fingerprints, per-run logs,
  aggregate CSVs, plotting/export scripts, and reference-code provenance and license.
- The original October 17 Member 3 report snapshot, editable section files, and figures.
  This is supporting evidence, not the latest integrated group/Overleaf report.

Main test means (three seeds):

| Task | RMSProp | ADOPT-0 |
|---|---:|---:|
| Fashion-MNIST accuracy (%) | 81.17 | 79.43 |
| CIFAR-10 accuracy (%) | 47.47 | 47.33 |
| Tiny Shakespeare CE (nats/character) | 2.028 | 2.086 |

Sample SD and all individual observations are included in the results. The synthetic
asymmetric-noise diagnostic favors previous-moment normalization at beta2=.9; the gap is
small at .999. These findings do not establish a universal optimizer ranking.

## Files for teammates

- [Measured tables](results/member3/tables/), [run logs](results/member3/raw/),
  [figures](results/member3/figures/), and [Chinese findings](results/member3/notes/findings.md).
- [Optimizer code](src/members/optimizers.py) and [training instrumentation](src/members/training.py).
- [Frozen protocol](configs/shared_protocol.json), [selected hyperparameters](configs/member3.json),
  and [split manifests](data/manifests/).
- [Report snapshot](report/member3_report.pdf), [editable sections](report/sections/),
  [integration notes](docs/MEMBER3_HANDOFF.md), and [presentation notes](docs/Member3_汇报提纲与答辩.md).

## Environment and checks without downloading data

Recorded environment: Python 3.12.14, PyTorch 2.14.1, torchvision 0.29.1, NumPy 2.5.4,
Apple M1 Pro, 16 GiB RAM, macOS 15.6.1, CPU only with four PyTorch threads.
Exact package versions are in `requirements-lock.txt`; bitwise equivalence is not promised
on other environments.

```sh
cd members/member3
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pytest -q
.venv/bin/python scripts/validate_deliverable.py
```

The validator checks source/config fingerprints, frozen-before-main timestamps, paired
initialization and sampling, recomputed means/SDs, and diagnostic completeness. It does
not run training or evaluate a test set. The installed official ADOPT reference emits
PyTorch deprecation warnings; the numerical comparisons still pass.

## Reproduce training in a separate checkout/copy

The included `results/member3/raw/` holds measured evidence. Before a new experiment,
move that directory to a separate backup location in your reproduction copy. The runner
skips existing run directories; a changed implementation fingerprint is rejected.
Never pool new machine timings with the included measurements.

```sh
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/run_member3.py smoke
.venv/bin/python scripts/run_member3.py tune
.venv/bin/python scripts/run_member3.py main
.venv/bin/python scripts/run_diagnostic.py
MPLCONFIGDIR=build/matplotlib .venv/bin/python scripts/export_results.py
.venv/bin/python scripts/write_report_tables.py
sh report/build_report.sh
.venv/bin/python scripts/validate_deliverable.py
.venv/bin/python scripts/package_member3.py
```

`prepare_data.py` downloads public datasets and verifies Fashion/CIFAR MD5 and the frozen
text SHA-256. CIFAR download uses up to eight range connections and validates the whole
archive. The final command creates a local ZIP under `deliverables/`; it does not publish
anything. LaTeX intermediates stay in `report/build/`. A local LaTeX/latexmk installation
is required only when rebuilding the report.

## Protocol boundaries

- Member 4's supplied data/model/train source is preserved unchanged, including a reference
  copy. It also matches the root shared source at base commit `f2d2f79`.
- Main budget is 300 gradient calls; ADOPT initializes its accumulator on the first call,
  so it makes 299 parameter updates. Both methods use beta2=.999, epsilon=1e-8 and zero
  weight decay. ADOPT's clipping and epsilon placement are explicitly documented.
- Each method/task receives three validation-only LR candidates and 100 tuning calls on
  seed 11; main seeds are 1, 2, 3. Selected rates are .0003/.001 on images and .003/.003 on text.
- Evaluation uses at most the first 1,024 examples/windows. Images have 1,000 validation/test
  examples; overlapping text windows cover only a small prefix of each held-out region.
- M3 validation targets 1.0/2.0/3.0 differ from M4's root configuration. Both timing
  conventions are saved; main tables exclude final testing. Extra finite-gradient checks
  are timed. Do not compare raw seconds or target times across members as a common benchmark.
- The root `scripts/run_member.py --member 3` remains the historical scaffold. Use this
  module's actual runner above. Root shared settings and other members' code are unchanged.

Datasets, virtual environments, model checkpoints, build caches, downloaded paper PDFs,
and the original private delivery archive are excluded from Git. The small `.npz` files
are synthetic diagnostic trajectories needed for validation, not raw training datasets.
