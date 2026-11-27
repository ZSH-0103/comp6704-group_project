# COMP6704 Member 1 — SGD and DoG

Member 1's independent SGD and ordinary global DoG implementation, shared CPU pipeline,
frozen settings, and completed experiment evidence. Tasks are Fashion-MNIST with an MLP,
CIFAR-10 with a small CNN, and Tiny Shakespeare with a character GRU.

All 18 formal runs completed. The quadratic study contains 30 comparison runs and 45
SGD tuning trials, including five recorded divergent tuning trials.

## Upload to GitHub / 上传说明

解压 ZIP 后上传本文件所在文件夹中的内容，保留 `src/`、`configs/`、`data/`、
`results/`、`scripts/`、`docs/` 和 `report/` 的相对位置。
ZIP 用于传输。GitHub 仓库里应能直接看到 Python 文件和 README。

如果上传到已有的小组仓库，可将整个文件夹放在 `member1/` 下，并从该目录运行命令，
这样便于保留你这一部分的完整配置、公共模块和证据。

If using GitHub's file uploader, open the destination repository and use **Add file →
Upload files**, then drag in the extracted contents. Files can be uploaded in batches while
keeping their directory paths. Commit the upload after reviewing the listed files.

For a new empty GitHub repository, another option is to run these commands from this folder.
Replace `YOUR_REPOSITORY_URL` with your actual repository address before the push.

```sh
git init
git add .
git commit -m "Add Member 1 SGD and DoG implementation and experiment evidence"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

The commands above assume a new empty remote repository. For a populated group repository,
copy this folder into a checkout of that repository and follow the group's branch workflow.

## Install

The experiment environment recorded Python 3.13.1 on macOS ARM64. The exact recorded
packages are in `requirements-lock.txt`, including PyTorch 2.15.0, torchvision 0.29.2,
NumPy 2.5.4, matplotlib 3.11.2 and dog-optimizer 1.0.3.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/verify_bundle.py
```

These commands assume macOS/Linux. On Windows, activate with `.venv\Scripts\activate`.
The lock records the original machine's packages. Availability of those exact versions on
another platform depends on its package index and Python installation. `requirements.txt`
provides the direct dependencies for a separately recorded compatible environment.
Installing from a new package index was outside this packaging validation.

`dog-optimizer` supplies the official reference for correctness checks. Training calls the
independent optimizers in `src/members/member1_optimizers.py`.

## Verify the implementation

Run from this folder after activating the environment. All three checks use generated fixtures.

```sh
python -m unittest src.members.member1_checks -v
python -m src.common.checks
python -m src.members.member1_quadratic_checks
```

The checks cover 11 optimizer cases, 8 shared-pipeline cases and 4 quadratic cases, including
fixed gradients, reference comparisons, checkpoint continuation, split isolation and target
logging. To save a new optimizer-check record, use `python scripts/run_member.py --member 1 --phase check`.

## Rebuild results from the supplied logs

The saved formal and quadratic evidence is sufficient to rebuild tables and figures offline.

```sh
python scripts/run_member.py --member 1 --phase export
python -m src.members.member1_quadratic_plots
python scripts/build_member1_report_assets.py
```

The export audits all 18 formal runs, their frozen settings, matched pairs, raw summaries
and target crossings. The compact report figures are regenerated in `report/figures/` and
LaTeX tables in `report/generated/`. PDF figure generation uses matplotlib.

`BUNDLE_MANIFEST.json` checks the original release files. Run its verification before
regenerating outputs, because regenerating a PDF or a check record can change its checksum.

## Data and additional experiments

The approved split indices and source checksums are in `data/manifests/`. Prepare the datasets
with an explicit download before running smoke, tuning or new neural training.

```sh
python -m src.common.data --download --cifar-mirror
python scripts/run_member.py --member 1 --phase smoke
```

The downloader checks the sources and creates `data/raw/`, which is ignored by Git.
To execute a new quadratic batch using the frozen diagnostic configuration, run
`python scripts/run_member.py --member 1 --phase diagnostic`. This records a new batch and
updates the quadratic pointer and exports. The validation-only tuning command is
`python scripts/run_member.py --member 1 --phase tune`.

The formal entry point is `python scripts/run_member.py --member 1 --phase main`.
This release includes the completed formal batch, so that command verifies the frozen
source hashes and reports its completion. A new formal batch needs a separate experiment
copy and an explicit record of the new batch. Preserve the original manifests and report
which batch supplies each result. `docs/FROZEN_SETTINGS_MEMBER1.md` explains the safeguards.

## Frozen formal configuration

| Task | Train / validation | Batch | Updates | Target val. CE | SGD rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Fashion-MNIST | 10,000 / 2,000 images | 64 | 2,000 | 0.5 | 0.3 |
| CIFAR-10 | 10,000 / 2,000 images | 64 | 3,000 | 1.5 | 0.1 |
| Tiny Shakespeare | 13,942 / 1,742 windows | 16 | 5,000 | 2.0 | 1.0 |

- Formal seeds 1, 2 and 3, tuning seed 11, split seed 6704.
- CPU float32 and one computation thread. Validation is evaluated every 100 updates.
- Both methods use polynomial averaging with gamma 8 for their final evaluations.
- DoG uses relative radius `1e-6`, epsilon `1e-8` and multiplier 1, with zero search trials.
- SGD uses a constant rate selected from five candidates per task by final validation CE.
- Final image tests use the official 10,000 examples. Text testing uses the full configured
  chronological test split. Test data are evaluated at the frozen final checkpoint.
- `elapsed_s` includes training and scheduled evaluation. Data preparation, tuning and final
  test evaluation have separate timing boundaries.

Authoritative settings are `configs/shared_protocol.json`, `configs/member1.json` and
`configs/member1_quadratic.json`. The quadratic diagnostic uses 20 dimensions, 500 updates,
exact gradients and float64 raw iterates, varying condition number and objective scale.
Its nine-candidate curvature-aware SGD search is separate from the neural search.

## Formal results

Mean ± sample SD over three formal seeds. Image quality is test accuracy in percent,
and text quality is test CE in nats per character.

| Task | SGD test quality | DoG test quality |
| --- | ---: | ---: |
| Fashion-MNIST | 84.15 ± 2.67 | 85.46 ± 0.34 |
| CIFAR-10 | 58.45 ± 0.65 | 57.93 ± 0.78 |
| Tiny Shakespeare | 1.8852 ± 0.0112 | 1.8989 ± 0.0069 |

Full precision, validation/training losses, timings and completion counts are in
`results/member1/tables/main_results.csv` and `cost_stability.csv`.
Interpretations and limitations are in `results/member1/notes/findings.md` and
`quadratic_findings.md`. Three seeds and a limited SGD grid constrain the conclusions.

## Layout and provenance

- `src/members/member1*.py` contains SGD/DoG, checks, smoke, pilot, tuning, formal and export code.
- `src/common/` contains the shared data, model and training pipeline.
- `results/member1/raw/` contains per-run configurations, environments, metric traces and summaries.
- `results/member1/notes/main_latest.json`, `tuning_latest.json` and `quadratic_latest.json`
  identify the batches supporting the results. Batch folders contain source snapshots.
- `results/member1/tables/` and `figures/` contain aggregated evidence and plots.
- `report/` contains Member 1 generated tables and figures. The integrated team report is
  maintained separately in Overleaf.

Earlier planning/specification notes are retained as dated development evidence. Statements
about unfinished work in those historical notes are superseded by this README, the frozen
configurations and completed batch manifests. Recorded local paths in historical logs identify
the original run environment. Current commands derive their paths from the extracted folder.

Original datasets, virtual environments and model checkpoints are excluded. All files needed
for the saved-results export are included. The scope of this package is Member 1's work and
the shared modules it uses. Other members supply their own implementations and evidence.

To rebuild this release into a new folder, run
`python scripts/package_member1.py --output dist/member1_sgd_dog_github`.
The packager uses explicit file selection and records SHA-256 checksums in the bundle.

Paper and reference implementation details are recorded in `results/member1/notes/literature.md`
and the implementation-check records, with the paper at https://proceedings.mlr.press/v202/ivgi23a.html
and the reference code at https://github.com/formll/dog.
