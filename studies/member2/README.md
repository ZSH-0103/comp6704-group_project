# Momentum and Schedule-Free SGD: final reproduction

This package reproduces the final Member 2 study using **PyTorch SGD with momentum** and **schedulefree.SGDScheduleFree 1.4.1**. These are library implementations, not independently implemented optimizers.

## Install and prepare data

Run commands from the repository root. The recorded environment was Python 3.13.13, PyTorch 2.14.1, torchvision 0.29.1, CPU float32, four threads. The complete environment snapshot is in `requirements-lock.txt` in this directory; per-run environments accompany the published records. Other hardware/library versions may change numerical results and timings.

```sh
python3.13 -m venv .venv-member2
.venv-member2/bin/python -m pip install -r studies/member2/requirements-lock.txt
.venv-member2/bin/python studies/member2/prepare_data.py
```

The preparation script downloads Fashion-MNIST from its official GitHub repository, CIFAR-10 from the Toronto dataset site, and Tiny Shakespeare from Karpathy's char-rnn repository. It verifies image archive checksums and records source URLs and SHA-256 digests. Training itself performs no downloads. Data goes under `studies/member2/data/raw/`, which is ignored by Git; do not commit datasets or virtual environments. The committed split manifests contain reproducible indices and a text source checksum; inconsistent data splits are rejected.

## Check and regenerate the reported results (no training)

```sh
.venv-member2/bin/python scripts/run_member.py --member 2 --phase check
.venv-member2/bin/python scripts/run_member.py --member 2 --phase export
```

`check` verifies hashes for all 45 published run records, final learning rates, paired initialization/split fingerprints, absence of test evaluation in diagnostics, and fixed-horizon behavior. It is a record/integration check, not a claim of independent optimizer correctness. `export` regenerates six-row main and nine-row diagnostic tables and two PDF plots under `studies/member2/generated/`. Means and sample SD use `ddof=1`. Published numerical observations are preserved in `published/`; timing curves plot loss against mean elapsed time at matched updates.

## Reproduce training

```sh
.venv-member2/bin/python scripts/run_member.py --member 2 --phase smoke
.venv-member2/bin/python scripts/run_member.py --member 2 --phase main
.venv-member2/bin/python scripts/run_member.py --member 2 --phase diagnostic
```

- `smoke`: six 5-update runs (two methods × three tasks), seed 11, no test evaluation. This is only a pipeline check.
- `main`: 18 runs, three tasks × two methods × seeds 1/2/3, **300 updates per run**. The frozen rates are used directly; there is no automatic tuning.
- `diagnostic`: 27 Fashion-MNIST runs, three methods × horizons 150/300/600 × seeds 1/2/3. The controls use cosine decay planned for 300 updates or cosine decay matched to the actual horizon. The fixed-300 control has zero LR after update 300. Diagnostics do not evaluate the test set.

Each invocation writes a new timestamped directory under `studies/member2/runs/`, with resolved configuration, per-run config/environment, JSONL curves, summary and model/optimizer/scheduler checkpoint. Published records are never overwritten. New checkpoints contain final training-mode parameters; Schedule-Free evaluation must use `optimizer.eval()` and restore `optimizer.train()` afterward (the runtime does this automatically).

To run a subset or export a new run directory:

```sh
.venv-member2/bin/python -m studies.member2.run main --task cifar10 --seed 1
.venv-member2/bin/python -m studies.member2.run export --source studies/member2/runs/REPLACE_WITH_RUN_DIRECTORY --output studies/member2/generated/new_run
```

## Final settings and evaluation

| Task | Momentum base LR | Schedule-Free base LR |
|---|---:|---:|
| Fashion-MNIST | 0.1 | 1.0 |
| CIFAR-10 | **0.03** | **0.3** |
| Tiny Shakespeare | 0.1 | 1.0 |

Both use momentum 0.9 and zero weight decay. Momentum uses cosine decay to zero; Schedule-Free uses 10-step warmup with the library defaults `r=0` and `weight_lr_power=2`. Main schedules end at 300 updates. The diagnostic reuses the Fashion rates without per-horizon retuning.

`configs/member2.json` holds the final method settings. `configs/member2_protocol.json` is a frozen **Member 2** snapshot, not a declaration that all pairs share one protocol. `runtime/` preserves the data/model/training modules used for these observations; it is separate from `src/common/` so changes made for another pair do not silently change our evaluation or Schedule-Free parameter handling.

Images use training subsets of 6,000 Fashion examples or 5,000 CIFAR examples and validation/test subsets of 1,000 each. Batch size is 64. Text uses a 65-character GRU, embedding 32, hidden width 64, sequence length 64 and batch size 32. Splits are chronological at character offsets 892312 and 1003851; vocabulary is derived from the full corpus. Training batches are sampled **with replacement**, with seed `100000 + run seed`. This is a fixed-update experiment, not full epochs.

Evaluation is every 50 updates (including step zero and the final step), with batch size 128 and at most the first 1024 examples. Thus image validation/test uses all 1000 selected images; text uses 1024 overlapping windows from each held-out prefix. Final test is evaluated once per main run, after rates are fixed. CE uses natural logs. Training time includes scheduled train/validation evaluation, but excludes data preparation and final test. No common target CE was prespecified: target fields are null, not zero.

## Provenance and reported results

`published/manifest.json` maps every included run to its original local experiment directory and hashes the unmodified config, environment, summary and curve files. The archive contains exactly:

- 12 Fashion/text main runs plus 27 diagnostics from `experiments_20261012T122713Z_fd94e6c4`;
- 6 final CIFAR main runs from `cifar_final_20261013T063657Z_7d5f5a73`.

Historical configuration metadata is preserved verbatim and can mention earlier directories not included in this package. Those references are provenance only, not dependencies of the reproduction runner. Earlier CIFAR configurations, pilots and exploratory search are excluded from the final main matrix. The selected rates were chosen on validation CE. This package reproduces frozen settings, not the full historical search, and makes no claim about total tuning cost. Original local logs remain intact.

Final CIFAR test accuracies (Momentum / Schedule-Free): **50.7667 ± 1.0970% / 51.5333 ± 0.9238%**, sample SD across three seeds. The 600-update horizon diagnostic has validation CE approximately **0.4565** for Schedule-Free versus **0.4075** for horizon-aware Momentum. These are small-budget observations, not evidence of universal superiority or statistical significance.

Published PDF figures and paper-ready CSV tables are under `published/figures/` and `published/tables/`. Checkpoints and datasets are not committed; rerunning produces new checkpoints. Numerical results should match closely in the recorded environment, while wall times depend on hardware and system load.
