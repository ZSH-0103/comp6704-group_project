# Shared CPU pipeline

Implemented for all three common tasks. Member 1's smoke runner exercises SGD and DoG;
other members can reuse the functions below. Shared task settings are FROZEN as
shared-v1.1-frozen. Members 2–4 must still declare evaluation policies and freeze their
individual hyperparameters. See configs/shared_protocol.json and the Chinese handoff.

## Commands

From the repository root:

```sh
# Network needed once; raw downloads are excluded from release ZIPs.
.venv/bin/python -m src.common.data --download
# Optional pinned CIFAR mirror if the original server fails; same official MD5.
.venv/bin/python -m src.common.data --download --cifar-mirror
# Offline fixture checks, not experimental results.
.venv/bin/python -m src.common.checks
# Portable wiring example (PyTorch SGD, not a personal independent implementation).
.venv/bin/python scripts/run_shared.py --member 1 --task fashion_mnist --phase smoke
```

The shared protocol contains smoke/pilot overrides and frozen formal task settings.
Member-specific evaluation policies and hyperparameters must be declared separately.

## Data: load_task(config, seed)

Pass resolved settings including repo_root, task, split_seed, subset counts and text
window settings. Returns TaskData with train, validation, fixed train_eval, lazy test()
and metadata. The run seed does not change split membership; split_seed does.

- Images: official training data shuffled with split seed 6704 and divided into
  disjoint training/validation subsets. Float32 pixels in [0,1], no augmentation or
  extra normalization. Official test data are held out.
- Text: chronological 80%/10%/10% character segments, split before windowing. Vocabulary
  is fitted only on the training segment; token 0 represents unknown characters.
  Each 64-character input predicts the next 64 characters; stride 64. Window subsets
  are sampled deterministically within each segment.
- Manifests record actual indices, text boundaries/vocabulary, transformations and
  SHA-256 source checksums. Different configurations have different manifest names.
- Downloads are explicit. Fashion-MNIST uses its authors' HTTPS mirror and torchvision
  checksums; CIFAR-10 uses the official archive (optionally from the pinned MIT Open
  Learning mirror, with the official MD5 enforced). Tiny Shakespeare uses a pinned commit.
  Provenance is saved in data/manifests/sources.json.

## Models: make_model(task_config, seed)

Returns (model, metadata) including architecture and initialization hashes. CPU float32,
seeded PyTorch default initialization; no dropout or batch normalization.

| Task | Specification |
| --- | --- |
| Fashion-MNIST | Flatten, 784–128–64–10 MLP, ReLU between linear layers |
| CIFAR-10 | 3x3 convolutions, channels 3–16–32, padding 1; ReLU and 2x2 max-pool after each; flatten and linear 10 logits |
| Tiny Shakespeare | Embedding 32, one-layer GRU 64, vocabulary logits; fresh hidden state for each window |

CNN padding and preprocessing/window choices are frozen task settings. Construct a fresh
model with the same seed for each optimizer; never reuse trained weights within a pair.

## Training: train_one(model, optimizer, data, config)

Accepts a compatible PyTorch optimizer. Seeded permutations determine each epoch's
sample order, retaining short final batches. The smoke runner checks paired initial
weights, sample order, model and split hashes.

Each run writes config.json, environment.json, metrics.jsonl and summary.json into
the configured results/memberN/raw/ directory. Logs are flushed at evaluations. Nonfinite losses, gradients or
weights, errors and timeouts retain available evidence; missing numbers are null.
The summary records the last evaluated step so pre-failure metrics remain identifiable.

CE uses natural logarithms (nats/image or nats/character). Evaluation is weighted by
label count, including short batches. Training CE uses a fixed subset in evaluation
mode, not a moving minibatch average. Timing includes initial/scheduled evaluation
and training, excludes downloads/preparation, final test evaluation and final summary
writing. Final test time is separately recorded as test_elapsed_s.

Smoke uses polynomial averaging gamma=8 for both optimizers, updated after each
successful update. A separate averaged model prevents evaluation from overwriting
training weights. Raw validation metrics are also logged. Optimizer eval()/train()
hooks restore training parameters in a finally block. Actual Schedule-Free integration
still requires Member 2's verification; externally averaging its internal training
point should not be assumed to be its intended evaluation policy.

Test evaluation is disabled for smoke and rejected outside frozen main runs. Main
requires both shared and member FROZEN status. This trainer evaluates the final raw or
averaged model. Best-checkpoint selection, full training resume, schedulers, tuning
searches and main-result export are not implemented. Agree and add what is needed
before formal experiments. Tiny smoke timings are not reliable speed rankings.

## Evidence

Eight offline checks cover model shapes and initialization, text split/vocabulary
isolation, window boundaries, epoch coverage, weighted CE, averaging and restoration,
optimizer evaluation hooks, failure evidence and test-set/main guards.

Member 1 smoke/main evidence is in the original project, not distributed in the shared
package. For this package, see VALIDATION.md and requirements-lock.txt. Other members
are responsible for their own algorithms and main/export entry points.
