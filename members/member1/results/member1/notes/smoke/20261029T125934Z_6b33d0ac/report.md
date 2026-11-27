# Member 1 real-data CPU smoke tests

Status: **passed**. Batch: `20261029T125934Z_6b33d0ac`.

20 updates; batch size 16; train subset 256; validation subset 128; train CE on a fixed 64-example/window subset.
One CPU thread. Polynomial averaging gamma=8 for both; raw validation CE also logged. No test evaluation.

| Task | Optimizer | Status | Train CE | Validation CE | Validation accuracy (%) | Seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| fashion_mnist | sgd | completed | 2.28503 | 2.29733 | 13.28125 | 0.055 |
| fashion_mnist | dog | completed | 2.30029 | 2.31448 | 11.71875 | 0.038 |
| tiny_shakespeare | sgd | completed | 4.18249 | 4.18911 | — | 0.149 |
| tiny_shakespeare | dog | completed | 4.20408 | 4.20937 | — | 0.144 |

Pairing checks:

- fashion_mnist: {'initialization_hash': True, 'data_order_hash': True, 'split_hash': True, 'model_hash': True, 'all_updates_evaluated': True, 'weights_changed': True, 'test_not_evaluated': True}
- tiny_shakespeare: {'initialization_hash': True, 'data_order_hash': True, 'split_hash': True, 'model_hash': True, 'all_updates_evaluated': True, 'weights_changed': True, 'test_not_evaluated': True}

These are small pipeline checks, not tuned optimizer comparisons or main results. Image CE is nats/example; text CE is nats/character.
Train/validation subset sizes count images for image tasks and character windows for text. Download and preparation time is excluded.
Shared/member training protocols remain DRAFT. See src/common/README.md for design choices and remaining work.
