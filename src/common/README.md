# Shared implementation interfaces to complete

The current delivery defines an experiment scaffold, not a completed training library.
Member-specific code can be developed and checked on deterministic gradients in parallel.
Before formal neural-network runs, agree and implement the following shared interfaces:

- `data.py`: `load_task(config, seed)` returns train/validation/test data and the saved split hash. Member 3 maintains the specification; all members reuse it.
- `models.py`: `make_model(task_config, seed)` constructs the same MLP/CNN/GRU and initial parameter hash. Member 3 maintains it.
- `train.py`: `train_one(model, optimizer, data, config)` logs the common schema. Member 1 maintains training order/budgets, Member 4 metrics, Member 2 optimizer train/eval transitions.
- No member should make an undocumented private change to splits, models or timing definitions.

These three Python files are planned interfaces, not yet implemented files. The per-member
entry modules are actual files with explicit NotImplementedError until their implementation phase.
