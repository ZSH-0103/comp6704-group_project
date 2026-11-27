# SGD and DoG implementation specification

Status: update kernels and correctness checks implemented; training protocol remains DRAFT.

## Interface and scope

Code: `src/members/member1_optimizers.py`. Public classes subclass `torch.optim.Optimizer` and support `zero_grad`, `step(closure=None)`, `state_dict` and `load_state_dict`.

- `SGD(params, lr)`: explicitly supply a finite positive learning rate.
- `DoG(params, reps_rel=1e-6, eps=1e-8, r_eps=None)`: optional absolute radius for transparent toy tests; otherwise initialize relative to the initial parameter norm.
- Pass all model parameters in one fixed group. Additional groups and duplicate tensors are rejected so global DoG cannot silently become a blockwise method.
- Supported: dense real CPU float32/float64 parameters and dense gradients. Unsupported: sparse gradients, GPU/MPS, mixed precision, momentum, weight decay, projection and dynamic parameter groups.
- A closure is evaluated with autograd enabled; parameter updates run without gradient tracking. `step` returns the closure loss or None.
- No official optimizer is called by production code. Its package is required only for the reference check.

## Exact update order

Let x_t concatenate all parameter tensors immediately before update t, and let g_t concatenate their current gradients. Index the first actual gradient update by t=0.

SGD:

```text
x_(t+1) = x_t - lr * g_t
```

There is no internal schedule. A future trainer may explicitly change SGD's group `lr` under a documented protocol.

DoG (ordinary global variant, numerical epsilon added once):

```text
At the first step with any supplied gradient:
    save x_initial = x_0 (detached copies)
    r_epsilon = explicit r_eps, or reps_rel * (1 + ||x_0||_2)
    rbar_previous = r_epsilon
    G_previous = eps

At every actual update, including the first:
    distance = ||x_t - x_initial||_2
    rbar_t = max(rbar_previous, distance)
    G_t = G_previous + ||g_t||_2^2
    eta_t = rbar_t / sqrt(G_t)
    x_(t+1) = x_t - eta_t * g_t
    retain rbar_t and G_t for the next step
```

Distance is measured BEFORE the update and includes all registered tensors. The current gradient contributes BEFORE eta is computed. The numerator is a maximum distance from initialization, not cumulative path length. Epsilon is inside the square root and is not accumulated each iteration. There is no multiplicative learning-rate parameter: its value is fixed to 1.

The paper's ideal formula has no numerical epsilon. The positive epsilon here follows the practical reference and provides finite behavior at zero gradients; exact scale invariance should not be claimed when epsilon is significant.

## Numerical and state behavior

- Default relative radius and epsilon are implementation defaults, not validated task hyperparameters. The main experiment configurations remain unfrozen.
- Norm reductions and scalar state use float64/Python floats. Parameters remain in their original dtype. Consequently small float32 differences from the official implementation are expected and tested with a stated tolerance.
- `grad=None` means no parameter update and contributes zero to the gradient norm. Its parameter still contributes to displacement and initial norm. An all-None call does nothing and does not capture initialization or advance the counter. A supplied all-zero gradient does advance the counter, but leaves weights unchanged.
- Zero gradients leave the accumulated gradient magnitude unchanged; tiny gradients stay finite with positive epsilon. An extremely small update can round away in the parameter dtype.
- Reject nonfinite input parameters/gradients before updating any parameter. Reject nonfinite DoG radius/accumulator/step size. Unsupported inputs raise errors instead of being silently changed. A future trainer must also detect nonfinite losses or post-update weights and record divergence.
- DoG's per-parameter state holds the initial tensor snapshot. Group state holds the realized radius, maximum radius, accumulated squared gradients including epsilon, step counter and most recent step size. `diagnostics` exposes these scalars for later logging.
- Save both model weights and optimizer state. Reconstruct with the same parameter order/shapes/dtypes, then load both. Saving only optimizer state cannot restore current model weights. Do not externally change the training parameters after initialization.

## Cost and evaluation

Both methods take O(d) update work. Plain SGD has no persistent tensor state beyond model parameters/gradients. DoG adds one d-element initial snapshot plus constant scalar state. Float64 reductions create temporary tensors, so peak memory exceeds persistent state; measure those concepts separately.

This implementation updates raw training weights only. An averaging utility and evaluation/checkpoint policy remain unimplemented. Proposed future policy: polynomial averaging with gamma=8 available for both methods, with raw/averaged selection determined by the shared validation rule. Average after each successful update and never overwrite training weights permanently during evaluation. The theoretical radius-weighted average is a different object. No main run should start before this policy is agreed and implemented.

## Hand-worked test

Use x_0=(0,0), explicit r_epsilon=1 and eps=1e-8. For g_0=(3,4):

```text
G_0 = 25 + 1e-8
eta_0 = 1 / sqrt(25 + 1e-8)
x_1 = (-3*eta_0, -4*eta_0) ≈ (-0.6, -0.8)
```

For g_1=(0,5), the first displacement is slightly below 1, so the radius floor remains 1:

```text
G_1 = 50 + 1e-8
eta_1 = 1 / sqrt(50 + 1e-8)
x_2 = (-3*eta_0, -4*eta_0 - 5*eta_1) ≈ (-0.6, -1.50710678)
```

At the next update the measured displacement exceeds the initial floor. The multi-step scalar oracle checks that growth and maximum-radius history. Separately, SGD with x_0=(1,-2), lr=0.1 and gradients (3,4), (-1,2) must produce (0.7,-2.4), then (0.8,-2.6).

## Reproduce the checks

From the repository root (the existing `.venv` is already prepared):

```sh
.venv/bin/python scripts/run_member.py --member 1 --phase check
```

To create an equivalent check environment on a compatible Python/macOS installation:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r results/member1/notes/requirements-check-lock.txt
.venv/bin/python scripts/run_member.py --member 1 --phase check
```

The dependency snapshot covers this check environment only; torchvision, plotting dependencies and the full training environment have not been validated. No GPU, downloaded datasets or external network access is needed to run checks once dependencies are installed.

Checks cover hand arithmetic, an independent scalar trajectory, PyTorch SGD, official DoG in two dtypes, global tensor partitioning, save/load continuation, zero/tiny/missing gradients, invalid inputs and autograd closures. They write unique evidence under `results/member1/notes/checks/` and refresh the readable `implementation_check.md`. Numerical tolerances and observed errors are recorded. Test failure raises an exception and retains available evidence.

The `check` phase validates these kernels. The `diagnostic` phase runs the separate quadratic study documented in `quadratic_findings.md`. The `smoke` phase now runs all three real tasks through the shared pipeline, including separately tested polynomial averaging; see `smoke_results.md` and `src/common/README.md`. Neural-network `tune`, `main` and general `export` remain unavailable. Optimizer tests and smoke results must not populate the main-results tables.
