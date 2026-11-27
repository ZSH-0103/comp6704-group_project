"""Independent CPU SGD and global DoG implementations for Member 1.

Specification: results/member1/notes/implementation_spec.md.
DoG equations: Ivgi, Hinder & Carmon (ICML 2023), Sections 1 and 3.1.
The official implementation is used only by the correctness checks.
"""

import math

import torch
from torch.optim import Optimizer


def _positive(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


class _CPUOptimizer(Optimizer):
    """One fixed parameter group, dense real CPU float32/float64 tensors."""

    def add_param_group(self, param_group):
        if self.param_groups:
            raise ValueError("Only one fixed global parameter group is supported")
        params = list(param_group["params"])
        if not params:
            raise ValueError("At least one parameter is required")
        if len({id(p) for p in params}) != len(params):
            raise ValueError("Duplicate parameters are not supported")
        param_group = dict(param_group, params=params)
        super().add_param_group(param_group)

    def _validated_params(self):
        params = self.param_groups[0]["params"]
        for p in params:
            if p.device.type != "cpu" or p.dtype not in (torch.float32, torch.float64):
                raise ValueError("Expected CPU float32 or float64 parameters")
            if not torch.isfinite(p).all():
                raise FloatingPointError("Nonfinite parameter")
            if p.grad is not None:
                if p.grad.layout != torch.strided:
                    raise ValueError("Only dense gradients are supported")
                if not torch.isfinite(p.grad).all():
                    raise FloatingPointError("Nonfinite gradient")
        return params

    @staticmethod
    def _loss(closure):
        if closure is None:
            return None
        with torch.enable_grad():
            return closure()


class SGD(_CPUOptimizer):
    """Plain SGD without momentum, weight decay or an internal schedule."""

    def __init__(self, params, lr):
        super().__init__(params, dict(lr=_positive("lr", lr)))
        self._validated_params()

    @torch.no_grad()
    def step(self, closure=None):
        loss = self._loss(closure)
        params = self._validated_params()
        lr = _positive("lr", self.param_groups[0]["lr"])
        for p in params:
            if p.grad is not None:
                p.add_(p.grad, alpha=-lr)
        return loss


class DoG(_CPUOptimizer):
    """Ordinary DoG with one global scalar step size and multiplier fixed to 1.

    r_eps defaults to reps_rel * (1 + ||x_0||). An explicit absolute r_eps
    supports transparent toy checks. eps is added ONCE to the accumulator.
    x_0 is captured immediately before the first step with any gradient.
    Missing gradients act as zero; an all-None call is a complete no-op.
    Averaging belongs to the future training/evaluation layer, not this update.
    """

    def __init__(self, params, reps_rel=1e-6, eps=1e-8, r_eps=None):
        defaults = dict(
            reps_rel=_positive("reps_rel", reps_rel),
            eps=_positive("eps", eps),
            r_eps=None if r_eps is None else _positive("r_eps", r_eps),
            step_count=0,
            rbar=None,
            grad_sq_sum=None,
            step_size=None,
        )
        super().__init__(params, defaults)
        self._validated_params()

    @property
    def diagnostics(self):
        """JSON-compatible values; grad_sq_sum includes the one-time epsilon."""
        group = self.param_groups[0]
        return {k: group[k] for k in
                ("step_count", "r_eps", "rbar", "grad_sq_sum", "step_size")}

    @torch.no_grad()
    def step(self, closure=None):
        loss = self._loss(closure)
        params = self._validated_params()
        if all(p.grad is None for p in params):
            return loss
        group = self.param_groups[0]
        first = group["step_count"] == 0
        # Reduce in float64, even for float32 models, to avoid squaring in
        # low precision. Python scalars also survive state_dict restoration.
        grad_sq = sum(p.grad.detach().double().square().sum().item()
                      for p in params if p.grad is not None)
        if first:
            initial_norm = math.sqrt(sum(p.double().square().sum().item() for p in params))
            radius = group["r_eps"]
            if radius is None:
                radius = group["reps_rel"] * (1 + initial_norm)
            rbar = radius
            total = group["eps"] + grad_sq
        else:
            radius = group["r_eps"]
            distance = math.sqrt(sum(
                (p.double() - self.state[p]["initial"].double()).square().sum().item()
                for p in params
            ))
            rbar = max(group["rbar"], distance)
            total = group["grad_sq_sum"] + grad_sq
        if not all(math.isfinite(v) and v > 0 for v in (rbar, total)):
            raise FloatingPointError("Invalid DoG radius or gradient accumulator")
        step_size = rbar / math.sqrt(total)
        if not math.isfinite(step_size):
            raise FloatingPointError("Nonfinite DoG step size")
        if first:
            for p in params:
                self.state[p]["initial"] = p.detach().clone()
        group.update(r_eps=radius, rbar=rbar, grad_sq_sum=total,
                     step_size=step_size, step_count=group["step_count"] + 1)
        for p in params:
            if p.grad is not None:
                p.add_(p.grad, alpha=-step_size)
        return loss
