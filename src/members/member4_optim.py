"""Independent Adam and AdEMAMix implementations for Member 4."""
import math
import torch
from torch.optim import Optimizer


def _half_life_warmup(step, beta_end, beta_start, warmup):
    if not warmup or step >= warmup:
        return beta_end
    def half_life(beta):
        return math.log(0.5) / math.log(beta + 1e-8) - 1
    def inverse(value):
        return 0.5 ** (1 / (value + 1))
    weight = step / float(warmup)
    return inverse((1 - weight) * half_life(beta_start) + weight * half_life(beta_end))


class IndependentAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("Sparse gradients are unsupported")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                state["exp_avg"].mul_(b1).add_(parameter.grad, alpha=1 - b1)
                state["exp_avg_sq"].mul_(b2).addcmul_(parameter.grad, parameter.grad, value=1 - b2)
                m = state["exp_avg"] / (1 - b1 ** state["step"])
                v = state["exp_avg_sq"] / (1 - b2 ** state["step"])
                update = m / (v.sqrt() + group["eps"])
                update.add_(parameter, alpha=group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss

class AdEMAMix(Optimizer):
    """Paper update; the slow EMA is intentionally not bias corrected."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=2.0,
                 beta3_warmup=None, alpha_warmup=None, eps=1e-8, weight_decay=0.0,
                 slow_enabled=True):
        defaults = dict(lr=lr, betas=betas, alpha=alpha, beta3_warmup=beta3_warmup,
                        alpha_warmup=alpha_warmup, eps=eps, weight_decay=weight_decay,
                        slow_enabled=slow_enabled)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2, b3_final = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg_fast"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    if group["slow_enabled"]:
                        state["exp_avg_slow"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                alpha = min(group["alpha"] * step / group["alpha_warmup"], group["alpha"]) \
                    if group["alpha_warmup"] else group["alpha"]
                b3 = _half_life_warmup(step, b3_final, b1, group["beta3_warmup"])
                fast, sq = state["exp_avg_fast"], state["exp_avg_sq"]
                fast.mul_(b1).add_(parameter.grad, alpha=1 - b1)
                sq.mul_(b2).addcmul_(parameter.grad, parameter.grad, value=1 - b2)
                numerator = fast / (1 - b1 ** step)
                if group["slow_enabled"]:
                    slow = state["exp_avg_slow"]
                    slow.mul_(b3).add_(parameter.grad, alpha=1 - b3)
                    numerator = numerator + alpha * slow
                denominator = (sq / (1 - b2 ** step)).sqrt().add_(group["eps"])
                update = numerator / denominator
                update.add_(parameter, alpha=group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss

