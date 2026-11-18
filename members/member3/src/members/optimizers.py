"""Real-valued, dense, CPU optimizer specializations for Member 3.

ADOPT with beta1=0, no weight decay; follows the official scalar implementation
at commit 6468572b4b3688a2f056b796b071da540568520e. No redundant first-moment
buffer is allocated when beta1=0. The first gradient call initializes v only.
"""
import torch
from torch.optim import Optimizer


class RMSProp(Optimizer):
    def __init__(self, params, lr=1e-3, beta2=.999, eps=1e-8):
        if lr <= 0 or not 0 <= beta2 < 1 or eps <= 0:
            raise ValueError('Invalid hyperparameters')
        super().__init__(params, dict(lr=lr, beta2=beta2, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse or p.is_complex():
                    raise RuntimeError('Only real dense gradients are supported')
                state = self.state[p]
                if not state:
                    state.update(step=0, square_avg=torch.zeros_like(p))
                state['step'] += 1
                v = state['square_avg']
                v.mul_(group['beta2']).addcmul_(p.grad, p.grad, value=1-group['beta2'])
                p.addcdiv_(p.grad, v.sqrt().add_(group['eps']), value=-group['lr'])
        return loss


class ADOPT0(Optimizer):
    def __init__(self, params, lr=1e-3, beta2=.999, eps=1e-8, clip=True,
                 current_moment=False):
        if lr <= 0 or not 0 <= beta2 < 1 or eps <= 0:
            raise ValueError('Invalid hyperparameters')
        super().__init__(params, dict(lr=lr, beta2=beta2, eps=eps,
                                     clip=clip, current_moment=current_moment))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse or p.is_complex():
                    raise RuntimeError('Only real dense gradients are supported')
                state = self.state[p]
                if not state:
                    state.update(step=1, square_avg=p.grad.square().clone())
                    continue
                v = state['square_avg']
                if group['current_moment']:
                    v.mul_(group['beta2']).addcmul_(p.grad, p.grad, value=1-group['beta2'])
                normalized = p.grad / v.sqrt().clamp_min_(group['eps'])
                if group['clip']:
                    bound = state['step'] ** .25
                    normalized.clamp_(-bound, bound)
                p.add_(normalized, alpha=-group['lr'])
                if not group['current_moment']:
                    v.mul_(group['beta2']).addcmul_(p.grad, p.grad, value=1-group['beta2'])
                state['step'] += 1
        return loss


def make_optimizer(name, params, lr, beta2=.999, eps=1e-8):
    if name == 'rmsprop':
        return RMSProp(params, lr, beta2, eps)
    if name in ('adopt_no_momentum', 'adopt_no_clip', 'current_moment_matched_clip'):
        return ADOPT0(params, lr, beta2, eps, clip=name != 'adopt_no_clip',
                      current_moment=name == 'current_moment_matched_clip')
    raise ValueError(name)
