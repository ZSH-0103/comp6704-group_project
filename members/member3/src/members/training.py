"""Instrument Member 4's loop without changing data, batch or loss semantics."""
import hashlib
import time
import torch
from torch.nn import functional as F
from src.common.train import _batch, evaluate, optimizer_state_bytes


def train_one(model, optimizer, data, task, seed, max_updates, batch_size,
              eval_interval, target_val_ce, eval_examples=1024,
              evaluate_test=True, eval_batch_size=128):
    generator = torch.Generator().manual_seed(int(seed) + 100_000)
    batch_digest = hashlib.sha256()
    metrics, target_step, target_time = [], None, None
    started = time.perf_counter()

    def record(step):
        train_ce, _ = evaluate(model, data.train, task, eval_batch_size, eval_examples)
        val_ce, val_acc = evaluate(model, data.validation, task, eval_batch_size, eval_examples)
        if not (torch.isfinite(torch.tensor(train_ce)) and torch.isfinite(torch.tensor(val_ce))):
            raise FloatingPointError(f'Nonfinite evaluation at {step}')
        elapsed = time.perf_counter() - started
        metrics.append(dict(step=step, elapsed_s=elapsed, train_ce=train_ce,
                            val_ce=val_ce, val_accuracy_pct=val_acc,
                            learning_rate=optimizer.param_groups[0]['lr']))
        return val_ce, elapsed

    val_ce, elapsed = record(0)
    if val_ce <= target_val_ce:
        target_step, target_time = 0, elapsed
    model.train()
    for step in range(1, max_updates + 1):
        indices = torch.randint(len(data.train), (batch_size,), generator=generator)
        batch_digest.update(indices.numpy().tobytes())
        x, y = _batch(data.train, indices)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if task == 'tiny_shakespeare':
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        else:
            loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at {step}')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f'Nonfinite gradient at {step}')
        optimizer.step()
        if step % eval_interval == 0 or step == max_updates:
            val_ce, elapsed = record(step)
            if target_step is None and val_ce <= target_val_ce:
                target_step, target_time = step, elapsed
    pre_test_time = time.perf_counter() - started
    test_ce, test_acc = (evaluate(model, data.test, task, eval_batch_size, eval_examples)
                         if evaluate_test else (None, None))
    final = metrics[-1]
    return metrics, dict(final_train_ce=final['train_ce'], final_val_ce=final['val_ce'],
                        test_accuracy_pct=test_acc, test_ce=test_ce,
                        elapsed_s=time.perf_counter()-started,
                        train_validation_time_s=pre_test_time,
                        target_val_ce=target_val_ce, target_step=target_step,
                        target_time_s=target_time, target_reached=target_step is not None,
                        optimizer_state_bytes=optimizer_state_bytes(optimizer),
                        batch_sequence_hash=batch_digest.hexdigest(), gradient_calls=max_updates,
                        parameter_updates=max_updates-int(type(optimizer).__name__=='ADOPT0'))
