"""Paired, fixed-update CPU training and evaluation utilities."""
from contextlib import contextmanager
import time
import torch
from torch.nn import functional as F


def _batch(dataset, indices):
    items = [dataset[int(i)] for i in indices]
    xs = [x if torch.is_tensor(x) else torch.as_tensor(x) for x, _ in items]
    ys = [y if torch.is_tensor(y) else torch.as_tensor(y) for _, y in items]
    return torch.stack(xs), torch.stack(ys)


@contextmanager
def evaluation_mode(model, optimizer=None):
    """Evaluate at SF averaged parameters, then restore prior modes even on error."""
    was_training = model.training
    sf = optimizer is not None and callable(getattr(optimizer, 'eval', None)) and callable(getattr(optimizer, 'train', None))
    modes = [g.get('train_mode') for g in optimizer.param_groups] if sf else []
    if sf and (None in modes or len(set(modes)) != 1):
        raise ValueError('Expected consistent explicit optimizer train_mode across groups')
    try:
        if sf:
            optimizer.eval()
        model.eval()
        yield
    finally:
        if sf and modes[0]:
            optimizer.train()
        model.train(was_training)


@torch.no_grad()
def evaluate(model, dataset, task, batch_size=128, max_examples=1024, optimizer=None):
    total_loss = 0.0
    total_units = 0
    correct = 0
    n = min(len(dataset), max_examples)
    if n <= 0:
        raise ValueError('Evaluation dataset must be nonempty')
    with evaluation_mode(model, optimizer):
        for start in range(0, n, batch_size):
            x, y = _batch(dataset, range(start, min(start + batch_size, n)))
            logits = model(x)
            if task == 'tiny_shakespeare':
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='sum')
            else:
                loss = F.cross_entropy(logits, y, reduction='sum')
                correct += int((logits.argmax(1) == y).sum())
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite evaluation loss')
            total_loss += float(loss)
            total_units += y.numel()
    return total_loss / total_units, (100.0 * correct / total_units if task != 'tiny_shakespeare' else None)


def optimizer_state_bytes(optimizer):
    return sum(v.numel() * v.element_size() for state in optimizer.state.values()
               for v in state.values() if torch.is_tensor(v))


def train_one(model, optimizer, data, task, seed, max_updates, batch_size, eval_interval,
              target_val_ce, eval_examples=1024, evaluate_test=False, scheduler=None, on_record=None):
    if max_updates < 1 or eval_interval < 1 or batch_size < 1:
        raise ValueError('Updates, interval and batch size must be positive')
    if callable(getattr(optimizer, 'train', None)):
        optimizer.train()
    generator = torch.Generator().manual_seed(int(seed) + 100_000)
    metrics = []
    target_step = None
    target_time = None
    started = time.perf_counter()

    def record(step):
        train_ce, _ = evaluate(model, data.train, task, max_examples=eval_examples, optimizer=optimizer)
        val_ce, val_acc = evaluate(model, data.validation, task, max_examples=eval_examples, optimizer=optimizer)
        elapsed = time.perf_counter() - started
        metrics.append({"step": step, "elapsed_s": elapsed, "train_ce": train_ce,
                        "val_ce": val_ce, "val_accuracy_pct": val_acc,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if on_record is not None:
            on_record(metrics[-1])
        return val_ce, elapsed

    val_ce, elapsed = record(0)
    if target_val_ce is not None and val_ce <= target_val_ce:
        target_step, target_time = 0, elapsed
    model.train()
    for step in range(1, max_updates + 1):
        indices = torch.randint(len(data.train), (batch_size,), generator=generator)
        x, y = _batch(data.train, indices)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if task == "tiny_shakespeare":
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        else:
            loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError(f'non-finite gradient at step {step}')
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if step % eval_interval == 0 or step == max_updates:
            val_ce, elapsed = record(step)
            if target_val_ce is not None and target_step is None and val_ce <= target_val_ce:
                target_step, target_time = step, elapsed

    training_elapsed = time.perf_counter() - started
    test_ce, test_acc = (evaluate(model, data.test, task, max_examples=eval_examples, optimizer=optimizer)
                         if evaluate_test else (None, None))
    final = metrics[-1]
    return metrics, {"final_train_ce": final["train_ce"], "final_val_ce": final["val_ce"],
                     "test_accuracy_pct": test_acc, "test_ce": test_ce,
                     "elapsed_s": training_elapsed, "target_val_ce": target_val_ce,
                     "target_step": target_step, "target_time_s": target_time,
                     "target_reached": (target_step is not None) if target_val_ce is not None else None,
                     "optimizer_state_bytes": optimizer_state_bytes(optimizer)}
