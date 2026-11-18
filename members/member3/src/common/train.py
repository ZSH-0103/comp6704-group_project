"""Paired, fixed-update CPU training and evaluation utilities."""
import math
import time
import torch
from torch.nn import functional as F


def _batch(dataset, indices):
    items = [dataset[int(i)] for i in indices]
    xs = [x if torch.is_tensor(x) else torch.as_tensor(x) for x, _ in items]
    ys = [y if torch.is_tensor(y) else torch.as_tensor(y) for _, y in items]
    return torch.stack(xs), torch.stack(ys)


@torch.no_grad()
def evaluate(model, dataset, task, batch_size=128, max_examples=1024):
    model.eval()
    total_loss = 0.0
    total_units = 0
    correct = 0
    n = min(len(dataset), max_examples)
    for start in range(0, n, batch_size):
        x, y = _batch(dataset, range(start, min(start + batch_size, n)))
        logits = model(x)
        if task == "tiny_shakespeare":
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
            units = y.numel()
        else:
            loss = F.cross_entropy(logits, y, reduction="sum")
            units = y.numel()
            correct += int((logits.argmax(1) == y).sum())
        total_loss += float(loss)
        total_units += units
    model.train()
    return total_loss / total_units, (100.0 * correct / total_units if task != "tiny_shakespeare" else None)


def optimizer_state_bytes(optimizer):
    return sum(v.numel() * v.element_size() for state in optimizer.state.values()
               for v in state.values() if torch.is_tensor(v))


def train_one(model, optimizer, data, task, seed, max_updates, batch_size, eval_interval,
              target_val_ce, eval_examples=1024, evaluate_test=True):
    generator = torch.Generator().manual_seed(int(seed) + 100_000)
    metrics = []
    target_step = None
    target_time = None
    started = time.perf_counter()

    def record(step):
        train_ce, _ = evaluate(model, data.train, task, max_examples=eval_examples)
        val_ce, val_acc = evaluate(model, data.validation, task, max_examples=eval_examples)
        elapsed = time.perf_counter() - started
        metrics.append({"step": step, "elapsed_s": elapsed, "train_ce": train_ce,
                        "val_ce": val_ce, "val_accuracy_pct": val_acc,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        return val_ce, elapsed

    val_ce, elapsed = record(0)
    if val_ce <= target_val_ce:
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
        optimizer.step()
        if step % eval_interval == 0 or step == max_updates:
            val_ce, elapsed = record(step)
            if target_step is None and val_ce <= target_val_ce:
                target_step, target_time = step, elapsed

    test_ce, test_acc = (evaluate(model, data.test, task, max_examples=eval_examples)
                         if evaluate_test else (None, None))
    final = metrics[-1]
    return metrics, {"final_train_ce": final["train_ce"], "final_val_ce": final["val_ce"],
                     "test_accuracy_pct": test_acc, "test_ce": test_ce,
                     "elapsed_s": time.perf_counter() - started, "target_val_ce": target_val_ce,
                     "target_step": target_step, "target_time_s": target_time,
                     "target_reached": target_step is not None,
                     "optimizer_state_bytes": optimizer_state_bytes(optimizer)}
