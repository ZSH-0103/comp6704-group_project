"""Shared CPU update-budget trainer with auditable pairing and failure logs."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import torch
from torch import nn
from torch.utils.data import DataLoader, default_collate

from .utils import environment, hash_json, hash_state, write_json


class PolynomialAverager:
    """Separate evaluation model; gamma=8, updated after each optimizer step."""
    def __init__(self, model, gamma=8.):
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        self.model = deepcopy(model).requires_grad_(False)
        self.gamma, self.count = gamma, 0

    @torch.no_grad()
    def update(self, base):
        self.count += 1
        weight = (self.gamma + 1) / (self.gamma + self.count)
        source = base.state_dict()
        for name, tensor in self.model.state_dict().items():
            if tensor.is_floating_point():
                tensor.lerp_(source[name], weight)
            else:
                tensor.copy_(source[name])

    def state_dict(self):
        return dict(gamma=self.gamma, count=self.count, model=self.model.state_dict())

    def load_state_dict(self, state):
        self.gamma, self.count = state["gamma"], state["count"]
        self.model.load_state_dict(state["model"])


@contextmanager
def evaluation_mode(model, optimizer):
    """SF-style optimizer hooks restore train parameters even if evaluation fails."""
    training = model.training
    has_eval, has_train = callable(getattr(optimizer, "eval", None)), callable(getattr(optimizer, "train", None))
    if has_eval != has_train:
        raise ValueError("Optimizer must expose both train/eval hooks or neither")
    try:
        if has_eval:
            optimizer.eval()
        model.eval()
        yield
    finally:
        if has_train:
            optimizer.train()
        model.train(training)


def cross_entropy(logits, target, reduction="mean"):
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction=reduction)


@torch.no_grad()
def evaluate(model, dataset, batch_size, is_text):
    """Mean CE weighted by examples (images) or predicted characters (text)."""
    old_mode = model.training
    model.eval()
    loss_sum, count, correct = 0., 0, 0
    try:
        for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            logits = model(x)
            loss_sum += float(cross_entropy(logits, y, reduction="sum"))
            count += y.numel()
            correct += int((logits.argmax(-1) == y).sum())
        if count == 0:
            raise ValueError("Cannot evaluate empty dataset")
        ce = loss_sum / count
        if not math.isfinite(ce):
            raise FloatingPointError("Nonfinite evaluation CE")
        return dict(ce=ce, accuracy_pct=None if is_text else 100. * correct / count, count=count)
    finally:
        model.train(old_mode)


def batch_indices(size, batch_size, seed):
    """Fresh seeded permutations each epoch; retain the short last batch."""
    if size <= 0 or batch_size <= 0:
        raise ValueError("Dataset size and batch size must be positive")
    generator = torch.Generator().manual_seed(seed)
    while True:
        permutation = torch.randperm(size, generator=generator).tolist()
        for start in range(0, size, batch_size):
            yield permutation[start:start + batch_size]


def optimizer_tensor_bytes(optimizer):
    """Count unique state tensors, including group state, but excluding params."""
    seen = set()
    def size(value):
        if isinstance(value, torch.Tensor):
            key = (value.device, value.data_ptr())
            if key in seen:
                return 0
            seen.add(key)
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(size(v) for k, v in value.items() if k != "params")
        if isinstance(value, (tuple, list)):
            return sum(size(v) for v in value)
        return 0
    return size(dict(state=optimizer.state, param_groups=optimizer.param_groups))


def train_one(model, optimizer, data, config):
    """Return a summary; always retain available per-run evidence after failures.

    config includes output_root, member, phase, task, optimizer, seed, budgets,
    model metadata, and complete shared/member configuration status. No downloads.
    """
    cfg = dict(config)
    if cfg["phase"] == "main" and (cfg["shared_status"] != "FROZEN" or cfg["member_status"] != "FROZEN"):
        raise ValueError("Main runs require frozen shared and member protocols")
    if cfg.get("evaluate_test") and cfg["phase"] != "main":
        raise ValueError("Test evaluation is allowed only in frozen main runs")
    if cfg["max_updates"] <= 0 or cfg["eval_interval"] <= 0:
        raise ValueError("Update budget and evaluation interval must be positive")
    if cfg.get("evaluation_point", "raw") not in ("raw", "averaged"):
        raise ValueError("Choose raw or averaged evaluation point")
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Shared pipeline requires CPU parameters")
    actual_initialization = hash_state(model)
    if actual_initialization != cfg["initialization_hash"]:
        raise ValueError("Model weights do not match recorded initialization")
    cfg.update(data.metadata)
    config_hash = hash_json(cfg)
    run_id = (f"m{cfg['member']}_{cfg['phase']}_{cfg['task']}_{cfg['optimizer']}_s{cfg['seed']}_"
              f"{config_hash[:12]}_{uuid.uuid4().hex[:8]}")
    directory = Path(cfg["output_root"]) / "raw" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", dict(cfg, config_hash=config_hash, run_id=run_id))
    write_json(directory / "environment.json", environment())
    averaging = cfg.get("averaging")
    if averaging and averaging["kind"] != "polynomial":
        raise ValueError("Only polynomial averaging is supported")
    averager = PolynomialAverager(model, averaging["gamma"]) if averaging else None
    if cfg.get("evaluation_point") == "averaged" and averager is None:
        raise ValueError("Averaged evaluation requires an averager")
    torch.manual_seed(cfg["seed"])
    model.train()
    if callable(getattr(optimizer, "train", None)):
        optimizer.train()
    batches = batch_indices(len(data.train), cfg["batch_size"], cfg["seed"])
    order_hash = hashlib.sha256()
    target_step = target_time = None
    status, error, last, step = "completed", None, None, 0
    test_result = None
    is_text = cfg["task"] == "tiny_shakespeare"
    start = time.perf_counter()
    def record():
        nonlocal target_step, target_time, last
        selected = averager.model if cfg.get("evaluation_point") == "averaged" else model
        with evaluation_mode(model, optimizer):
            train_result = evaluate(selected, data.train_eval, cfg["eval_batch_size"], is_text)
            val_result = evaluate(selected, data.validation, cfg["eval_batch_size"], is_text)
            raw = None
            if cfg.get("evaluate_raw_also") and selected is not model:
                raw = evaluate(model, data.validation, cfg["eval_batch_size"], is_text)
        lr = optimizer.param_groups[0].get("lr")
        if hasattr(optimizer, "diagnostics"):
            lr = optimizer.diagnostics.get("step_size")
        last = dict(step=step, elapsed_s=time.perf_counter() - start, train_ce=train_result["ce"],
                    val_ce=val_result["ce"], val_accuracy_pct=val_result["accuracy_pct"],
                    learning_rate=None if lr is None else float(lr), evaluation_point=cfg.get("evaluation_point", "raw"),
                    raw_val_ce=None if raw is None else raw["ce"],
                    raw_val_accuracy_pct=None if raw is None else raw["accuracy_pct"],
                    data_order_hash=order_hash.hexdigest(), train_ce_scope=data.metadata["train_ce_scope"])
        if hasattr(optimizer, "diagnostics"):
            last["optimizer_diagnostics"] = optimizer.diagnostics
        threshold = cfg.get("target_val_ce")
        if target_step is None and threshold is not None and last["val_ce"] <= threshold:
            target_step, target_time = step, last["elapsed_s"]
        metrics.write(json.dumps(last, allow_nan=False) + "\n")
        metrics.flush()
    with (directory / "metrics.jsonl").open("w") as metrics:
        try:
            record()  # includes initial evaluation in measured run time
            for step in range(1, cfg["max_updates"] + 1):
                if time.perf_counter() - start > cfg.get("timeout_s", float("inf")):
                    status = "timeout"
                    error = "Configured run timeout exceeded"
                    step -= 1
                    break
                ids = next(batches)
                order_hash.update(json.dumps(ids, separators=(",", ":")).encode() + b"\n")
                x, y = default_collate([data.train[i] for i in ids])
                optimizer.zero_grad(set_to_none=True)
                loss = cross_entropy(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite minibatch loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite gradient")
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite model weights")
                if averager is not None:
                    averager.update(model)
                if step % cfg["eval_interval"] == 0 or step == cfg["max_updates"]:
                    record()
            if status == "completed" and cfg.get("evaluate_test"):
                selected = averager.model if cfg.get("evaluation_point") == "averaged" else model
                with evaluation_mode(model, optimizer):
                    test_result = evaluate(selected, data.test(), cfg["eval_batch_size"], is_text)
        except FloatingPointError as exc:
            status, error = "diverged", str(exc)
        except Exception as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start
        if status != "completed":
            metrics.write(json.dumps(dict(step=step, elapsed_s=elapsed, event="run_failed", status=status,
                                          error_message=error), allow_nan=False) + "\n")
    summary = dict(run_id=run_id, phase=cfg["phase"], task=cfg["task"], optimizer=cfg["optimizer"],
                   seed=cfg["seed"], config_hash=config_hash, status=status, error_message=error,
                   final_train_ce=None if last is None else last["train_ce"],
                   final_val_ce=None if last is None else last["val_ce"],
                   final_val_accuracy_pct=None if last is None else last["val_accuracy_pct"],
                   last_evaluated_step=None if last is None else last["step"], steps_attempted=step,
                   test_accuracy_pct=None if test_result is None else test_result["accuracy_pct"],
                   test_ce=None if test_result is None else test_result["ce"], elapsed_s=elapsed,
                   target_val_ce=cfg.get("target_val_ce"), target_step=target_step, target_time_s=target_time,
                   target_reached=target_step is not None, optimizer_state_bytes=optimizer_tensor_bytes(optimizer),
                   averaged_model_state_bytes=0 if averager is None else sum(t.numel()*t.element_size() for t in averager.model.state_dict().values()),
                   evaluation_point=cfg.get("evaluation_point", "raw"), data_order_hash=order_hash.hexdigest(),
                   initialization_hash=actual_initialization, final_parameter_hash=hash_state(model),
                   split_hash=data.metadata["split_hash"], model_hash=cfg["model_hash"])
    write_json(directory / "summary.json", summary)
    return summary
