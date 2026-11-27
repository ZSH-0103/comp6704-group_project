"""Reproducible exact-gradient quadratic study; separate from main NN runs.

Run via scripts/run_member.py --member 1 --phase diagnostic.
Pilot: python -m src.members.member1_diagnostics --pilot
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import uuid

import numpy as np
import torch

from .member1_optimizers import SGD, DoG


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def problem(cfg, kappa, scale, seed):
    """Fixed eigenvectors, geometric eigenvalues, known minimizer x*=0."""
    d = cfg["dimension"]
    q, _ = np.linalg.qr(np.random.default_rng(cfg["rotation_seed"]).normal(size=(d, d)))
    eigenvalues = np.geomspace(1., float(kappa), d)
    h = torch.tensor(scale * (q * eigenvalues) @ q.T, dtype=torch.float64)
    x = np.random.default_rng(seed).normal(size=d)
    x *= cfg["initial_norm"] / np.linalg.norm(x)
    return h, torch.tensor(x, dtype=torch.float64)


def conditions(cfg):
    pairs = [(k, cfg["reference_scale"]) for k in cfg["condition_numbers"]]
    pairs += [(cfg["reference_condition_number"], a) for a in cfg["objective_scales"]]
    return list(dict.fromkeys(pairs))  # reference is measured once, reused in both views


def condition_id(k, a):
    return f"k{k:g}_a{a:g}"


def environment():
    def sysctl(key):
        try:
            return subprocess.check_output(["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                platform=platform.platform(), cpu=sysctl("machdep.cpu.brand_string"),
                ram_bytes=sysctl("hw.memsize"), threads=torch.get_num_threads(), device="cpu",
                git_commit=None, git_note="Project directory is not a Git checkout")


def run_one(root, cfg, batch, env, kappa, scale, seed, optimizer, phase, lr=None):
    h, initial = problem(cfg, kappa, scale, seed)
    source = Path(__file__).with_name("member1_optimizers.py")
    resolved = dict(protocol=cfg, batch=batch, task="quadratic", condition=condition_id(kappa, scale),
                    kappa=kappa, scale=scale, seed=seed, optimizer=optimizer, phase=phase, lr=lr,
                    model_hash=digest(h.tolist()), initialization_hash=digest(initial.tolist()),
                    dataset_hash=None, split_hash=None, schedule="constant SGD; adaptive DoG",
                    implementation_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    hash_id = digest(resolved)
    run_id = f"m1_{phase}_quadratic_{optimizer}_s{seed}_{hash_id}"
    directory = root / "raw" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", resolved)
    write_json(directory / "environment.json", env)
    x = torch.nn.Parameter(initial.clone())
    opt = SGD([x], lr=lr) if optimizer == "sgd" else DoG([x], **cfg["dog"])
    def gap():
        return float((x.detach() @ h @ x.detach()) / 2)
    initial_gap = gap()
    metrics = [dict(step=0, elapsed_s=0., objective_gap=initial_gap, relative_gap=1.,
                    learning_rate=None, rbar=None, grad_sq_sum=None)]
    target_step = target_time = None
    status, error = "completed", None
    start = time.perf_counter()
    try:
        for step in range(1, cfg["max_updates"] + 1):
            x.grad = h @ x.detach()
            opt.step()
            value = gap()
            if not math.isfinite(value) or value < 0:
                raise FloatingPointError("Nonfinite or negative objective gap")
            relative = value / initial_gap
            elapsed = time.perf_counter() - start
            diag = opt.diagnostics if optimizer == "dog" else {}
            metrics.append(dict(step=step, elapsed_s=elapsed, objective_gap=value,
                                relative_gap=relative, learning_rate=lr if optimizer == "sgd" else diag["step_size"],
                                rbar=diag.get("rbar"), grad_sq_sum=diag.get("grad_sq_sum")))
            if target_step is None and relative <= cfg["target_relative_gap"]:
                target_step, target_time = step, elapsed
            if relative > cfg["divergence_relative_gap"]:
                raise FloatingPointError("Predeclared divergence threshold exceeded")
    except FloatingPointError as exc:
        status, error = "diverged", str(exc)
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - start
    summary = dict(run_id=run_id, phase=phase, task="quadratic", optimizer=optimizer, seed=seed,
                   config_hash=hash_id, condition=condition_id(kappa, scale), kappa=kappa, scale=scale,
                   status=status, error_message=error, initial_gap=initial_gap,
                   final_objective_gap=metrics[-1]["objective_gap"] if status == "completed" else None,
                   final_relative_gap=metrics[-1]["relative_gap"] if status == "completed" else None,
                   elapsed_s=elapsed, target_relative_gap=cfg["target_relative_gap"], target_step=target_step,
                   target_time_s=target_time, target_reached=target_step is not None, lr=lr,
                   optimizer_state_bytes=sum(t.numel() * t.element_size() for s in opt.state.values()
                                             for t in s.values() if isinstance(t, torch.Tensor)),
                   state_bytes_scope="persistent tensor buffers only; excludes Python scalars and temporaries",
                   final_train_ce=None, final_val_ce=None, test_accuracy_pct=None, test_ce=None,
                   target_val_ce=None, steps_recorded=metrics[-1]["step"])
    with (directory / "metrics.jsonl").open("w") as file:
        for row in metrics:
            row.update(train_ce=None, val_ce=None, val_accuracy_pct=None)
            file.write(json.dumps(row, allow_nan=False) + "\n")
    write_json(directory / "summary.json", summary)
    return summary


def aggregate(summaries, cfg, experiment_hash):
    rows = []
    for k, a in conditions(cfg):
        for opt in ("sgd", "dog"):
            group = [r for r in summaries if r["kappa"] == k and r["scale"] == a and r["optimizer"] == opt]
            assert len(group) == len(cfg["seeds"]), "Missing comparison runs"
            completed = [r for r in group if r["status"] == "completed"]
            reached = [r for r in group if r["target_reached"]]
            values = {
                "final_objective_gap": ([r["final_objective_gap"] for r in completed], "objective units"),
                "final_relative_gap": ([r["final_relative_gap"] for r in completed], "fraction of initial gap"),
                "target_reached_rate": ([float(r["target_reached"]) for r in group], "fraction of all runs"),
                "completed_rate": ([float(r["status"] == "completed") for r in group], "fraction of all runs"),
                "target_steps_reached_only": ([r["target_step"] for r in reached], "updates; conditional on reaching"),
                "target_time_s_reached_only": ([r["target_time_s"] for r in reached], "seconds; conditional on reaching"),
            }
            for name, (v, unit) in values.items():
                rows.append(dict(diagnostic_id=cfg["experiment"], condition=condition_id(k, a), optimizer=opt,
                                 metric_name=name, metric_unit=unit, n=len(v), mean=float(np.mean(v)) if v else None,
                                 sd=float(np.std(v, ddof=1)) if len(v) > 1 else None, config_hash=experiment_hash))
    return rows


def run_experiment(repo_root, pilot=False):
    cfg = json.loads((repo_root / "configs/member1_quadratic.json").read_text())
    if not pilot and cfg["status"] != "FROZEN":
        raise ValueError("Freeze the quadratic protocol after its CPU pilot")
    torch.set_num_threads(cfg["threads"])
    root = repo_root / "results/member1"
    batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    env = environment()
    if pilot:
        cfg = dict(cfg, max_updates=50)
        runs = [run_one(root, cfg, batch, env, 10, 1, cfg["tuning_seed"], opt,
                        "diagnostic_pilot", 0.1 if opt == "sgd" else None) for opt in ("sgd", "dog")]
        write_json(root / "notes" / f"quadratic_pilot_{batch}.json", runs)
        print(json.dumps(runs, indent=2))
        return runs
    evidence = root / "notes/quadratic" / batch
    evidence.mkdir(parents=True, exist_ok=False)
    write_json(evidence / "protocol.json", cfg)
    # Preserve the exact executed experiment and plotting code with each batch.
    for name in ("member1_diagnostics.py", "member1_quadratic_plots.py", "member1_optimizers.py"):
        (evidence / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    trials, selected, summaries = [], {}, []
    for k, a in conditions(cfg):
        current = []
        for c in cfg["sgd_lr_times_L"]:
            lr = c / (a * k)
            r = run_one(root, cfg, batch, env, k, a, cfg["tuning_seed"], "sgd", "diagnostic_tune", lr)
            current.append(r)
        eligible = [r for r in current if r["status"] == "completed"]
        if not eligible:
            raise RuntimeError(f"All tuning trials failed at {condition_id(k, a)}")
        winner = min(eligible, key=lambda r: (r["final_relative_gap"], r["lr"]))
        selected[condition_id(k, a)] = winner["lr"]
        for r in current:
            trials.append(dict(trial_id=r["run_id"], condition=r["condition"], seed=r["seed"],
                               lr=r["lr"], lr_times_L=r["lr"] * a * k,
                               final_relative_gap=r["final_relative_gap"], status=r["status"],
                               elapsed_s=r["elapsed_s"], selected=r["run_id"] == winner["run_id"]))
        print(f"{condition_id(k, a)}: selected SGD lr={winner['lr']:.6g}", flush=True)
        for seed in cfg["seeds"]:
            for opt in ("sgd", "dog"):
                summaries.append(run_one(root, cfg, batch, env, k, a, seed, opt, "diagnostic",
                                         winner["lr"] if opt == "sgd" else None))
    # Immutable batch manifest lists every trial and comparison, including failures.
    manifest = dict(batch=batch, config=cfg, config_hash=digest(cfg), selected_sgd_lr=selected,
                    tuning=trials, runs=summaries)
    write_json(evidence / "manifest.json", manifest)
    write_csv(evidence / "diagnostic_results.csv", aggregate(summaries, cfg, digest(cfg)))
    write_csv(evidence / "tuning_log.csv", trials)
    from .member1_quadratic_plots import export_plots
    export_plots(root, evidence / "manifest.json")
    write_json(root / "notes/quadratic_latest.json", {"manifest": str((evidence / "manifest.json").relative_to(root))})
    print(f"Completed {len(summaries)} comparison runs and {len(trials)} tuning trials.\nEvidence: {evidence}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    run_experiment(Path(__file__).resolve().parents[2], pilot=args.pilot)
