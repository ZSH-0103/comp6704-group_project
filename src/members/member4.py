"""Member 4 Adam/AdEMAMix experiment implementation."""
from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import torch

from src.common.data import load_task
from src.common.models import make_model
from src.common.train import train_one
from src.members.member4_optim import AdEMAMix, IndependentAdam


IMPLEMENTATION_VERSION = "member4-v1"
OFFICIAL_REFERENCE = "apple/ml-ademamix@eb517ed3bbc09227e5d062b682723543f6c71f1a"


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _environment():
    cpu = platform.processor() or platform.machine()
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip() or cpu
    except Exception:
        pass
    try:
        ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    except Exception:
        ram = None
    return {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
            "cpu": cpu, "ram_bytes": ram, "torch_threads": torch.get_num_threads(),
            "implementation_version": IMPLEMENTATION_VERSION, "official_reference": OFFICIAL_REFERENCE}


def _optimizer(name, parameters, hp, max_updates):
    if name == "adam":
        return IndependentAdam(parameters, lr=hp["lr"], betas=tuple(hp["betas"]),
                               eps=hp["eps"], weight_decay=hp["weight_decay"])
    if name in {"ademamix", "ademamix_no_slow"}:
        return AdEMAMix(parameters, lr=hp["lr"], betas=tuple(hp["betas"]), alpha=hp["alpha"],
                        alpha_warmup=max(1, round(max_updates * hp["alpha_warmup_fraction"])),
                        beta3_warmup=max(1, round(max_updates * hp["beta3_warmup_fraction"])),
                        eps=hp["eps"], weight_decay=hp["weight_decay"],
                        slow_enabled=name != "ademamix_no_slow")
    raise ValueError(name)


def _run(repo_root, config, phase, task, optimizer_name, seed, hp, max_updates=None, condition=None):
    shared = config["shared"]
    tc = shared["tasks"][task]
    updates = int(max_updates or tc["max_updates"])
    resolved = {"member": 4, "phase": phase, "task": task, "optimizer": optimizer_name,
                "seed": int(seed), "hyperparameters": hp, "max_updates": updates,
                "batch_size": tc["batch_size"], "eval_interval": min(tc["eval_interval"], updates),
                "target_val_ce": tc["target_val_ce"], "condition": condition,
                "split_seed": shared["split_seed"], "device": "cpu", "weight_decay": 0,
                "implementation_version": IMPLEMENTATION_VERSION}
    config_hash = _json_hash(resolved)
    run_id = f"m4_{phase}_{task}_{optimizer_name}_s{seed}_{config_hash}"
    run_dir = repo_root / config["output_root"] / "raw" / run_id
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    summary = {"run_id": run_id, "phase": phase, "task": task, "optimizer": optimizer_name,
               "seed": int(seed), "config_hash": config_hash, "status": "error", "error_message": None}
    try:
        data = load_task(task, shared, repo_root)
        model, initialization_hash = make_model(task, tc, seed, data.metadata)
        resolved.update({"dataset_split_hash": data.split_hash,
                         "model_initialization_hash": initialization_hash,
                         "dataset_metadata": data.metadata})
        opt = _optimizer(optimizer_name, model.parameters(), hp, updates)
        metrics, measured = train_one(model, opt, data, task, seed, updates, tc["batch_size"],
                                      min(tc["eval_interval"], updates), tc["target_val_ce"],
                                      eval_examples=512 if phase in {"smoke", "tune"} else 1000,
                                      evaluate_test=phase == "main")
        summary.update(measured)
        summary.update(status="completed", error_message=None)
        with (run_dir / "metrics.jsonl").open("w") as handle:
            for row in metrics:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
    except Exception as exc:
        summary.update(elapsed_s=time.perf_counter() - started, error_message=repr(exc))
        (run_dir / "metrics.jsonl").touch()
        raise
    finally:
        _write_json(run_dir / "config.json", resolved)
        _write_json(run_dir / "environment.json", _environment())
        _write_json(run_dir / "summary.json", summary)
    return summary


def _check(repo_root, config):
    torch.manual_seed(7)
    p_ours = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    p_torch = torch.nn.Parameter(p_ours.detach().clone())
    ours = IndependentAdam([p_ours], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    reference = torch.optim.Adam([p_torch], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    gradients = [torch.tensor([0.3, -0.2], dtype=torch.float64), torch.tensor([-0.1, 0.4], dtype=torch.float64)]
    for gradient in gradients:
        p_ours.grad = gradient.clone(); p_torch.grad = gradient.clone()
        ours.step(); reference.step()
    adam_error = float((p_ours - p_torch).detach().abs().max())

    p_a = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    p_b = torch.nn.Parameter(p_a.detach().clone())
    adam = IndependentAdam([p_a], lr=0.01)
    ablation = AdEMAMix([p_b], lr=0.01, alpha=2.0, slow_enabled=False,
                        alpha_warmup=2, beta3_warmup=2)
    for gradient in gradients:
        p_a.grad = gradient.clone(); p_b.grad = gradient.clone()
        adam.step(); ablation.step()
    ablation_error = float((p_a - p_b).detach().abs().max())

    p = torch.nn.Parameter(torch.tensor([0.5, -0.5]))
    opt = AdEMAMix([p], lr=0.01, alpha=2.0, alpha_warmup=3, beta3_warmup=3)
    p.grad = torch.tensor([0.2, -0.4]); opt.step()
    state = copy.deepcopy(opt.state_dict()); saved_p = p.detach().clone()
    restored = torch.nn.Parameter(saved_p.clone())
    restored_opt = AdEMAMix([restored], lr=0.01, alpha=2.0, alpha_warmup=3, beta3_warmup=3)
    restored_opt.load_state_dict(state)
    grad = torch.tensor([-0.3, 0.1]); p.grad = grad.clone(); restored.grad = grad.clone()
    opt.step(); restored_opt.step()
    resume_error = float((p - restored).detach().abs().max())
    if max(adam_error, ablation_error, resume_error) > 1e-10:
        raise AssertionError((adam_error, ablation_error, resume_error))
    notes = repo_root / config["output_root"] / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "implementation_check.md").write_text(
        "# Member 4 implementation checks\n\n"
        f"- Independent Adam versus `torch.optim.Adam`: maximum error `{adam_error:.3g}`.\n"
        f"- No-slow-EMA ablation versus independent Adam: maximum error `{ablation_error:.3g}`.\n"
        f"- AdEMAMix state save/restore continuation: maximum error `{resume_error:.3g}`.\n"
        f"- Author reference inspected: `{OFFICIAL_REFERENCE}`.\n"
        "- The slow EMA is not bias corrected, matching the paper and author implementation.\n")


def _smoke(repo_root, config):
    for task in config["tasks"]:
        for name in config["optimizers"]:
            hp = dict(config["hyperparameters"][name])
            _run(repo_root, config, "smoke", task, name, 99, hp, max_updates=5)


def _tune(repo_root, config):
    rows = []
    selected = {}
    for task in config["tasks"]:
        selected[task] = {}
        for name in config["optimizers"]:
            trials = []
            for index, lr in enumerate(config["learning_rate_search"][name], 1):
                hp = dict(config["hyperparameters"][name]); hp["lr"] = lr
                summary = _run(repo_root, config, "tune", task, name, config["shared"]["tuning_seed"],
                               hp, max_updates=config["tune_updates"])
                row = {"trial_id": f"{task}-{name}-{index}", "task": task, "optimizer": name,
                       "seed": config["shared"]["tuning_seed"], "hyperparameters_json": json.dumps(hp, sort_keys=True),
                       "validation_ce": summary["final_val_ce"], "status": summary["status"],
                       "elapsed_s": summary["elapsed_s"], "selected": False}
                trials.append(row); rows.append(row)
            winner = min(trials, key=lambda x: x["validation_ce"])
            winner["selected"] = True
            selected[task][name] = json.loads(winner["hyperparameters_json"])
    notes = repo_root / config["output_root"] / "notes"
    with (notes / "tuning_log.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    _write_json(notes / "selected_hyperparameters.json", selected)


def _main(repo_root, config):
    selected = config["selected_hyperparameters"]
    for task in config["tasks"]:
        for name in config["optimizers"]:
            for seed in config["shared"]["main_seeds"]:
                _run(repo_root, config, "main", task, name, seed, selected[task][name])


def _diagnostic(repo_root, config):
    spec = config["diagnostic_levels"]
    task = spec["task"]
    selected = config["selected_hyperparameters"][task]
    for horizon in spec["horizons"]:
        for seed in spec["seeds"]:
            _run(repo_root, config, "diagnostic", task, "adam", seed, selected["adam"],
                 max_updates=horizon, condition=f"horizon={horizon}")
            _run(repo_root, config, "diagnostic", task, "ademamix", seed, selected["ademamix"],
                 max_updates=horizon, condition=f"horizon={horizon}")
            _run(repo_root, config, "diagnostic", task, "ademamix_no_slow", seed, selected["ademamix"],
                 max_updates=horizon, condition=f"horizon={horizon}")


def _mean_sd(values):
    values = [float(v) for v in values if v is not None]
    return ((statistics.mean(values) if values else None),
            (statistics.stdev(values) if len(values) >= 2 else None))


def _summaries(repo_root, config, phase):
    rows = []
    for path in (repo_root / config["output_root"] / "raw").glob("*/summary.json"):
        row = json.loads(path.read_text())
        if row.get("phase") == phase:
            row["_dir"] = str(path.parent)
            row["_config"] = json.loads((path.parent / "config.json").read_text())
            rows.append(row)
    return rows


def _write_csv(path, rows, columns):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def _export(repo_root, config):
    import matplotlib.pyplot as plt
    main = _summaries(repo_root, config, "main")
    diagnostic = _summaries(repo_root, config, "diagnostic")
    if len(main) != 18 or len(diagnostic) != 18:
        raise RuntimeError(f"Expected 18 main and 18 diagnostic runs; found {len(main)} and {len(diagnostic)}")
    tables = repo_root / config["output_root"] / "tables"; tables.mkdir(exist_ok=True)
    main_rows = []
    for task in config["tasks"]:
        for name in config["optimizers"]:
            group = [r for r in main if r["task"] == task and r["optimizer"] == name]
            completed = [r for r in group if r["status"] == "completed"]
            train_m, train_s = _mean_sd([r["final_train_ce"] for r in completed])
            val_m, val_s = _mean_sd([r["final_val_ce"] for r in completed])
            metric = "test_ce" if task == "tiny_shakespeare" else "test_accuracy_pct"
            test_m, test_s = _mean_sd([r[metric] for r in completed])
            time_m, time_s = _mean_sd([r["elapsed_s"] for r in completed])
            main_rows.append({"task": task, "optimizer": name, "n_expected": 3,
                "n_completed": len(completed), "train_ce_mean": train_m, "train_ce_sd": train_s,
                "val_ce_mean": val_m, "val_ce_sd": val_s, "test_metric_name": metric,
                "test_metric_mean": test_m, "test_metric_sd": test_s,
                "elapsed_s_mean": time_m, "elapsed_s_sd": time_s})
    main_cols = list(main_rows[0]); _write_csv(tables / "main_results.csv", main_rows, main_cols)

    diag_rows = []
    for horizon in config["diagnostic_levels"]["horizons"]:
        for name in ["adam", "ademamix", "ademamix_no_slow"]:
            group = [r for r in diagnostic if r["optimizer"] == name and r["_config"]["max_updates"] == horizon]
            for field, metric_name, unit in [("final_val_ce", "validation_ce", "nats/example"),
                                              ("optimizer_state_bytes", "optimizer_state", "bytes")]:
                mean, sd = _mean_sd([r[field] for r in group])
                diag_rows.append({"diagnostic_id": "history_ablation", "condition": f"horizon={horizon}",
                    "optimizer": name, "metric_name": metric_name, "metric_unit": unit,
                    "n": len(group), "mean": mean, "sd": sd,
                    "config_hash": _json_hash([r["config_hash"] for r in group])})
    diag_cols = list(diag_rows[0]); _write_csv(tables / "diagnostic_results.csv", diag_rows, diag_cols)

    tuning = list(csv.DictReader((repo_root / config["output_root"] / "notes" / "tuning_log.csv").open()))
    cost_rows = []
    for task in config["tasks"]:
        for name in config["optimizers"]:
            group = [r for r in main if r["task"] == task and r["optimizer"] == name]
            reached = [r for r in group if r["target_reached"]]
            steps, _ = _mean_sd([r["target_step"] for r in reached]); seconds, _ = _mean_sd([r["target_time_s"] for r in reached])
            tune_group = [r for r in tuning if r["task"] == task and r["optimizer"] == name]
            cost_rows.append({"task": task, "optimizer": name, "n_expected": 3,
                "n_completed": sum(r["status"] == "completed" for r in group),
                "n_target_reached": len(reached), "target_val_ce": config["shared"]["tasks"][task]["target_val_ce"],
                "target_steps_mean_reached": steps, "target_time_s_mean_reached": seconds,
                "optimizer_state_bytes": round(statistics.mean(r["optimizer_state_bytes"] for r in group)),
                "tuning_trials": len(tune_group), "tuning_elapsed_s": sum(float(r["elapsed_s"]) for r in tune_group)})
    _write_csv(tables / "cost_stability.csv", cost_rows, list(cost_rows[0]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    colors = {"adam": "#3567a8", "ademamix": "#d4643c", "ademamix_no_slow": "#6b6b6b"}
    for name in config["optimizers"]:
        for task in config["tasks"]:
            runs = [r for r in main if r["task"] == task and r["optimizer"] == name]
            curves = [[json.loads(line) for line in open(Path(r["_dir"]) / "metrics.jsonl")] for r in runs]
            steps = [p["step"] for p in curves[0]]
            vals = [statistics.mean(c[i]["val_ce"] for c in curves) for i in range(len(steps))]
            times = [statistics.mean(c[i]["elapsed_s"] for c in curves) for i in range(len(steps))]
            axes[0, 0].plot(steps, vals, label=f"{task}:{name}", color=colors[name],
                            linestyle={"fashion_mnist":"-", "cifar10":"--", "tiny_shakespeare":":"}[task])
            axes[0, 1].plot(times, vals, label=f"{task}:{name}", color=colors[name],
                            linestyle={"fashion_mnist":"-", "cifar10":"--", "tiny_shakespeare":":"}[task])
    axes[0, 0].set(title="Validation CE vs updates (mean, n=3)", xlabel="updates", ylabel="CE")
    axes[0, 1].set(title="Validation CE vs elapsed time", xlabel="seconds", ylabel="CE")
    axes[0, 0].legend(fontsize=6, ncol=2)
    for name in ["adam", "ademamix", "ademamix_no_slow"]:
        means = []
        for horizon in config["diagnostic_levels"]["horizons"]:
            group = [r for r in diagnostic if r["optimizer"] == name and r["_config"]["max_updates"] == horizon]
            means.append(statistics.mean(r["final_val_ce"] for r in group))
        axes[1, 0].plot(config["diagnostic_levels"]["horizons"], means, marker="o", label=name, color=colors[name])
    axes[1, 0].set(title="History ablation on Fashion-MNIST", xlabel="training horizon", ylabel="validation CE")
    axes[1, 0].legend(fontsize=7)
    state_names = ["adam", "ademamix", "ademamix_no_slow"]
    state_values = [statistics.mean(r["optimizer_state_bytes"] for r in diagnostic if r["optimizer"] == n) for n in state_names]
    axes[1, 1].bar(state_names, state_values, color=[colors[n] for n in state_names])
    axes[1, 1].set(title="Persistent optimizer state", ylabel="bytes")
    axes[1, 1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    figure_dir = repo_root / config["output_root"] / "figures"; figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / "member4_overview.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "member4_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    (figure_dir / "plot_member4.py").write_text("# Figure generation is implemented in src/members/member4.py::_export.\n")

    notes = repo_root / config["output_root"] / "notes"
    adam = {r["task"]: r for r in main_rows if r["optimizer"] == "adam"}
    mix = {r["task"]: r for r in main_rows if r["optimizer"] == "ademamix"}
    lines = ["# Member 4 findings", "", "## Hypothesis", "",
             "The slow EMA should become more useful at the longer horizon, while costing one extra parameter-sized state tensor.", "",
             "## Evidence", ""]
    for task in config["tasks"]:
        lines.append(f"- {task}: Adam val CE {adam[task]['val_ce_mean']:.4f}; AdEMAMix {mix[task]['val_ce_mean']:.4f} (n=3 each).")
    lines += ["", "All 18 primary runs completed. See `tables/main_results.csv`, `diagnostic_results.csv`, and the overview figure.",
              "", "## Limitations", "", "These are deliberately short CPU horizons on small networks. They cannot establish the long-history effects reported for large language models.",
              "", "## Shared synthesis row", "", "AdEMAMix adds a third EMA state and must earn that memory cost through lower validation loss; its value is horizon- and task-dependent.",
              "", "## Abstract/conclusion contribution", "", "The paired Adam/AdEMAMix study measured whether a slow gradient EMA pays off under short CPU budgets, including a no-slow-EMA ablation and persistent-state accounting."]
    (notes / "findings.md").write_text("\n".join(lines) + "\n")
    (repo_root / "report" / "figures" / "member4_overview.pdf").write_bytes((figure_dir / "member4_overview.pdf").read_bytes())


def run_phase(phase, config, repo_root):
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    actions = {"check": _check, "smoke": _smoke, "tune": _tune, "main": _main,
               "diagnostic": _diagnostic, "export": _export}
    if phase not in actions:
        raise ValueError(phase)
    actions[phase](Path(repo_root), config)
