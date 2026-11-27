"""Six paired CPU smoke runs on real data; no test evaluation or main results."""
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import torch

from src.common.data import load_task
from src.common.models import make_model
from src.common.train import train_one
from src.common.utils import hash_file, write_json
from .member1_optimizers import SGD, DoG


def run_smoke(config, repo_root):
    smoke = json.loads((repo_root / "configs/member1_smoke.json").read_text())
    shared = config["shared"]
    torch.set_num_threads(smoke["threads"])
    torch.use_deterministic_algorithms(True)
    batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    output = repo_root / config["output_root"]
    evidence = output / "notes/smoke" / batch
    evidence.mkdir(parents=True, exist_ok=False)
    files = list((repo_root / "src/common").glob("*.py")) + [Path(__file__), Path(__file__).with_name("member1_optimizers.py")]
    sources = {str(p.relative_to(repo_root)): hash_file(p) for p in files}
    for p in files:
        target = evidence / "source" / p.relative_to(repo_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    write_json(evidence / "configuration.json", dict(smoke=smoke, shared=shared, member=config, source_hashes=sources))
    runs, pairs = [], []
    for task in config["tasks"]:
        resolved = dict(shared["tasks"][task], **smoke, repo_root=str(repo_root), task=task,
                        split_seed=shared["split_seed"])
        data = load_task(resolved, smoke["seed"])
        resolved["vocab_size"] = data.metadata["vocab_size"]
        task_runs = []
        for name in ("sgd", "dog"):
            model, metadata = make_model(resolved, smoke["seed"])
            optimizer = SGD(model.parameters(), lr=smoke["sgd_lr"]) if name == "sgd" else DoG(model.parameters(), **smoke["dog"])
            settings = dict(resolved, **metadata, output_root=str(output), member=1, phase="smoke", optimizer=name,
                            optimizer_hyperparameters={"lr": smoke["sgd_lr"]} if name == "sgd" else smoke["dog"],
                            shared_status=shared["status"], member_status=config["status"], schedule="constant SGD; adaptive DoG",
                            source_hashes=sources)
            r = train_one(model, optimizer, data, settings)
            runs.append(r)
            task_runs.append(r)
            write_json(evidence / "runs.json", runs)
            print(f"{task} / {name}: {r['status']}; val CE={r['final_val_ce']}; {r['elapsed_s']:.3f}s", flush=True)
        fields = ("initialization_hash", "data_order_hash", "split_hash", "model_hash")
        checks = {key: task_runs[0][key] == task_runs[1][key] for key in fields}
        checks["all_updates_evaluated"] = all(r["last_evaluated_step"] == smoke["max_updates"] for r in task_runs)
        checks["weights_changed"] = all(r["final_parameter_hash"] != r["initialization_hash"] for r in task_runs)
        checks["test_not_evaluated"] = all(r["test_ce"] is None and r["test_accuracy_pct"] is None for r in task_runs)
        passed = all(checks.values()) and all(r["status"] == "completed" for r in task_runs)
        pairs.append(dict(task=task, checks=checks, passed=passed))
    result = dict(batch=batch, runs=runs, pairing_checks=pairs, status="passed" if all(p["passed"] for p in pairs) else "failed")
    write_json(evidence / "manifest.json", result)
    write_json(output / "notes/smoke_latest.json", {"manifest": str((evidence / "manifest.json").relative_to(output))})
    lines = ["# Member 1 real-data CPU smoke tests", "", f"Status: **{result['status']}**. Batch: `{batch}`.", "",
             "20 updates; batch size 16; train subset 256; validation subset 128; train CE on a fixed 64-example/window subset.",
             "One CPU thread. Polynomial averaging gamma=8 for both; raw validation CE also logged. No test evaluation.", "",
             "| Task | Optimizer | Status | Train CE | Validation CE | Validation accuracy (%) | Seconds |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for r in runs:
        def fmt(value):
            return "—" if value is None else f"{value:.5f}"
        lines.append(f"| {r['task']} | {r['optimizer']} | {r['status']} | {fmt(r['final_train_ce'])} | {fmt(r['final_val_ce'])} | {fmt(r['final_val_accuracy_pct'])} | {r['elapsed_s']:.3f} |")
    lines += ["", "Pairing checks:", ""] + [f"- {p['task']}: {p['checks']}" for p in pairs]
    lines += ["", "These are small pipeline checks, not tuned optimizer comparisons or main results. Image CE is nats/example; text CE is nats/character.",
              "Train/validation subset sizes count images for image tasks and character windows for text. Download and preparation time is excluded.",
              "Shared/member training protocols remain DRAFT. See src/common/README.md for design choices and remaining work."]
    report = "\n".join(lines) + "\n"
    (evidence / "report.md").write_text(report)
    (output / "notes/smoke_results.md").write_text(report)
    if result["status"] != "passed":
        raise RuntimeError("Smoke validation failed; inspect retained evidence")
    return result
