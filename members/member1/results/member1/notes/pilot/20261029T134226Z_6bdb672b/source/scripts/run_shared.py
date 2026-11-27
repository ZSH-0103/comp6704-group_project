"""Run one member's optimizer on the common pilot/smoke pipeline.

Factories accept (parameters, **kwargs). Default PyTorch SGD is an integration
example only. Members must supply their independent optimizer implementations.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.data import load_task
from src.common.models import make_model
from src.common.protocol import resolve_task_config
from src.common.train import train_one
from src.common.utils import hash_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--task", choices=["fashion_mnist", "cifar10", "tiny_shakespeare"], required=True)
    parser.add_argument("--phase", choices=["smoke", "pilot"], default="pilot")
    parser.add_argument("--optimizer", default="torch_sgd_example")
    parser.add_argument("--factory", default="torch.optim:SGD", help="importable module:callable")
    parser.add_argument("--optimizer-config", type=Path, help="JSON file containing constructor kwargs")
    parser.add_argument("--evaluation", choices=["raw", "averaged"], help="Explicit member evaluation policy")
    parser.add_argument("--data-root", type=Path, help="Existing raw datasets; no copying required")
    args = parser.parse_args()
    policy = None
    if args.evaluation:
        policy = dict(evaluation_point=args.evaluation,
                      averaging={"kind": "polynomial", "gamma": 8} if args.evaluation == "averaged" else None,
                      evaluate_raw_also=args.evaluation == "averaged")
    cfg = resolve_task_config(ROOT, args.task, args.phase, args.member, evaluation_policy=policy, data_root=args.data_root)
    import torch
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(cfg["deterministic_algorithms"])
    data = load_task(cfg, cfg["seed"])
    cfg["vocab_size"] = data.metadata["vocab_size"]
    model, metadata = make_model(cfg, cfg["seed"])
    kwargs = json.loads(args.optimizer_config.read_text()) if args.optimizer_config else {"lr": 0.01}
    module, name = args.factory.split(":")
    implementation = importlib.import_module(module)
    optimizer = getattr(implementation, name)(model.parameters(), **kwargs)
    if callable(getattr(optimizer, "eval", None)) and cfg.get("averaging"):
        parser.error("For optimizers with eval/train hooks, select --evaluation raw and validate intended evaluation parameters")
    cfg.update(metadata, optimizer=args.optimizer, optimizer_hyperparameters=kwargs,
               optimizer_factory=args.factory, output_root=str(ROOT / f"results/member{args.member}"),
               schedule="optimizer-defined; no external scheduler",
               source_hashes={str(p.relative_to(ROOT)): hash_file(p) for p in (ROOT / "src/common").glob("*.py")})
    if getattr(implementation, "__file__", None):
        cfg["optimizer_source_hash"] = hash_file(implementation.__file__)
    summary = train_one(model, optimizer, data, cfg)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
