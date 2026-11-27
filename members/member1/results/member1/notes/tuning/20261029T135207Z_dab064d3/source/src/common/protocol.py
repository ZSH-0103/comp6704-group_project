"""Resolve the agreed shared protocol without silently filling main budgets."""
from copy import deepcopy
import json
from pathlib import Path

from .utils import hash_json


def resolve_task_config(repo_root, task, phase, member, member_status="DRAFT", seed=None,
                        evaluation_policy=None, data_root=None):
    root = Path(repo_root).resolve()
    shared = json.loads((root / "configs/shared_protocol.json").read_text())
    if phase not in ("smoke", "pilot", "tune", "main"):
        raise ValueError("Expected smoke, pilot, tune or main")
    if phase == "main" and (shared["status"] != "FROZEN" or member_status != "FROZEN"):
        raise ValueError("Formal main runs need FROZEN shared and member protocols")
    cfg = deepcopy(shared["tasks"][task])
    if phase in ("smoke", "pilot"):
        cfg.update(shared[phase])
    elif cfg["max_updates"] is None or cfg["eval_interval"] is None:
        raise ValueError("Set formal update budget and evaluation interval after the pilot")
    policy = deepcopy(evaluation_policy or shared["evaluation_policies"][str(member)])
    if "evaluation_point" not in policy:
        raise ValueError("Member must declare raw/averaged evaluation policy before running")
    if policy["evaluation_point"] not in ("raw", "averaged"):
        raise ValueError("Unknown evaluation point")
    if policy["evaluation_point"] == "averaged" and not policy.get("averaging"):
        raise ValueError("Averaged evaluation requires an averaging specification")
    cfg.update(policy)
    cfg.setdefault("eval_batch_size", shared["evaluation"]["eval_batch_size_text" if task == "tiny_shakespeare" else "eval_batch_size_images"])
    cfg.update(repo_root=str(root), task=task, phase=phase, member=member,
               shared_status=shared["status"], member_status=member_status,
               protocol_version=shared["protocol_version"], protocol_hash=hash_json(shared),
               split_seed=shared["split_seed"], threads=shared["threads"],
               deterministic_algorithms=shared["deterministic_algorithms"], weight_decay=shared["weight_decay"],
               seed=seed if seed is not None else shared["tuning_seed"],
               evaluate_test=phase == "main", checkpoint_selection=shared["evaluation"]["checkpoint"])
    if data_root is not None:
        cfg["data_root"] = str(Path(data_root).resolve())
    return cfg
