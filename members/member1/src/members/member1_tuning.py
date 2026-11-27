"""Validation-only SGD search and matched-budget DoG comparison."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

import torch

from src.common.data import load_task
from src.common.models import make_model
from src.common.protocol import resolve_task_config
from src.common.train import train_one
from src.common.utils import hash_file, hash_json, write_json
from .member1_optimizers import SGD, DoG


def select_sgd(runs):
    eligible = [r for r in runs if r["status"] == "completed" and r["last_evaluated_step"] == r["budget"]]
    if not eligible:
        raise RuntimeError("All SGD tuning candidates failed")
    return min(eligible, key=lambda r: (r["final_val_ce"], r["lr"]))


def run_tuning(repo_root):
    root = Path(repo_root)
    shared = json.loads((root / "configs/shared_protocol.json").read_text())
    tuning = json.loads((root / "configs/member1_tuning.json").read_text())
    assert tuning["seed"] == shared["tuning_seed"]
    assert all(len(v) <= shared["tuning"]["max_candidates_per_task_per_optimizer"] for v in tuning["sgd_learning_rates"].values())
    torch.set_num_threads(shared["threads"])
    torch.use_deterministic_algorithms(shared["deterministic_algorithms"])
    batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    output = root / "results/member1"
    evidence = output / "notes/tuning" / batch
    evidence.mkdir(parents=True, exist_ok=False)
    files = list((root/"src/common").glob("*.py")) + [Path(__file__),root/"src/members/member1_optimizers.py",
             root/"configs/member1_tuning.json",root/"configs/shared_protocol.json"]
    hashes = {str(p.relative_to(root)): hash_file(p) for p in files}
    for source in files:
        target=evidence/"source"/source.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    manifest = dict(batch=batch,status="running",tuning=tuning,shared_config=shared,source_hashes=hashes,runs=[],selected={},pairing=[])
    def save():
        write_json(evidence/"manifest.json",manifest)
        rows=[]
        for r in manifest["runs"]:
            if r["optimizer"] != "sgd": continue
            rows.append(dict(trial_id=r["run_id"],task=r["task"],optimizer="sgd",seed=r["seed"],
                             hyperparameters_json=json.dumps({"lr":r["lr"]}),
                             validation_ce=r["final_val_ce"] if r["status"]=="completed" else None,
                             status=r["status"],elapsed_s=r["elapsed_s"],
                             selected=manifest["selected"].get(r["task"],{}).get("run_id")==r["run_id"]))
        if rows:
            with (evidence/"tuning_log.csv").open("w",newline="") as file:
                w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save()
    for task,budget in tuning["budgets"].items():
        # Resolve the agreed shared data/evaluation choices, then record the
        # explicit authorized tuning budget without changing the shared protocol.
        base=resolve_task_config(root,task,"pilot",1)
        base.update(budget,phase="tune",seed=tuning["seed"],evaluate_test=False,
                    tuning_config_hash=hash_json(tuning),budget_source="configs/member1_tuning.json",
                    source_hashes=hashes,output_root=str(output),schedule="constant SGD; adaptive DoG")
        data=load_task(base,base["seed"])
        base["vocab_size"]=data.metadata["vocab_size"]
        sgd_runs=[]
        for name,kwargs in [("sgd",{"lr":lr}) for lr in tuning["sgd_learning_rates"][task]] + [("dog",tuning["dog"])]:
            print(f"START {task}/{name} {kwargs}, {budget['max_updates']} updates",flush=True)
            model,info=make_model(base,base["seed"])
            opt=SGD(model.parameters(),**kwargs) if name=="sgd" else DoG(model.parameters(),**kwargs)
            cfg=dict(base,**info,optimizer=name,optimizer_hyperparameters=kwargs)
            r=train_one(model,opt,data,cfg)
            r.update(lr=kwargs.get("lr"),budget=budget["max_updates"])
            manifest["runs"].append(r)
            if name=="sgd": sgd_runs.append(r)
            else: dog=r
            save()
            print(f"DONE {task}/{name} {kwargs}: {r['status']}, val CE={r['final_val_ce']}, {r['elapsed_s']:.1f}s",flush=True)
        winner=select_sgd(sgd_runs)
        candidates=tuning["sgd_learning_rates"][task]
        manifest["selected"][task]=dict(run_id=winner["run_id"],lr=winner["lr"],validation_ce=winner["final_val_ce"],
                                          boundary_winner=winner["lr"] in (min(candidates),max(candidates)))
        fields=("initialization_hash","data_order_hash","model_hash","split_hash")
        checks={key:winner[key]==dog[key] for key in fields}
        assert all(checks.values()) and dog["status"]=="completed",checks
        manifest["pairing"].append(dict(task=task,checks=checks))
        save()
        print(f"SELECTED {task}: SGD lr={winner['lr']}",flush=True)
    manifest["status"]="completed"
    save()
    shutil.copyfile(evidence/"tuning_log.csv",output/"notes/tuning_log.csv")
    write_json(output/"notes/tuning_latest.json",{"manifest":str((evidence/"manifest.json").relative_to(output))})
    write_json(output/"notes/selected_hyperparameters_proposal.json",dict(status="AWAITING_TEAM_FREEZE",batch=batch,
               sgd={task:{"lr":v["lr"]} for task,v in manifest["selected"].items()},dog=tuning["dog"],
               budgets=tuning["budgets"],boundary_winners=[t for t,v in manifest["selected"].items() if v["boundary_winner"]]))
    print(f"Evidence: {evidence}",flush=True)
    return manifest


if __name__=="__main__":
    run_tuning(Path(__file__).resolve().parents[2])
