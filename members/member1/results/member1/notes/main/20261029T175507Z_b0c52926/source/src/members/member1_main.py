"""Guarded 18-run formal experiment batch with persistent resume bookkeeping."""
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


def run_main(root):
    root=Path(root)
    member=json.loads((root/"configs/member1.json").read_text())
    shared=json.loads((root/member["shared_protocol"]).read_text())
    if member["status"]!="FROZEN" or shared["status"]!="FROZEN":
        raise ValueError("Shared and Member 1 settings must be frozen")
    if set(shared["main_seeds"])!={1,2,3}:
        raise ValueError("Expected formal seeds 1/2/3")
    output=root/member["output_root"]
    sources=list((root/"src/common").glob("*.py"))+[Path(__file__),root/"src/members/member1_optimizers.py",
            root/"configs/member1.json",root/"configs/shared_protocol.json"]
    hashes={str(p.relative_to(root)):hash_file(p) for p in sources}
    latest=output/"notes/main_latest.json"
    if latest.exists():
        path=output/json.loads(latest.read_text())["manifest"]
        manifest=json.loads(path.read_text())
        if manifest["source_hashes"]!=hashes:
            raise ValueError("Existing formal batch has different frozen sources/settings; preserve it and explicitly plan a new batch")
        if manifest["status"]=="completed":
            print(f"All formal runs already recorded. Use export; no test evaluations repeated. {path}")
            return manifest
        evidence=path.parent
    else:
        batch=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")+uuid.uuid4().hex[:8]
        evidence=output/"notes/main"/batch;evidence.mkdir(parents=True,exist_ok=False)
        path=evidence/"manifest.json"
        manifest=dict(batch=batch,status="running",shared=shared,member=member,source_hashes=hashes,runs=[],pairing=[])
        for p in sources:
            dest=evidence/"source"/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        write_json(path,manifest)
        write_json(latest,{"manifest":str(path.relative_to(output))})
    torch.set_num_threads(shared["threads"])
    torch.use_deterministic_algorithms(shared["deterministic_algorithms"])
    for task in member["tasks"]:
        base=resolve_task_config(root,task,"main",1,member_status=member["status"],seed=1)
        data=load_task(base,1)
        for seed in shared["main_seeds"]:
            for name in member["optimizers"]:
                existing=[r for r in manifest["runs"] if (r["task"],r["seed"],r["optimizer"])==(task,seed,name)]
                if existing: continue
                assert all(hash_file(root/p)==value for p,value in hashes.items()),"Frozen source changed during batch"
                # Recover an interrupted runner's already-written summary, without
                # re-evaluating its test set. Refuse ambiguous/incomplete evidence.
                recovery=[]
                for file in (output/"raw").glob("m1_main_*/config.json"):
                    c=json.loads(file.read_text())
                    if c.get("formal_batch")==manifest["batch"] and (c["task"],c["seed"],c["optimizer"])==(task,seed,name):
                        recovery.append(file.parent)
                if recovery:
                    if len(recovery)!=1 or not (recovery[0]/"summary.json").exists():
                        raise RuntimeError("Interrupted raw run requires inspection; refusing automatic repeat of possible test evaluation")
                    summary=json.loads((recovery[0]/"summary.json").read_text())
                else:
                    cfg=resolve_task_config(root,task,"main",1,member_status=member["status"],seed=seed)
                    cfg["vocab_size"]=data.metadata["vocab_size"]
                    model,info=make_model(cfg,seed)
                    kwargs=member["hyperparameters"]["sgd"][task] if name=="sgd" else member["hyperparameters"]["dog"]
                    opt=SGD(model.parameters(),**kwargs) if name=="sgd" else DoG(model.parameters(),**kwargs)
                    cfg.update(info,optimizer=name,optimizer_hyperparameters=kwargs,formal_batch=manifest["batch"],
                               member_config_hash=hash_json(member),output_root=str(output),source_hashes=hashes,
                               timeout_s=1800,schedule="constant SGD; adaptive DoG")
                    print(f"START main {task}/{name}/seed{seed}: {cfg['max_updates']} updates",flush=True)
                    summary=train_one(model,opt,data,cfg)
                manifest["runs"].append(summary);write_json(path,manifest)
                print(f"DONE {task}/{name}/seed{seed}: {summary['status']}; test CE={summary['test_ce']}; {summary['elapsed_s']:.1f}s",flush=True)
            pair=[r for r in manifest["runs"] if r["task"]==task and r["seed"]==seed]
            checks={k:pair[0][k]==pair[1][k] for k in ("initialization_hash","data_order_hash","split_hash","model_hash")}
            # A failed run can have a shorter order hash; keep it, don't hide it.
            manifest["pairing"]=[p for p in manifest["pairing"] if (p["task"],p["seed"])!=(task,seed)]
            manifest["pairing"].append(dict(task=task,seed=seed,checks=checks))
            write_json(path,manifest)
    assert len(manifest["runs"])==18
    manifest["status"]="completed"
    manifest["n_completed"]=sum(r["status"]=="completed" for r in manifest["runs"])
    write_json(path,manifest)
    print(f"Formal batch recorded: {manifest['n_completed']}/18 completed. {path}",flush=True)
    return manifest
