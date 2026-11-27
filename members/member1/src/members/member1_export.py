"""Strict formal aggregation and plots; reads only the recorded main batch."""
import csv
import json
import os
from pathlib import Path
import shutil
import statistics

os.environ.setdefault("MPLCONFIGDIR","/tmp/comp6704-member1-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.utils import write_json, hash_file


def stats(values):
    values=[x for x in values if x is not None]
    return (statistics.mean(values) if values else None,
            statistics.stdev(values) if len(values)>1 else None)


def aggregate_results(runs,tuning,shared):
    expected={(t,o,s) for t in shared["tasks"] for o in ("sgd","dog") for s in shared["main_seeds"]}
    keys=[(r["task"],r["optimizer"],r["seed"]) for r in runs]
    if len(keys)!=len(set(keys)) or set(keys)!=expected or any(r["phase"]!="main" for r in runs):
        raise ValueError("Expected exactly 18 unique main runs; no tune/smoke mixing")
    main,cost=[],[]
    for task in shared["tasks"]:
        for opt in ("sgd","dog"):
            group=[r for r in runs if r["task"]==task and r["optimizer"]==opt]
            done=[r for r in group if r["status"]=="completed"]
            metric="test_ce" if task=="tiny_shakespeare" else "test_accuracy_pct"
            if any(r[metric] is None for r in done): raise ValueError("Completed main run lacks final test result")
            row=dict(task=task,optimizer=opt,n_expected=3,n_completed=len(done))
            for field,label in (("final_train_ce","train_ce"),("final_val_ce","val_ce")):
                row[label+"_mean"],row[label+"_sd"]=stats([r[field] for r in done])
            row["test_metric_name"]="CE_nats_per_character" if task=="tiny_shakespeare" else "accuracy_pct"
            row["test_metric_mean"],row["test_metric_sd"]=stats([r[metric] for r in done])
            row["elapsed_s_mean"],row["elapsed_s_sd"]=stats([r["elapsed_s"] for r in done])
            main.append(row)
            reached=[r for r in group if r["target_reached"]]
            trials=[r for r in tuning["runs"] if r["task"]==task and r["optimizer"]==opt] if opt=="sgd" else []
            cost.append(dict(task=task,optimizer=opt,n_expected=3,n_completed=len(done),n_target_reached=len(reached),
                             target_val_ce=shared["tasks"][task]["target_val_ce"],
                             target_steps_mean_reached=stats([r["target_step"] for r in reached])[0],
                             target_time_s_mean_reached=stats([r["target_time_s"] for r in reached])[0],
                             optimizer_state_bytes=max(r["optimizer_state_bytes"] for r in group),
                             tuning_trials=len(trials),tuning_elapsed_s=sum(r["elapsed_s"] for r in trials)))
    return main,cost


def export_results(repo_root):
    root=Path(repo_root);output=root/"results/member1"
    pointer=output/"notes/main_latest.json"
    if not pointer.exists(): raise ValueError("No main batch exists")
    path=output/json.loads(pointer.read_text())["manifest"]
    m=json.loads(path.read_text())
    if m["status"]!="completed": raise ValueError("Main batch inventory is incomplete")
    tuning=json.loads((root/m["member"]["tuning_evidence"]).read_text())
    main,cost=aggregate_results(m["runs"],tuning,m["shared"])
    audit=[]
    traces={}
    for r in m["runs"]:
        directory=output/"raw"/r["run_id"]
        config=json.loads((directory/"config.json").read_text())
        summary=json.loads((directory/"summary.json").read_text())
        if summary!=r: raise ValueError("Batch and raw summary disagree")
        if config["phase"]!="main" or config["formal_batch"]!=m["batch"] or not config["evaluate_test"]:
            raise ValueError("Run is outside this formal batch")
        if config["source_hashes"]!=m["source_hashes"] or config["optimizer_hyperparameters"]!=(m["member"]["hyperparameters"]["sgd"][r["task"]] if r["optimizer"]=="sgd" else m["member"]["hyperparameters"]["dog"]):
            raise ValueError("Run settings differ from frozen batch")
        if config["split_hash"]!=m["shared"]["tasks"][r["task"]]["approved_split_hash"]:
            raise ValueError("Run split differs from approved split")
        tr=[json.loads(l) for l in (directory/"metrics.jsonl").read_text().splitlines()]
        tr=[x for x in tr if "val_ce" in x];traces[r["run_id"]]=tr
        if r["status"]=="completed" and tr[-1]["step"]!=m["shared"]["tasks"][r["task"]]["max_updates"]:
            raise ValueError("Completed run did not finish its budget")
        crossing=next((x for x in tr if x["val_ce"]<=r["target_val_ce"]),None)
        if r["target_step"]!=(None if crossing is None else crossing["step"]):
            raise ValueError("Target attainment disagrees with logs")
        audit.append(dict(run_id=r["run_id"],status=r["status"],target_reached=r["target_reached"]))
    for pair in m["pairing"]:
        if all(r["status"]=="completed" for r in m["runs"] if r["task"]==pair["task"] and r["seed"]==pair["seed"]) and not all(pair["checks"].values()):
            raise ValueError("Completed paired runs do not match")
    def csv_out(name,rows):
        for dest in (path.parent/name,output/"tables"/name):
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csv_out("main_results.csv",main);csv_out("cost_stability.csv",cost)
    csv_out("main_run_inventory.csv",[dict(task=r["task"],optimizer=r["optimizer"],seed=r["seed"],status=r["status"],run_id=r["run_id"],test_ce=r["test_ce"],test_accuracy_pct=r["test_accuracy_pct"],test_elapsed_s=r["test_elapsed_s"],target_reached=r["target_reached"]) for r in m["runs"]])
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,3,figsize=(13,7));colors={"sgd":"#2566a3","dog":"#de7029"}
    titles={"fashion_mnist":"Fashion-MNIST / MLP","cifar10":"CIFAR-10 / CNN","tiny_shakespeare":"Tiny Shakespeare / GRU"}
    for column,(task,title) in enumerate(titles.items()):
        for opt in ("sgd","dog"):
            group=[r for r in m["runs"] if r["task"]==task and r["optimizer"]==opt and r["status"]=="completed"]
            if not group: continue
            steps=np.array([x["step"] for x in traces[group[0]["run_id"]]])
            curves=np.array([[x["val_ce"] for x in traces[r["run_id"]]] for r in group])
            mean=curves.mean(axis=0);sd=curves.std(axis=0,ddof=1) if len(group)>1 else np.zeros_like(mean)
            for row in (0,1):
                xs=steps if row==0 else np.mean([[x["elapsed_s"] for x in traces[r["run_id"]]] for r in group],axis=0)
                axes[row,column].plot(xs,mean,color=colors[opt],label=f"{opt.upper() if opt=='sgd' else 'DoG'} (n={len(group)})")
                axes[row,column].fill_between(xs,mean-sd,mean+sd,color=colors[opt],alpha=.15)
            # Raw validation trajectories remain in the logs; main figure uses
            # the frozen primary averaged-parameter evaluation only.
        axes[0,column].set_title(title,fontweight="bold")
        for row in (0,1):
            axes[row,column].axhline(m["shared"]["tasks"][task]["target_val_ce"],color="#555",linestyle=":",linewidth=1)
            axes[row,column].grid(alpha=.17);axes[row,column].legend(frameon=False)
            axes[row,column].set_xlabel("Parameter updates" if row==0 else "Training + scheduled evaluation time (s)")
    axes[0,0].set_ylabel("Validation CE (averaged weights)");axes[1,0].set_ylabel("Validation CE (averaged weights)")
    fig.suptitle("Formal SGD versus DoG: three independent seeds",x=.07,ha="left",fontweight="bold",fontsize=16)
    fig.text(.07,.91,"Frozen shared-v1.1 · CPU, one thread · polynomial averaging γ=8 · final-budget checkpoint · seeds 1, 2, 3",fontsize=9.5)
    fig.text(.07,.035,"Lines: mean; bands: ±1 sample SD, not confidence intervals. Dotted lines: frozen target validation CE. CE: nats/image or nats/character.\n"
             "Time curves use mean elapsed time at common evaluation steps; test evaluation is excluded. Tuning cost is reported separately.",fontsize=8.5)
    fig.subplots_adjust(top=.83,bottom=.15,left=.07,right=.985,hspace=.25,wspace=.25)
    for ext in ("png","pdf"):
        dest=output/"figures"/f"member1_overview.{ext}";dest.parent.mkdir(exist_ok=True)
        fig.savefig(dest,dpi=170,facecolor="white");shutil.copyfile(dest,path.parent/dest.name)
    plt.close(fig)
    (root/"report/figures").mkdir(exist_ok=True)
    shutil.copyfile(output/"figures/member1_overview.pdf",root/"report/figures/member1_overview.pdf")
    shutil.copyfile(Path(__file__),path.parent/Path(__file__).name)
    write_json(path.parent/"export_audit.json",dict(status="passed",expected_runs=18,n_completed=sum(r["status"]=="completed" for r in m["runs"]),n_rows=len(main),runs=audit,export_source_sha256=hash_file(__file__)))
    lines=["# Member 1 formal results","",f"Batch: `{m['batch']}`. Three formal seeds 1/2/3; frozen settings; final averaged-model test evaluation.","",
           "| Task | Optimizer | Test metric | Mean ± sample SD | Completed | Validation CE |",
           "| --- | --- | --- | --- | --- | --- |"]
    def fmt(mean,sd): return "unavailable" if mean is None else f"{mean:.5f} ± {sd:.5f}" if sd is not None else f"{mean:.5f} (SD unavailable)"
    for r in main:
        lines.append(f"| {r['task']} | {r['optimizer']} | {r['test_metric_name']} | {fmt(r['test_metric_mean'],r['test_metric_sd'])} | {r['n_completed']}/3 | {fmt(r['val_ce_mean'],r['val_ce_sd'])} |")
    lines += ["","## Target attainment (conditional means)","","| Task | Optimizer | Target CE | Reached | Mean updates among reached | Mean seconds among reached |","| --- | --- | ---: | --- | ---: | ---: |"]
    for r in cost:
        lines.append(f"| {r['task']} | {r['optimizer']} | {r['target_val_ce']} | {r['n_target_reached']}/3 | {r['target_steps_mean_reached']} | {r['target_time_s_mean_reached']} |")
    lines += ["","Unreached runs retain null target times/steps and are counted above. Failed runs remain in main_run_inventory.csv; means use completed runs only. Training CE uses the fixed training subset. Test CE is nats/character for text; image test accuracy is percent. Mean ± sample SD is descriptive, not a significance test.","",
              "The mechanism diagnostic is separate: see quadratic_findings.md and tables/diagnostic_results.csv. The main figure contains formal task curves only; the quadratic plots remain companion figures."]
    report="\n".join(lines)+"\n";(path.parent/"results.md").write_text(report)
    (output/"notes/main_results.md").write_text(report)
    findings=[report,"## Hypothesis and interpretation","",
              "The study asks whether global DoG can reduce external learning-rate search while achieving comparable task quality to SGD selected from a declared five-candidate grid. Formal results use independent seeds 1/2/3; tuning seed 11 is excluded. No test result was used to change settings.",""]
    for task in m["shared"]["tasks"]:
        s=next(r for r in main if r["task"]==task and r["optimizer"]=="sgd")
        d=next(r for r in main if r["task"]==task and r["optimizer"]=="dog")
        if s["test_metric_mean"] is not None and d["test_metric_mean"] is not None:
            delta=d["test_metric_mean"]-s["test_metric_mean"]
            findings.append(f"- {task}: DoG minus SGD mean test metric = {delta:+.5f} {s['test_metric_name']}. Higher is better for image accuracy; lower is better for text CE. Report SD and all three seeds alongside this difference; this is not a significance test.")
    findings += ["","## Costs, failures and mechanism","",
                 f"Completed {sum(r['status']=='completed' for r in m['runs'])}/18 formal runs. SGD tuning trial cost is recorded in cost_stability.csv and retained separately from formal training time. DoG used fixed defaults with zero search trials. Target-time means are conditional on reaching the threshold; the reached counts must accompany them.",
                 "The quadratic experiment is separate evidence about conditioning and scaling; see quadratic_findings.md. It uses exact gradients and raw iterates, whereas the neural study uses minibatches and averaged evaluation parameters. Its theoretical implications cannot be transferred automatically.",
                 "","## Limitations","",
                 "Only three formal seeds; small CPU models and subsets; fixed learning rates for SGD; five-candidate search with accepted upper-bound winners on Fashion/text; final-budget evaluation instead of best-checkpoint selection; no broad claim of optimizer optimality. Training CE is measured on a fixed subset, validation is complete, and time-to-target has 100-update evaluation resolution. Absolute timing comparisons are within this Mac only.",
                 "","## Shared-table handoff","",
                 "- tab:methods: plain SGD and independently implemented ordinary global DoG; DoG keeps an initial parameter snapshot and scalar accumulator, not per-coordinate preconditioning. Both updates have O(d) work; DoG's persistent initial snapshot is O(d).",
                 f"- tab:protocol: frozen settings in docs/FROZEN_SETTINGS_MEMBER1.md and configs/member1.json; SGD rates {m['member']['hyperparameters']['sgd']}; DoG {m['member']['hyperparameters']['dog']}; averaging gamma=8, final-budget checkpoint.",
                 "- tab:diagnostic-plan: independent quadratic conditioning/scale sweeps; settings, measurements and limitations in quadratic_findings.md.",
                 "- tab:hardware: use environment.json from the main raw directories; unavailable fields stay null. Version and thread settings are measured, not inferred.",
                 "- tab:synthesis: use the task-specific mean ± SD above and completion/attainment counts; do not combine accuracy and text CE into a single unnormalized score.",
                 "- tab:contributions: Member 1 independently implemented SGD/DoG, fixed-gradient checks, quadratic diagnostics, shared CPU pipeline, tuning, formal-run orchestration and exports. Report integration/presentation work remains to be completed.",
                 "","## Abstract/conclusion contribution","",
                 "The frozen three-task CPU comparison separates formal quality across three seeds from validation search cost; the main and cost tables quantify DoG against a five-candidate SGD baseline. Interpret any task-specific advantage with the observed SD and the study's limited model/data scope.",""]
    (output/"notes/findings.md").write_text("\n".join(findings))
    (path.parent/"findings.md").write_text("\n".join(findings))
    print(report)
    return main,cost
