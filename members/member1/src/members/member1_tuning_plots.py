"""Plot recorded validation searches and matched-budget selected trajectories."""
import csv
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR","/tmp/comp6704-member1-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parents[2]
    output=root/"results/member1"
    p=output/json.loads((output/"notes/tuning_latest.json").read_text())["manifest"]
    m=json.loads(p.read_text())
    assert m["status"]=="completed"
    figures=p.parent/"figures";figures.mkdir(exist_ok=True)
    names={"fashion_mnist":"Fashion-MNIST / MLP","cifar10":"CIFAR-10 / CNN","tiny_shakespeare":"Tiny Shakespeare / GRU"}
    colors={"sgd":"#2566a3","dog":"#de7029"}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    comparison,axes=plt.subplots(2,3,figsize=(13,7))
    search,saxes=plt.subplots(1,3,figsize=(13,4.8))
    timed,taxes=plt.subplots(1,3,figsize=(13,4.8))
    rows=[]
    def trace(r):
        records=[json.loads(l) for l in (output/"raw"/r["run_id"]/"metrics.jsonl").read_text().splitlines()]
        return [x for x in records if "val_ce" in x]
    for column,(task,title) in enumerate(names.items()):
        winner=m["selected"][task]
        sgd=next(r for r in m["runs"] if r["run_id"]==winner["run_id"])
        dog=next(r for r in m["runs"] if r["task"]==task and r["optimizer"]=="dog")
        budget=m["tuning"]["budgets"][task]["max_updates"]
        for r in [x for x in m["runs"] if x["task"]==task and x["optimizer"]=="sgd"]:
            data=trace(r);selected=r["run_id"]==winner["run_id"]
            saxes[column].plot([x["step"] for x in data],[x["val_ce"] for x in data],
                               linewidth=2.8 if selected else 1.2,alpha=1 if selected else .65,
                               label=f"lr={r['lr']:g}"+(" (selected)" if selected else "")+(" FAILED" if r["status"]!="completed" else ""))
        for r in (sgd,dog):
            opt=r["optimizer"];data=trace(r);steps=[x["step"] for x in data]
            label=f"Tuned SGD (lr={winner['lr']:g})" if opt=="sgd" else "DoG (fixed defaults)"
            for row,key in enumerate(("train_ce","val_ce")):
                axes[row,column].plot(steps,[x[key] for x in data],color=colors[opt],label=label)
            axes[1,column].plot(steps,[x["raw_val_ce"] for x in data],color=colors[opt],linestyle=":",alpha=.6)
            taxes[column].plot([x["elapsed_s"] for x in data],[x["val_ce"] for x in data],color=colors[opt],label=label)
            late=next(x for x in data if x["step"]>=budget*.8)
            best=min(data,key=lambda x:x["val_ce"])
            rows.append(dict(task=task,optimizer=opt,run_id=r["run_id"],seed=r["seed"],updates=budget,
                             lr=winner["lr"] if opt=="sgd" else None,train_ce=data[-1]["train_ce"],
                             validation_ce=data[-1]["val_ce"],raw_validation_ce=data[-1]["raw_val_ce"],
                             validation_accuracy_pct=data[-1]["val_accuracy_pct"],elapsed_s=r["elapsed_s"],
                             minimum_observed_validation_ce=best["val_ce"],minimum_observed_step=best["step"],
                             final_20pct_validation_drop_pct=100*(late["val_ce"]-data[-1]["val_ce"])/late["val_ce"],
                             sgd_selected_at_search_boundary=winner["boundary_winner"] if opt=="sgd" else False,
                             sgd_search_trials=5 if opt=="sgd" else 0,
                             sgd_search_elapsed_s=sum(x["elapsed_s"] for x in m["runs"] if x["task"]==task and x["optimizer"]=="sgd") if opt=="sgd" else 0))
        axes[0,column].set_title(f"{title}\n{budget:,} updates",fontweight="bold")
        for ax in axes[:,column]:
            ax.grid(alpha=.17);ax.set_xlim(0,budget);ax.legend(frameon=False,fontsize=8)
        axes[1,column].set_xlabel("Parameter updates")
        saxes[column].set(title=title,xlabel="Parameter updates");saxes[column].legend(frameon=False,fontsize=8);saxes[column].grid(alpha=.17)
        taxes[column].set(title=title,xlabel="Elapsed training + evaluation time (s)");taxes[column].legend(frameon=False,fontsize=8);taxes[column].grid(alpha=.17)
    axes[0,0].set_ylabel("Training CE (fixed subset)");axes[1,0].set_ylabel("Validation CE (complete split)")
    saxes[0].set_ylabel("Averaged validation CE");taxes[0].set_ylabel("Averaged validation CE")
    comparison.suptitle("Validation-tuned SGD versus DoG at proposed formal budgets",fontsize=15,fontweight="bold",x=.07,ha="left")
    comparison.text(.07,.91,"Shared data/models · seed 11 · CPU, one thread · paired initialization and batch order · evaluation every 100 updates",fontsize=9.5)
    comparison.text(.07,.035,"Solid: polynomial-averaged weights (γ=8); dotted: raw validation weights. SGD selected by final averaged validation CE from 5 candidates/task.\n"
                    "DoG reps_rel=10⁻⁶, ε=10⁻⁸. Single tuning seed; selection favors SGD on this seed. No test evaluation or uncertainty estimates.",fontsize=8.5)
    comparison.subplots_adjust(top=.82,bottom=.14,left=.07,right=.985,hspace=.22,wspace=.25)
    search.suptitle("SGD validation search: five predeclared learning rates per task",fontsize=15,fontweight="bold")
    search.text(.07,.025,"Selection uses final-budget averaged validation loss; intermediate minima are not used for checkpoint or hyperparameter selection.",fontsize=8.5)
    search.subplots_adjust(top=.8,bottom=.17,left=.07,right=.985,wspace=.25)
    timed.suptitle("Selected SGD and DoG: validation loss versus measured runtime",fontsize=15,fontweight="bold")
    timed.text(.07,.025,"SGD curve time excludes its separate tuning cost; all 15 search trials and their elapsed times are retained in tuning_log.csv.",fontsize=8.5)
    timed.subplots_adjust(top=.8,bottom=.17,left=.07,right=.985,wspace=.25)
    for fig,name in ((comparison,"tuned_sgd_vs_dog"),(search,"sgd_validation_search"),(timed,"tuned_loss_vs_time")):
        for ext in ("png","pdf"):
            file=figures/f"{name}.{ext}";fig.savefig(file,dpi=170,facecolor="white")
            shutil.copyfile(file,output/"figures"/file.name)
        plt.close(fig)
    with (p.parent/"comparison.csv").open("w",newline="") as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    shutil.copyfile(p.parent/"comparison.csv",output/"notes/tuned_comparison.csv")
    shutil.copyfile(Path(__file__),p.parent/Path(__file__).name)
    print(json.dumps(rows,indent=2))


if __name__=="__main__":
    main()
