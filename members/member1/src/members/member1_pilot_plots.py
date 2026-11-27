"""Plot measured pilot losses and export progress statistics; no retraining."""
import csv
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/comp6704-member1-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "results/member1"
    manifest_path = output / json.loads((output / "notes/pilot_latest.json").read_text())["manifest"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "passed" and len(manifest["runs"]) == 6
    figures = manifest_path.parent / "figures"
    figures.mkdir(exist_ok=True)
    names = {"fashion_mnist": "Fashion-MNIST / MLP", "cifar10": "CIFAR-10 / CNN", "tiny_shakespeare": "Tiny Shakespeare / GRU"}
    colors = {"sgd": "#2767a0", "dog": "#df702a"}
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(2, 3, figsize=(13,7), sharex=True)
    timed, time_axes = plt.subplots(1, 3, figsize=(13,4.3))
    rows = []
    for column, (task, title) in enumerate(names.items()):
        for opt in ("sgd", "dog"):
            run = next(r for r in manifest["runs"] if r["task"]==task and r["optimizer"]==opt)
            raw = output / "raw" / run["run_id"]
            config = json.loads((raw/"config.json").read_text())
            data = [json.loads(line) for line in (raw/"metrics.jsonl").read_text().splitlines()]
            assert [r["step"] for r in data]==[0,100,200,300,400,500]
            steps = [r["step"] for r in data]
            for row, metric in enumerate(("train_ce","val_ce")):
                axes[row,column].plot(steps,[r[metric] for r in data],color=colors[opt],marker="o",markersize=4,label=opt.upper() if opt=="sgd" else "DoG")
            axes[1,column].plot(steps,[r["raw_val_ce"] for r in data],color=colors[opt],linestyle=":",alpha=.7)
            time_axes[column].plot([r["elapsed_s"] for r in data],[r["val_ce"] for r in data],color=colors[opt],marker="o",markersize=4,label=opt.upper() if opt=="sgd" else "DoG")
            v400,v500 = data[-2]["val_ce"],data[-1]["val_ce"]
            rows.append(dict(task=task,optimizer=opt,run_id=run["run_id"],seed=run["seed"],
                             train_count=config["train_count"],validation_count=config["validation_count"],
                             batch_size=config["batch_size"],updates=500,elapsed_s=run["elapsed_s"],
                             initial_val_ce=data[0]["val_ce"],final_train_ce=data[-1]["train_ce"],
                             val_ce_300=data[3]["val_ce"],val_ce_400=v400,final_val_ce=v500,
                             late_val_ce_drop=v400-v500,late_val_ce_drop_pct=100*(v400-v500)/v400,
                             final_raw_val_ce=data[-1]["raw_val_ce"],final_val_accuracy_pct=data[-1]["val_accuracy_pct"]))
        axes[0,column].set_title(title,fontweight="bold")
        axes[1,column].set_xlabel("Parameter updates")
        for ax in axes[:,column]:
            ax.grid(alpha=.18);ax.legend(frameon=False);ax.set_xlim(0,500)
        time_axes[column].set(title=title,xlabel="Elapsed training + evaluation time (s)")
        time_axes[column].grid(alpha=.18);time_axes[column].legend(frameon=False)
    axes[0,0].set_ylabel("Training CE (fixed subset)")
    axes[1,0].set_ylabel("Validation CE (complete split)")
    time_axes[0].set_ylabel("Validation CE (complete split)")
    fig.suptitle("500-update CPU pilots: SGD and DoG",fontsize=16,fontweight="bold",x=.07,ha="left")
    fig.text(.07,.91,"Seed 11 · one CPU thread · identical starting weights and batch order within each task",fontsize=10)
    fig.text(.07,.035,"Solid lines: polynomial-averaged weights (γ=8); dotted lines: raw validation weights. Six evaluation points per curve.\n"
             "SGD lr=0.01; DoG reps_rel=10⁻⁶, ε=10⁻⁸. One run per method/task; no uncertainty bands. CE: nats/image or nats/character.",fontsize=8.5)
    fig.subplots_adjust(top=.84,bottom=.14,left=.07,right=.98,hspace=.2,wspace=.24)
    timed.suptitle("Pilot validation loss versus measured runtime",fontsize=15,fontweight="bold")
    timed.text(.07,.025,"Averaged weights; single-seed CPU measurements include initial and scheduled evaluations. Timings exclude data preparation.",fontsize=8.5)
    timed.subplots_adjust(top=.82,bottom=.18,left=.07,right=.98,wspace=.25)
    for figure, name in ((fig,"pilot_loss_curves"),(timed,"pilot_loss_vs_time")):
        for ext in ("png","pdf"):
            path = figures / f"{name}.{ext}"
            figure.savefig(path,dpi=170,facecolor="white")
            (output/"figures").mkdir(exist_ok=True)
            shutil.copyfile(path,output/"figures"/path.name)
        plt.close(figure)
    with (manifest_path.parent/"pilot_summary.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    shutil.copyfile(manifest_path.parent/"pilot_summary.csv",output/"notes/pilot_summary.csv")
    shutil.copyfile(Path(__file__),manifest_path.parent/Path(__file__).name)
    print(json.dumps(rows,indent=2))


if __name__=="__main__":
    main()
