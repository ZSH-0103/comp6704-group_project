"""Export quadratic plots from measured logs, without rerunning experiments."""
import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/comp6704-member1-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .member1_diagnostics import aggregate, write_csv


COLORS = {"sgd": "#2563a6", "dog": "#dc6b28"}
NAMES = {"sgd": "Tuned SGD", "dog": "DoG"}
FLOOR = 1e-16


def export_plots(root, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    cfg, runs = manifest["config"], manifest["runs"]
    output = manifest_path.parent / "figures"
    output.mkdir(exist_ok=True)
    traces = {}
    for run in runs:
        raw = root / "raw" / run["run_id"]
        traces[run["run_id"]] = [json.loads(line) for line in (raw / "metrics.jsonl").read_text().splitlines()]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    for sweep, levels in (("conditioning", cfg["condition_numbers"]), ("scale", cfg["objective_scales"])):
        fig, axes = plt.subplots(2, len(levels), figsize=(13, 7), sharex=True, squeeze=False)
        for column, level in enumerate(levels):
            k = level if sweep == "conditioning" else cfg["reference_condition_number"]
            a = cfg["reference_scale"] if sweep == "conditioning" else level
            L = a * k
            top, bottom = axes[:, column]
            for opt in ("sgd", "dog"):
                group = [r for r in runs if r["kappa"] == k and r["scale"] == a and r["optimizer"] == opt]
                complete = [r for r in group if r["status"] == "completed"]
                # Preserve failed trajectories visibly, but do not invent tail values.
                for r in group:
                    tr = traces[r["run_id"]]
                    top.plot([v["step"] for v in tr], np.maximum([v["relative_gap"] for v in tr], FLOOR),
                             color=COLORS[opt], alpha=0.18, linewidth=0.7)
                if not complete:
                    continue
                data = [[v["relative_gap"] for v in traces[r["run_id"]]] for r in complete]
                data = np.asarray(data)
                mean = data.mean(axis=0)
                sd = data.std(axis=0, ddof=1) if len(data) > 1 else np.zeros_like(mean)
                steps = np.arange(len(mean))
                top.plot(steps, np.maximum(mean, FLOOR), color=COLORS[opt], linewidth=2,
                         label=f"{NAMES[opt]} ({len(complete)}/{len(group)})")
                top.fill_between(steps, np.maximum(mean - sd, FLOOR), np.maximum(mean + sd, FLOOR),
                                 color=COLORS[opt], alpha=0.13)
                etas = np.asarray([[v["learning_rate"] for v in traces[r["run_id"]][1:]] for r in complete])
                factor = L if sweep == "conditioning" else 1.
                bottom.plot(steps[1:], factor * etas.mean(axis=0), color=COLORS[opt], linewidth=2)
                if len(etas) > 1:
                    sd_eta = etas.std(axis=0, ddof=1)
                    bottom.fill_between(steps[1:], np.maximum(factor * (etas.mean(axis=0) - sd_eta), 1e-20),
                                        factor * (etas.mean(axis=0) + sd_eta), color=COLORS[opt], alpha=0.13)
            top.axhline(cfg["target_relative_gap"], color="#555555", linestyle=":", linewidth=1)
            top.set(title=f"Condition number {k:g}  |  Scale {a:g}", yscale="log", ylim=(FLOOR, 3))
            bottom.axhline(2 if sweep == "conditioning" else 2/L, color="#555555", linestyle=":", linewidth=1)
            bottom.set(yscale="log", xlabel="Parameter updates")
            if column == 0:
                top.set_ylabel("Objective gap / initial gap")
                bottom.set_ylabel("Step size × largest eigenvalue" if sweep == "conditioning" else "Step size")
            top.legend(loc="upper right", fontsize=8, frameon=False)
            for ax in (top, bottom):
                ax.grid(alpha=0.16)
                ax.set_xlim(0, cfg["max_updates"])
        title = "Ill-conditioning: can one scalar step size handle unequal curvature?" if sweep == "conditioning" else "Objective scaling: same landscape shape, different gradient magnitudes"
        fig.suptitle(title, fontsize=15, fontweight="bold", x=0.06, ha="left")
        fig.text(0.06, 0.905, f"Exact gradients · {cfg['dimension']} dimensions · {len(cfg['seeds'])} paired starting points · SGD tuned separately per condition", fontsize=10)
        fig.text(0.06, 0.035, "Lines: arithmetic mean; shading: ±1 sample SD across completed runs. Thin loss lines: individual runs.\n"
                 "Dotted lines: relative-gap target (top), constant-step stability boundary ηL = 2 (bottom). "
                 "Gaps below 10⁻¹⁶ are displayed at 10⁻¹⁶.", fontsize=8.5, color="#444444")
        fig.subplots_adjust(left=0.075, right=0.985, top=0.85, bottom=0.14, hspace=0.24, wspace=0.27)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"quadratic_{sweep}.{extension}", dpi=170)
        plt.close(fig)
    # Separate tuning evidence: losses for failed trials stay blank, never zero.
    fig, ax = plt.subplots(figsize=(9, 5))
    for condition in dict.fromkeys(r["condition"] for r in runs):
        trials = [t for t in manifest["tuning"] if t["condition"] == condition and t["status"] == "completed"]
        ax.plot([t["lr_times_L"] for t in trials], [max(t["final_relative_gap"], FLOOR) for t in trials],
                marker="o", markersize=4, label=condition)
        best = next(t for t in trials if t["selected"])
        ax.scatter(best["lr_times_L"], max(best["final_relative_gap"], FLOOR), marker="*", s=150, edgecolor="black", zorder=5)
    ax.set(xlabel="SGD learning rate × largest eigenvalue (ηL)", ylabel="Final relative gap, tuning seed 11",
           yscale="log", title="SGD tuning: stars mark selected settings")
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8)
    fig.text(0.12, 0.025, "Scale-sweep curves can overlap. Diverged trials omitted here; all trials remain in tuning_log.csv. Display floor: 10⁻¹⁶.", fontsize=8)
    fig.subplots_adjust(bottom=0.17)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"quadratic_tuning.{extension}", dpi=170)
    plt.close(fig)
    rows = aggregate(runs, cfg, manifest["config_hash"])
    write_csv(manifest_path.parent / "diagnostic_results.csv", rows)
    # Stable convenience copies; immutable evidence remains in the batch folder.
    (root / "figures").mkdir(exist_ok=True)
    (root / "tables").mkdir(exist_ok=True)
    for file in output.iterdir():
        shutil.copyfile(file, root / "figures" / file.name)
    shutil.copyfile(manifest_path.parent / "diagnostic_results.csv", root / "tables/diagnostic_results.csv")
    shutil.copyfile(manifest_path.parent / "tuning_log.csv", root / "notes/quadratic_tuning_log.csv")
    costs = []
    for condition in manifest["selected_sgd_lr"]:
        for opt in ("sgd", "dog"):
            group = [r for r in runs if r["condition"] == condition and r["optimizer"] == opt]
            reached = [r for r in group if r["target_reached"]]
            trials = [t for t in manifest["tuning"] if t["condition"] == condition] if opt == "sgd" else []
            costs.append(dict(condition=condition, optimizer=opt, n_expected=len(cfg["seeds"]),
                              n_completed=sum(r["status"] == "completed" for r in group), n_target_reached=len(reached),
                              target_relative_gap=cfg["target_relative_gap"],
                              target_steps_mean_reached=float(np.mean([r["target_step"] for r in reached])) if reached else None,
                              target_time_s_mean_reached=float(np.mean([r["target_time_s"] for r in reached])) if reached else None,
                              optimizer_tensor_state_bytes=group[0]["optimizer_state_bytes"], tuning_trials=len(trials),
                              tuning_failures=sum(t["status"] != "completed" for t in trials),
                              tuning_elapsed_s=sum(t["elapsed_s"] for t in trials)))
    write_csv(manifest_path.parent / "cost_stability.csv", costs)
    write_csv(root / "tables/quadratic_cost_stability.csv", costs)
    (manifest_path.parent / "export_source.py").write_bytes(Path(__file__).read_bytes())
    table = ["# Quadratic results", "", f"Batch: `{manifest['batch']}`; protocol hash: `{manifest['config_hash']}`.", "",
             f"Exact gradients; raw iterates; {cfg['max_updates']} updates; seeds {cfg['seeds']} vary starting direction. Target: relative gap <= {cfg['target_relative_gap']:g}.", "",
             "| Condition | Method | SGD lr | Final relative gap, mean ± sample SD | Reached | Mean target updates (reached only) |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for condition in dict.fromkeys(r["condition"] for r in runs):
        for opt in ("sgd", "dog"):
            g = [r for r in runs if r["condition"] == condition and r["optimizer"] == opt]
            done = [r for r in g if r["status"] == "completed"]
            values = [r["final_relative_gap"] for r in done]
            reached = [r["target_step"] for r in g if r["target_reached"]]
            gap = f"{np.mean(values):.4g} ± {np.std(values, ddof=1):.3g}" if len(values) > 1 else "unavailable"
            target = f"{np.mean(reached):.1f}" if reached else "unreached"
            lr = f"{g[0]['lr']:.6g}" if opt == "sgd" else "adaptive"
            table.append(f"| {condition} | {NAMES[opt]} | {lr} | {gap} | {len(reached)}/{len(g)} | {target} |")
    failures = [r for r in runs if r["status"] != "completed"]
    tuning_failed = [r for r in manifest["tuning"] if r["status"] != "completed"]
    table += ["", f"Comparison failures: {len(failures)}/{len(runs)}. Tuning failures: {len(tuning_failed)}/{len(manifest['tuning'])}.",
              f"Tuning cost: {len(manifest['tuning'])} SGD trials; {sum(t['elapsed_s'] for t in manifest['tuning']):.3f} measured update/evaluation seconds. DoG defaults were not searched.",
              "Timing excludes problem setup and file writes; tiny CPU timings should not be used as a speed ranking.",
              "", "See quadratic_findings.md for interpretation and limitations. No neural-network/main results are included."]
    (manifest_path.parent / "results.md").write_text("\n".join(table) + "\n")
    shutil.copyfile(manifest_path.parent / "results.md", root / "notes/quadratic_results.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2] / "results/member1"
    path = args.manifest or root / json.loads((root / "notes/quadratic_latest.json").read_text())["manifest"]
    export_plots(root, path)
