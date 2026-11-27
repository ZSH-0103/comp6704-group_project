"""Run six sequential shared-protocol CPU pilots and preserve a batch manifest."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from src.common.utils import write_json, hash_file


def main():
    root = Path(__file__).resolve().parents[2]
    shared = json.loads((root / "configs/shared_protocol.json").read_text())
    batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    evidence = root / "results/member1/notes/pilot" / batch
    evidence.mkdir(parents=True, exist_ok=False)
    files = list((root / "src/common").glob("*.py")) + [
        root / "src/members/member1_optimizers.py", Path(__file__),
        root / "scripts/run_shared.py", root / "configs/shared_protocol.json",
        root / "configs/member1_pilot_sgd.json", root / "configs/member1_pilot_dog.json"]
    for file in files:
        destination = evidence / "source" / file.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, destination)
    manifest = dict(batch=batch, runs=[], commands=[], pairing=[], shared_config=shared,
                    source_hashes={str(p.relative_to(root)): hash_file(p) for p in files},
                    note="SGD lr=0.01 is a pilot baseline, not tuned. Single seed 11; no test evaluation.")
    for task in shared["tasks"]:
        pair = []
        for name, cls in (("sgd", "SGD"), ("dog", "DoG")):
            command = [sys.executable, "scripts/run_shared.py", "--member", "1", "--task", task,
                       "--phase", "pilot", "--optimizer", name, "--factory",
                       f"src.members.member1_optimizers:{cls}", "--optimizer-config",
                       f"configs/member1_pilot_{name}.json"]
            print(f"Running {task} / {name}: {shared['pilot']['max_updates']} updates", flush=True)
            completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
            (evidence / f"{task}_{name}_stdout.txt").write_text(completed.stdout)
            (evidence / f"{task}_{name}_stderr.txt").write_text(completed.stderr)
            manifest["commands"].append(command)
            if completed.returncode:
                write_json(evidence / "manifest.json", manifest)
                raise RuntimeError(f"Pilot failed: {task}/{name}. See {evidence}")
            summary = json.loads(completed.stdout)
            assert summary["last_evaluated_step"] == shared["pilot"]["max_updates"]
            assert summary["test_ce"] is None
            manifest["runs"].append(summary)
            pair.append(summary)
            write_json(evidence / "manifest.json", manifest)
            print(f"Completed: val CE={summary['final_val_ce']:.5f}, {summary['elapsed_s']:.2f}s", flush=True)
        fields = ("initialization_hash", "split_hash", "model_hash", "data_order_hash")
        checks = {field: pair[0][field] == pair[1][field] for field in fields}
        assert all(checks.values()), f"Pairing mismatch: {checks}"
        manifest["pairing"].append(dict(task=task, checks=checks))
        write_json(evidence / "manifest.json", manifest)
    manifest["status"] = "passed"
    write_json(evidence / "manifest.json", manifest)
    write_json(root / "results/member1/notes/pilot_latest.json",
               {"manifest": str((evidence / "manifest.json").relative_to(root / "results/member1"))})
    print(f"Evidence: {evidence}")


if __name__ == "__main__":
    main()
