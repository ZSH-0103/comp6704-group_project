"""Package Member 1's implementation and saved experiment evidence for GitHub."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORIES = {"__pycache__", "build", "checkpoints", ".git", ".venv"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".zip", ".log", ".aux"}


def release_files():
    files = [ROOT / "src/__init__.py", ROOT / "src/members/__init__.py"]
    files += list((ROOT / "src/common").glob("*.py"))
    files += list((ROOT / "src/members").glob("member1*.py"))
    files += list((ROOT / "configs").glob("member1*.json"))
    files += [ROOT / "configs/shared_protocol.json", ROOT / "src/common/README.md"]
    files += list((ROOT / "data/manifests").glob("*.json"))
    for directory in ("results/member1", "report/generated"):
        files += [p for p in (ROOT / directory).rglob("*") if p.is_file()]
    for figure in ("member1_convergence", "member1_mechanism", "member1_overview"):
        files += list((ROOT / "report/figures").glob(figure + ".*"))
    for name in ("run_member.py", "run_shared.py", "verify_bundle.py",
                 "build_member1_report_assets.py", "package_member1.py"):
        files.append(ROOT / "scripts" / name)
    for name in ("FROZEN_SETTINGS_MEMBER1.md", "EXPERIMENT_CONTRACT.md",
                 "TEAM_SHARED_SETUP_FINAL.md", "MEMBER1_GITHUB_README.md"):
        files.append(ROOT / "docs" / name)
    return sorted(set(files))


def finalize(folder):
    inventory = {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "BUNDLE_MANIFEST.json"
    }
    (folder / "BUNDLE_MANIFEST.json").write_text(json.dumps(inventory, indent=2) + "\n")
    archive = folder.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for relative in [*inventory, "BUNDLE_MANIFEST.json"]:
            bundle.write(folder / relative, folder.name + "/" + relative)
        if bundle.testzip() is not None:
            raise ValueError("ZIP integrity verification failed")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/member1_sgd_dog_github")
    args = parser.parse_args()
    folder = args.output.resolve()
    if folder.exists() or folder.with_suffix(".zip").exists():
        parser.error("Choose a new output path to preserve the existing release")
    folder.mkdir(parents=True)
    for source in release_files():
        relative = source.relative_to(ROOT)
        if (source.is_symlink() or source.name.startswith((".env", ".DS_Store"))
                or set(relative.parts) & EXCLUDED_DIRECTORIES
                or source.suffix in EXCLUDED_SUFFIXES):
            continue
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = folder / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    shutil.copyfile(ROOT / "docs/MEMBER1_GITHUB_README.md", folder / "README.md")
    lock = ROOT / "shared_team_setup_v1.1/dependencies/member1-tested-environment.txt"
    if not lock.is_file():
        lock = ROOT / "requirements-lock.txt"
    shutil.copyfile(lock, folder / "requirements-lock.txt")
    (folder / "requirements.txt").write_text(
        "torch\ntorchvision\nnumpy\nmatplotlib\ndog-optimizer==1.0.3\n"
    )
    (folder / ".gitignore").write_text(
        ".venv/\n__pycache__/\n*.py[cod]\n.DS_Store\n.env\n.env.*\n*.pem\n"
        "data/raw/\ncheckpoints/\n*.pt\n*.pth\nbuild/\ndist/\n*.zip\n"
    )
    archive = finalize(folder)
    print(json.dumps({"folder": str(folder), "zip": str(archive),
                      "zip_bytes": archive.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
