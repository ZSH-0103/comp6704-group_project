"""Deterministic metadata and JSON helpers shared by all members."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import torch


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_state(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def environment():
    def command(args):
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return dict(python=platform.python_version(), platform=platform.platform(),
                cpu=command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                ram_bytes=command(["sysctl", "-n", "hw.memsize"]), device="cpu",
                dependencies={name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy")},
                threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                git_commit=command(["git", "rev-parse", "HEAD"]))
