"""Real datasets, saved disjoint splits, train-only vocabulary and lazy tests.

load_task(config, seed) accepts a resolved per-task config including repo_root.
seed is the run seed; split_seed alone controls shared data membership.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import urllib.request
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from .utils import hash_file, hash_json, write_json


TEXT_COMMIT = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
TEXT_URL = f"https://raw.githubusercontent.com/karpathy/char-rnn/{TEXT_COMMIT}/data/tinyshakespeare/input.txt"
FASHION_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"


def download_data(repo_root):
    """Download only on explicit preparation; torchvision verifies source MD5s."""
    raw = Path(repo_root) / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    # HTTPS mirror in the dataset authors' repository; filenames and checksum
    # verification remain torchvision's. Restore class metadata afterwards.
    mirrors = datasets.FashionMNIST.mirrors
    try:
        datasets.FashionMNIST.mirrors = [FASHION_URL]
        datasets.FashionMNIST(str(raw), train=True, download=True)
    finally:
        datasets.FashionMNIST.mirrors = mirrors
    datasets.CIFAR10(str(raw), train=True, download=True)
    path = raw / "tiny_shakespeare" / "input.txt"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".download")
        with urllib.request.urlopen(TEXT_URL, timeout=60) as response:
            temporary.write_bytes(response.read())
        temporary.replace(path)
    provenance = {
        "fashion_mnist": {"url": FASHION_URL, "source_md5": datasets.FashionMNIST.resources},
        "cifar10": {"url": datasets.CIFAR10.url, "source_md5": datasets.CIFAR10.tgz_md5},
        "tiny_shakespeare": {"url": TEXT_URL, "commit": TEXT_COMMIT, "sha256": hash_file(path)},
    }
    write_json(Path(repo_root) / "data/manifests/sources.json", provenance)
    print("Real datasets prepared in data/raw; provenance saved in data/manifests/sources.json")


class CharacterWindows(Dataset):
    """Windows stay within one already-split text segment; unknown token is 0."""
    def __init__(self, text, vocabulary, sequence_length, stride):
        self.tokens = torch.tensor([vocabulary.get(c, 0) for c in text], dtype=torch.long)
        self.length, self.stride = sequence_length, stride
        if sequence_length <= 0 or stride <= 0:
            raise ValueError("Window length and stride must be positive")

    def __len__(self):
        return max(0, (len(self.tokens) - self.length - 1) // self.stride + 1)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.stride
        return (self.tokens[start:start + self.length],
                self.tokens[start + 1:start + self.length + 1])


@dataclass
class TaskData:
    train: Dataset
    validation: Dataset
    train_eval: Dataset
    test_factory: Callable
    metadata: dict

    def test(self):
        """Only the final frozen evaluation should call this method."""
        return self.test_factory()


def load_task(config, seed):
    root, task = Path(config["repo_root"]), config["task"]
    raw = root / "data/raw"
    counts = [config[k] for k in ("train_count", "validation_count", "train_eval_count")]
    if any(not isinstance(n, int) or n <= 0 for n in counts) or counts[2] > counts[0]:
        raise ValueError("Specify positive subset counts and train_eval_count <= train_count")
    rng = np.random.default_rng(config["split_seed"])
    sources = {}
    vocab_size = None
    if task in ("fashion_mnist", "cifar10"):
        cls = datasets.FashionMNIST if task == "fashion_mnist" else datasets.CIFAR10
        transform = transforms.ToTensor()  # fixed [0,1], no augmentation/normalization
        base = cls(str(raw), train=True, download=False, transform=transform)
        if sum(counts[:2]) > len(base):
            raise ValueError("Requested image split exceeds official training split")
        indices = rng.permutation(len(base)).tolist()
        train_indices = indices[:counts[0]]
        val_indices = indices[counts[0]:sum(counts[:2])]
        train, validation = Subset(base, train_indices), Subset(base, val_indices)
        test_factory = lambda: cls(str(raw), train=False, download=False, transform=transform)
        folder = raw / ("FashionMNIST/raw" if task == "fashion_mnist" else "cifar-10-batches-py")
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix != ".gz":
                sources[str(path.relative_to(raw))] = hash_file(path)
        split = dict(train_indices=train_indices, validation_indices=val_indices,
                     test="official test split; not used for smoke/tuning",
                     transform="uint8_to_float32_0_1_no_augmentation")
    elif task == "tiny_shakespeare":
        path = raw / "tiny_shakespeare/input.txt"
        text = path.read_text(encoding="utf-8")
        sources[str(path.relative_to(raw))] = hash_file(path)
        f, v = config["train_fraction"], config["validation_fraction"]
        if not 0 < f < f + v < 1:
            raise ValueError("Text fractions must leave nonempty train/validation/test segments")
        b1, b2 = int(len(text) * f), int(len(text) * (f + v))
        # Fit vocabulary ONLY to the training segment; no held-out vocabulary leakage.
        vocabulary = {c: i + 1 for i, c in enumerate(sorted(set(text[:b1])))}
        vocab_size = len(vocabulary) + 1
        length, stride = config["sequence_length"], config["text_stride"]
        train_base = CharacterWindows(text[:b1], vocabulary, length, stride)
        val_base = CharacterWindows(text[b1:b2], vocabulary, length, stride)
        if counts[0] > len(train_base) or counts[1] > len(val_base):
            raise ValueError("Requested text subset exceeds split windows")
        train_indices = rng.permutation(len(train_base))[:counts[0]].tolist()
        val_indices = rng.permutation(len(val_base))[:counts[1]].tolist()
        train, validation = Subset(train_base, train_indices), Subset(val_base, val_indices)
        test_factory = lambda: CharacterWindows(text[b2:], vocabulary, length, stride)
        split = dict(train_indices=train_indices, validation_indices=val_indices,
                     index_units="window index local to each chronological split",
                     character_boundaries=[0, b1, b2, len(text)], vocabulary=vocabulary, unknown_token_id=0,
                     sequence_length=length, stride=stride, source_url=TEXT_URL)
    else:
        raise ValueError(f"Unknown task: {task}")
    split.update(task=task, split_seed=config["split_seed"], train_eval_indices=list(range(counts[2])),
                 dataset_sha256=sources, schema_version=1)
    split_hash = hash_json(split)
    path = root / "data/manifests" / f"{task}_{split_hash[:16]}.json"
    if path.exists():
        import json
        if hash_json(json.loads(path.read_text())) != split_hash:
            raise ValueError("Existing split manifest was modified")
    else:
        write_json(path, split)
    metadata = dict(task=task, split_hash=split_hash, dataset_hash=hash_json(sources),
                    manifest=str(path.relative_to(root)), vocab_size=vocab_size,
                    train_count=len(train), validation_count=len(validation),
                    train_eval_count=counts[2], train_ce_scope="fixed training subset, evaluation mode")
    return TaskData(train, validation, Subset(train, list(range(counts[2]))), test_factory, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the three real shared datasets")
    parser.add_argument("--download", action="store_true", required=True)
    parser.parse_args()
    download_data(Path(__file__).resolve().parents[2])
