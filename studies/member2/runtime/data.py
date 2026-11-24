"""Deterministic CPU datasets for the shared optimizer study."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class TaskData:
    train: Dataset
    validation: Dataset
    test: Dataset
    split_hash: str
    metadata: dict


class CharacterDataset(Dataset):
    def __init__(self, encoded: torch.Tensor, sequence_length: int):
        self.encoded = encoded
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.encoded) - self.sequence_length)

    def __getitem__(self, index):
        end = index + self.sequence_length
        return self.encoded[index:end], self.encoded[index + 1:end + 1]


def _digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _save_manifest(repo_root: Path, task: str, payload: dict) -> str:
    digest = _digest(payload)
    output = repo_root / "data" / "manifests" / f"{task}_split.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and json.loads(output.read_text())['split_hash'] != digest:
        raise ValueError(f'Existing split manifest differs: {output}')
    output.write_text(json.dumps({"split_hash": digest, **payload}, indent=2, sort_keys=True) + "\n")
    return digest


def _fixed_subset(dataset, indices):
    return Subset(dataset, [int(i) for i in indices])


def load_task(task: str, config: dict, repo_root: Path) -> TaskData:
    """Load fixed splits. Dataset preparation time is outside experiment timing."""
    root = repo_root / "data" / "raw"
    root.mkdir(parents=True, exist_ok=True)
    split_seed = int(config["split_seed"])
    tc = config["tasks"][task]
    generator = torch.Generator().manual_seed(split_seed)

    if task == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        full = datasets.FashionMNIST(root, train=True, download=False, transform=transform)
        test_full = datasets.FashionMNIST(root, train=False, download=False, transform=transform)
        order = torch.randperm(len(full), generator=generator)
        n_train, n_val = tc["train_count"], tc["validation_count"]
        test_idx = torch.randperm(len(test_full), generator=generator)[:tc["test_count"]]
        payload = {"task": task, "seed": split_seed, "train": order[:n_train].tolist(),
                   "validation": order[n_train:n_train+n_val].tolist(), "test": test_idx.tolist()}
        split_hash = _save_manifest(repo_root, task, payload)
        return TaskData(_fixed_subset(full, order[:n_train]),
                        _fixed_subset(full, order[n_train:n_train+n_val]),
                        _fixed_subset(test_full, test_idx), split_hash,
                        {"classes": 10, "normalization": "mean=0.2860,std=0.3530"})

    if task == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        full = datasets.CIFAR10(root, train=True, download=False, transform=transform)
        test_full = datasets.CIFAR10(root, train=False, download=False, transform=transform)
        order = torch.randperm(len(full), generator=generator)
        n_train, n_val = tc["train_count"], tc["validation_count"]
        test_idx = torch.randperm(len(test_full), generator=generator)[:tc["test_count"]]
        payload = {"task": task, "seed": split_seed, "train": order[:n_train].tolist(),
                   "validation": order[n_train:n_train+n_val].tolist(), "test": test_idx.tolist()}
        split_hash = _save_manifest(repo_root, task, payload)
        return TaskData(_fixed_subset(full, order[:n_train]),
                        _fixed_subset(full, order[n_train:n_train+n_val]),
                        _fixed_subset(test_full, test_idx), split_hash,
                        {"classes": 10, "normalization": "standard CIFAR-10 channel statistics"})

    if task == "tiny_shakespeare":
        path = root / "tiny_shakespeare.txt"
        if not path.exists():
            raise FileNotFoundError(f'{path}: run scripts/prepare_data.py first')
        text = path.read_text(encoding="utf-8")
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        b1, b2 = tc["split_boundaries"]
        seq = tc["sequence_length"]
        payload = {"task": task, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "boundaries": [b1, b2], "sequence_length": seq, "vocab": "".join(chars)}
        split_hash = _save_manifest(repo_root, task, payload)
        return TaskData(CharacterDataset(encoded[:b1], seq), CharacterDataset(encoded[b1:b2], seq),
                        CharacterDataset(encoded[b2:], seq), split_hash,
                        {"vocab_size": len(chars), "vocab": "".join(chars), "source_sha256": payload["sha256"]})
    raise ValueError(f"Unknown task: {task}")
