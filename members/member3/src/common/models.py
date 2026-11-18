"""Shared small models with deterministic initialization."""
import hashlib
import torch
from torch import nn


class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class CharGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        y, _ = self.gru(self.embedding(x))
        return self.output(y)


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_model(task, task_config, seed, metadata):
    torch.manual_seed(int(seed))
    if task == "fashion_mnist":
        model = FashionMLP()
    elif task == "cifar10":
        model = SmallCNN()
    elif task == "tiny_shakespeare":
        model = CharGRU(metadata["vocab_size"], task_config["embedding_dim"], task_config["hidden_size"])
    else:
        raise ValueError(task)
    return model, model_hash(model)
