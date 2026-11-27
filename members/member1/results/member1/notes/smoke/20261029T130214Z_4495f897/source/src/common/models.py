"""Shared small MLP/CNN/character-GRU architectures, deterministic initialization."""
import torch
from torch import nn

from .utils import hash_json, hash_state


class CharacterGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, layers, batch_first=True, dropout=0.)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens):
        features, _ = self.gru(self.embedding(tokens))  # fresh zero hidden state per window
        return self.head(features)


def make_model(task_config, seed):
    cfg = task_config
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if cfg["model"] == "mlp":
            widths = cfg["widths"]
            modules = [nn.Flatten()]
            for i, (a, b) in enumerate(zip(widths, widths[1:])):
                modules.append(nn.Linear(a, b))
                if i < len(widths) - 2:
                    modules.append(nn.ReLU())
            model = nn.Sequential(*modules)
            spec = dict(kind="mlp", widths=widths, activation="relu")
        elif cfg["model"] == "small_cnn":
            c1, c2 = cfg["conv_channels"]
            kernel, padding = cfg["kernel_size"], cfg["conv_padding"]
            if kernel != 3 or padding != 1:
                raise ValueError("Shared CNN currently specifies 3x3 convolutions, padding 1")
            model = nn.Sequential(nn.Conv2d(3, c1, kernel, padding=padding), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(c1, c2, kernel, padding=padding), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Flatten(), nn.Linear(c2 * 8 * 8, 10))
            spec = dict(kind="cnn", channels=[3, c1, c2], kernel=kernel, padding=padding,
                        pooling="2x2_after_each_conv", input=[3, 32, 32], classifier=[c2 * 8 * 8, 10])
        elif cfg["model"] == "char_gru":
            model = CharacterGRU(cfg["vocab_size"], cfg["embedding_dim"], cfg["hidden_size"], cfg["layers"])
            spec = {k: cfg[k] for k in ("model", "vocab_size", "embedding_dim", "hidden_size", "layers")}
            spec["hidden_state"] = "reset_per_window"
        else:
            raise ValueError(f"Unknown model: {cfg['model']}")
        model = model.cpu().float()
    spec.update(dtype="float32", dropout=0., batch_norm=False, initializer="PyTorch defaults, seeded")
    return model, dict(model_hash=hash_json(spec), initialization_hash=hash_state(model),
                       model_spec=spec, parameter_count=sum(p.numel() for p in model.parameters()))
