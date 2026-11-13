"""PyTorch networks, demo trainer and checkpoint storage helpers.

Only training and loading legacy ``.pt`` checkpoints need this module (and
PyTorch). Inference runs on numpy: see numpy_runtime.py.
"""

import csv
from pathlib import Path

import torch
from torch import nn

from language_detector import BUCKETS, encode

def batch(texts: list[str], **encoding_options) -> tuple[torch.Tensor, torch.Tensor]:
    ids, offsets = [], []
    for text in texts:
        offsets.append(len(ids))
        ids.extend(encode(text, **encoding_options))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


class LanguageModel(nn.Module):
    """Original linear architecture (arch=linear). Do not change."""

    def __init__(self, languages: int, buckets: int = BUCKETS, embedding_dim: int = 32):
        super().__init__()
        self.embedding = nn.EmbeddingBag(buckets, embedding_dim, mode="mean")
        self.classifier = nn.Linear(embedding_dim, languages)

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embedding(ids, offsets))


class LanguageModelMLP(nn.Module):
    """Experimental hidden-layer architecture (arch=mlp)."""

    def __init__(
        self,
        languages: int,
        buckets: int = BUCKETS,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.embedding = nn.EmbeddingBag(buckets, embedding_dim, mode="mean")
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, languages)

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids, offsets)
        x = self.activation(self.hidden(x))
        return self.classifier(x)


def build_model(
    languages: int, buckets: int = BUCKETS, embedding_dim: int = 32, hidden_dim: int = 128, arch: str = "linear"
) -> nn.Module:
    if arch == "mlp":
        return LanguageModelMLP(languages, buckets, embedding_dim, hidden_dim)
    if arch == "linear":
        return LanguageModel(languages, buckets, embedding_dim)
    raise ValueError(f"Unknown architecture: {arch}")


def train(data: Path, output: Path, epochs: int) -> None:
    torch.manual_seed(42)
    torch.set_num_threads(1)
    with data.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not {"text", "language"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain text and language columns.")
        rows = list(reader)
    if not rows or any(not row.get("language", "").strip() for row in rows):
        raise ValueError("CSV must contain texts with nonempty language labels.")
    languages = sorted({row["language"].strip() for row in rows})
    if len(languages) < 2:
        raise ValueError("Examples of at least two languages are required.")
    # Validate all text before starting training.
    for row in rows:
        encode(row.get("text") or "")
    model = LanguageModel(len(languages))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()
    targets = torch.tensor([languages.index(row["language"].strip()) for row in rows])
    for epoch in range(epochs):
        for indices in torch.randperm(len(rows)).split(64):
            inputs = batch([rows[i]["text"] for i in indices.tolist()])
            optimizer.zero_grad()
            loss = criterion(model(*inputs), targets[indices])
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch + 1}/{epochs}, last batch loss: {loss.item():.4f}"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "languages": languages}, output)
    print(f"Saved model: {output}")


class LanguageModelEnsemble(nn.Module):
    """Weighted logits from compatible models, with an optional agreement gate."""

    def __init__(self, models, weights, temperatures, require_agreement=False):
        super().__init__()
        if len(models) != len(weights) or len(models) != len(temperatures) or not models:
            raise ValueError("Invalid ensemble components")
        if any(w <= 0 for w in weights) or any(t <= 0 for t in temperatures):
            raise ValueError("Weights and temperatures must be positive")
        self.members = nn.ModuleList(models)
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self.temperatures = temperatures
        self.require_agreement = require_agreement

    def forward_with_agreement(self, ids, offsets):
        outputs = [model(ids, offsets) for model in self.members]
        logits = sum(w * value / t for w, value, t in zip(self.weights, outputs, self.temperatures))
        predictions = torch.stack([value.argmax(1) for value in outputs])
        agree = (predictions == predictions[0]).all(0)
        if not self.require_agreement:
            agree = torch.ones_like(agree)
        return logits, agree

    def forward(self, ids, offsets):
        return self.forward_with_agreement(ids, offsets)[0]


WEIGHT_STORAGES = ("fp32", "fp16", "int8_rowwise")
ROW_SCALE_SUFFIX = ".__row_scale"


def quantize_rowwise_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row int8: weight ~= values * scale, values in [-127, 127]."""
    if weight.dim() != 2 or not weight.is_floating_point():
        raise ValueError("Row-wise int8 needs a 2-D floating-point tensor")
    weight = weight.detach().float()
    scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
    # An all-zero row has scale 0; its values are 0 either way.
    values = torch.where(scale > 0, weight / torch.where(scale > 0, scale, 1.0), 0.0)
    return values.round().clamp(-127, 127).to(torch.int8), scale


def compress_state_dict(state_dict: dict, storage: str, int8_tensors: tuple[str, ...] = ("embedding.weight",)) -> dict:
    """Return the stored form of a float32 state dict (see decompress_state_dict)."""
    if storage not in WEIGHT_STORAGES:
        raise ValueError(f"Unknown weight storage: {storage!r}")
    for name, tensor in state_dict.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"Expected float32 weights, {name} is {tensor.dtype}")
    if storage == "fp32":
        return dict(state_dict)
    if storage == "fp16":
        return {name: tensor.to(torch.float16) for name, tensor in state_dict.items()}
    missing = [name for name in int8_tensors if name not in state_dict]
    if missing:
        raise ValueError(f"Tensors to quantize are missing: {missing}")
    stored = {}
    for name, tensor in state_dict.items():
        if name in int8_tensors:
            stored[name], stored[name + ROW_SCALE_SUFFIX] = quantize_rowwise_int8(tensor)
        else:
            stored[name] = tensor.clone()
    return stored


def decompress_state_dict(state_dict: dict, storage: str) -> dict:
    """Float32 state dict from a stored one; refuses dtypes the storage does not allow."""
    if storage == "fp32":
        return state_dict
    if storage == "fp16":
        wrong = [name for name, tensor in state_dict.items() if tensor.dtype != torch.float16]
        if wrong:
            raise ValueError(f"fp16 storage with non-float16 tensors: {wrong}")
        return {name: tensor.float() for name, tensor in state_dict.items()}
    if storage == "int8_rowwise":
        restored = {}
        scales = {name for name in state_dict if name.endswith(ROW_SCALE_SUFFIX)}
        unused_scales = set(scales)
        for name, tensor in state_dict.items():
            if name in scales:
                continue
            if tensor.dtype == torch.int8:
                scale = state_dict.get(name + ROW_SCALE_SUFFIX)
                if tensor.dim() != 2 or scale is None or scale.dtype != torch.float32 or scale.shape != (tensor.shape[0], 1):
                    raise ValueError(f"int8 tensor {name} needs 2 dimensions and a float32 row scale of shape (rows, 1)")
                unused_scales.discard(name + ROW_SCALE_SUFFIX)
                restored[name] = tensor.float() * scale
            elif tensor.dtype == torch.float32:
                restored[name] = tensor
            else:
                raise ValueError(f"int8_rowwise storage with {tensor.dtype} tensor {name}")
        if unused_scales:
            raise ValueError(f"Row scales without an int8 tensor: {sorted(unused_scales)}")
        return restored
    raise ValueError(f"Unknown weight storage: {storage!r}")


def model_from_checkpoint(checkpoint):
    config = checkpoint.get("config", {})
    model = build_model(len(checkpoint["languages"]), config.get("buckets", BUCKETS), config.get("embedding_dim", 32), config.get("hidden_dim", 128), config.get("arch", "linear"))
    model.load_state_dict(decompress_state_dict(checkpoint["state_dict"], config.get("weight_storage", "fp32")))
    ensemble = checkpoint.get("ensemble")
    if ensemble:
        companions = ensemble["companions"]
        feature_keys = ("buckets", "ngrams", "max_length", "word_features", "preprocessing", "word_repeat")
        defaults = {"buckets": BUCKETS, "ngrams": (1, 2, 3), "max_length": 2000, "word_features": False, "preprocessing": "none", "word_repeat": 4}
        for other in companions:
            if other["languages"] != checkpoint["languages"]:
                raise ValueError("Ensemble members must use the same language order")
            for key in feature_keys:
                left = config.get(key, defaults[key])
                right = other.get("config", {}).get(key, defaults[key])
                if key == "ngrams":
                    left, right = tuple(left), tuple(right)
                if left != right:
                    raise ValueError("Incompatible ensemble feature encoding: " + key)
        model = LanguageModelEnsemble([model] + [model_from_checkpoint(c) for c in companions], ensemble["weights"], ensemble["temperatures"], ensemble.get("require_agreement", False))
    return model
