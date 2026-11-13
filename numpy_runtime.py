"""Torch-free inference for hashed-feature language classifiers.

The networks are EmbeddingBag(mean) followed by a linear layer (arch=linear) or
one ReLU hidden layer and a linear layer (arch=mlp); an ensemble combines the
temperature-scaled logits of several such networks. This module runs them with
numpy only, from either:

* a ``.npz`` release file written by ``tools/export_numpy.py`` (no PyTorch
  needed at all), or
* the arrays of a PyTorch checkpoint (``from_checkpoint``), which the caller has
  already loaded with ``torch.load``.

Row-wise int8 and float16 embedding tables stay in their stored dtype in
memory; only the rows a text uses are converted to float32.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FORMAT = "windup-language-recog/npz"
FORMAT_VERSION = 1
WEIGHT_STORAGES = ("fp32", "fp16", "int8_rowwise")
ROW_SCALE_SUFFIX = ".__row_scale"
META_KEY = "__meta__"
BUCKETS = 8192  # default of language_detector.BUCKETS for old checkpoints
FEATURE_DEFAULTS = {
    "buckets": BUCKETS,
    "ngrams": (1, 2, 3),
    "max_length": 2000,
    "word_features": False,
    "preprocessing": "none",
    "word_repeat": 4,
}
_LAYERS = {
    "linear": ("classifier.weight", "classifier.bias"),
    "mlp": ("hidden.weight", "hidden.bias", "classifier.weight", "classifier.bias"),
}


def encoding_options(config: dict) -> dict:
    """The encode() keyword arguments a checkpoint config implies."""
    options = {key: config.get(key, default) for key, default in FEATURE_DEFAULTS.items()}
    options["ngrams"] = tuple(options["ngrams"])
    return options


class Embedding:
    """Embedding table with mean pooling over variable-length bags."""

    def __init__(self, weight: np.ndarray, scale: np.ndarray | None = None):
        if weight.ndim != 2:
            raise ValueError("Embedding weight must be 2-D")
        if weight.dtype == np.int8:
            if scale is None or scale.dtype != np.float32 or scale.shape != (weight.shape[0], 1):
                raise ValueError("int8 embedding needs a float32 row scale of shape (rows, 1)")
        elif scale is not None:
            raise ValueError("Only an int8 embedding takes a row scale")
        elif weight.dtype not in (np.float32, np.float16):
            raise ValueError(f"Unsupported embedding dtype {weight.dtype}")
        self.weight, self.scale = weight, scale

    @property
    def shape(self) -> tuple[int, int]:
        return self.weight.shape

    def rows(self, ids: np.ndarray) -> np.ndarray:
        """Float32 rows, bit-identical to dequantizing the whole table first."""
        rows = self.weight[ids]
        if self.scale is not None:
            return rows.astype(np.float32) * self.scale[ids]
        return rows.astype(np.float32, copy=False)

    def full(self) -> np.ndarray:
        return self.rows(np.arange(self.weight.shape[0]))

    def mean(self, ids: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        offsets = np.asarray(offsets, dtype=np.int64)
        if ids.ndim != 1 or offsets.ndim != 1 or offsets.size == 0 or offsets[0] != 0:
            raise ValueError("ids and offsets must be 1-D and offsets must start at 0")
        lengths = np.diff(np.append(offsets, ids.size))
        if (lengths <= 0).any():
            raise ValueError("Every text needs at least one feature")
        if ids.size and (ids.min() < 0 or ids.max() >= self.weight.shape[0]):
            raise ValueError("Feature id outside the embedding table")
        sums = np.add.reduceat(self.rows(ids), offsets, axis=0)
        return sums / lengths.astype(np.float32)[:, None]


class Network:
    """One linear or MLP member."""

    def __init__(self, arch: str, embedding: Embedding, layers: dict[str, np.ndarray]):
        if arch not in _LAYERS:
            raise ValueError(f"Unknown architecture: {arch}")
        missing = [name for name in _LAYERS[arch] if name not in layers]
        extra = sorted(set(layers) - set(_LAYERS[arch]))
        if missing or extra:
            raise ValueError(f"{arch} weights: missing {missing}, unexpected {extra}")
        for name, value in layers.items():
            if value.dtype != np.float32:
                raise ValueError(f"{name} must be float32, got {value.dtype}")
        width = embedding.shape[1]
        if arch == "mlp":
            hidden = layers["hidden.weight"]
            if hidden.shape[1] != width or layers["hidden.bias"].shape != (hidden.shape[0],):
                raise ValueError("Hidden layer shape does not match the embedding")
            width = hidden.shape[0]
        classifier = layers["classifier.weight"]
        if classifier.shape[1] != width or layers["classifier.bias"].shape != (classifier.shape[0],):
            raise ValueError("Classifier shape does not match its input")
        self.arch, self.embedding, self.layers = arch, embedding, layers
        self.languages = classifier.shape[0]

    def logits(self, ids: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        x = self.embedding.mean(ids, offsets)
        if self.arch == "mlp":
            x = np.maximum(x @ self.layers["hidden.weight"].T + self.layers["hidden.bias"], np.float32(0))
        return x @ self.layers["classifier.weight"].T + self.layers["classifier.bias"]

    def logits_with_agreement(self, ids, offsets):
        logits = self.logits(ids, offsets)
        return logits, np.ones(logits.shape[0], dtype=bool)


class Ensemble:
    """Weighted temperature-scaled logits with an optional agreement gate."""

    def __init__(self, members: list[Network], weights, temperatures, require_agreement=False):
        if not members or len(members) != len(weights) or len(members) != len(temperatures):
            raise ValueError("Invalid ensemble components")
        if any(w <= 0 for w in weights) or any(t <= 0 for t in temperatures):
            raise ValueError("Weights and temperatures must be positive")
        if len({m.languages for m in members}) != 1:
            raise ValueError("Ensemble members must predict the same languages")
        total = sum(weights)
        self.members = members
        self.weights = [w / total for w in weights]
        self.temperatures = list(temperatures)
        self.require_agreement = bool(require_agreement)
        self.languages = members[0].languages

    def logits_with_agreement(self, ids, offsets):
        outputs = [member.logits(ids, offsets) for member in self.members]
        logits = sum(np.float32(w) * value / np.float32(t)
                     for w, value, t in zip(self.weights, outputs, self.temperatures))
        predictions = np.stack([value.argmax(1) for value in outputs])
        agree = (predictions == predictions[0]).all(0)
        if not self.require_agreement:
            agree = np.ones_like(agree)
        return logits, agree

    def logits(self, ids, offsets):
        return self.logits_with_agreement(ids, offsets)[0]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Building networks from stored arrays
# --------------------------------------------------------------------------


def _network(config: dict, arrays: dict[str, np.ndarray], languages: int) -> Network:
    storage = config.get("weight_storage", "fp32")
    if storage not in WEIGHT_STORAGES:
        raise ValueError(f"Unknown weight storage: {storage!r}")
    arrays = dict(arrays)
    weight = arrays.pop("embedding.weight", None)
    scale = arrays.pop("embedding.weight" + ROW_SCALE_SUFFIX, None)
    if weight is None:
        raise ValueError("Missing embedding.weight")
    allowed = {"fp32": (np.float32,), "fp16": (np.float16,), "int8_rowwise": (np.int8, np.float32)}[storage]
    if weight.dtype not in allowed:
        raise ValueError(f"{storage} storage with {weight.dtype} embedding")
    layers = {}
    for name, value in arrays.items():
        if ROW_SCALE_SUFFIX in name:
            raise ValueError(f"Row scale without an int8 tensor: {name}")
        if storage == "fp16":
            if value.dtype != np.float16:
                raise ValueError(f"fp16 storage with non-float16 tensor {name}")
            value = value.astype(np.float32)
        layers[name] = value
    embedding = Embedding(weight, scale)
    buckets = config.get("buckets", BUCKETS)
    if embedding.shape[0] != buckets:
        raise ValueError(f"Embedding has {embedding.shape[0]} rows, config says {buckets} buckets")
    if "embedding_dim" in config and embedding.shape[1] != config["embedding_dim"]:
        raise ValueError("Embedding width does not match config embedding_dim")
    network = Network(config.get("arch", "linear"), embedding, layers)
    if network.languages != languages:
        raise ValueError(f"Classifier has {network.languages} outputs for {languages} languages")
    return network


def _check_members(configs: list[dict], language_lists: list[list[str]]) -> None:
    first = encoding_options(configs[0])
    for config, languages in zip(configs[1:], language_lists[1:]):
        if languages != language_lists[0]:
            raise ValueError("Ensemble members must use the same language order")
        other = encoding_options(config)
        for key in FEATURE_DEFAULTS:
            if other[key] != first[key]:
                raise ValueError("Incompatible ensemble feature encoding: " + key)


def build(meta: dict, member_arrays: list[dict[str, np.ndarray]]):
    """Model from release metadata and per-member arrays."""
    languages = meta["languages"]
    configs = [member["config"] for member in meta["members"]]
    if len(configs) != len(member_arrays):
        raise ValueError("Metadata and arrays disagree on the number of members")
    _check_members(configs, [languages] * len(configs))
    networks = [_network(c, a, len(languages)) for c, a in zip(configs, member_arrays)]
    ensemble = meta.get("ensemble")
    if ensemble:
        return Ensemble(networks, ensemble["weights"], ensemble["temperatures"],
                        ensemble.get("require_agreement", False))
    if len(networks) != 1:
        raise ValueError("Several members need ensemble weights")
    return networks[0]


# --------------------------------------------------------------------------
# PyTorch checkpoints (already loaded by the caller) and .npz files
# --------------------------------------------------------------------------


def _arrays(state_dict: dict) -> dict[str, np.ndarray]:
    return {name: tensor.detach().cpu().numpy() for name, tensor in state_dict.items()}


def split_checkpoint(checkpoint: dict) -> tuple[dict, list[dict[str, np.ndarray]]]:
    """Release metadata and member arrays of a torch-loaded checkpoint dict."""
    members, arrays = [], []
    for member in [checkpoint] + list((checkpoint.get("ensemble") or {}).get("companions", [])):
        if member is not checkpoint and member.get("ensemble"):
            raise ValueError("Nested ensembles are not supported")
        if member is not checkpoint and member["languages"] != checkpoint["languages"]:
            raise ValueError("Ensemble members must use the same language order")
        members.append({"config": dict(member.get("config", {}))})
        arrays.append(_arrays(member["state_dict"]))
    meta = {"languages": list(checkpoint["languages"]), "members": members,
            "calibration": checkpoint.get("calibration", {})}
    ensemble = checkpoint.get("ensemble")
    if ensemble:
        meta["ensemble"] = {key: ensemble[key] for key in ("weights", "temperatures")}
        meta["ensemble"]["require_agreement"] = ensemble.get("require_agreement", False)
    return meta, arrays


def from_checkpoint(checkpoint: dict):
    meta, arrays = split_checkpoint(checkpoint)
    return meta, build(meta, arrays)


def save_npz(path: Path, meta: dict, member_arrays: list[dict[str, np.ndarray]]) -> None:
    """Write a release file; refuses metadata that is not plain JSON."""
    meta = dict(meta, format=FORMAT, format_version=FORMAT_VERSION)
    encoded = json.dumps(meta, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if json.loads(encoded) != meta:
        raise ValueError("Metadata does not survive a JSON round trip")
    build(meta, member_arrays)  # refuse files that would not load
    payload = {META_KEY: np.frombuffer(encoded.encode("utf-8"), dtype=np.uint8)}
    for index, arrays in enumerate(member_arrays):
        for name, value in arrays.items():
            payload[f"member{index}/{name}"] = value
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **payload)


def load_npz(path: str | Path):
    with np.load(path, allow_pickle=False) as data:
        if META_KEY not in data.files:
            raise ValueError(f"{path} is not a {FORMAT} file (no {META_KEY})")
        meta = json.loads(bytes(data[META_KEY]).decode("utf-8"))
        if meta.get("format") != FORMAT or meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported format {meta.get('format')!r} "
                             f"version {meta.get('format_version')!r}")
        member_arrays: list[dict[str, np.ndarray]] = [{} for _ in meta["members"]]
        for key in data.files:
            if key == META_KEY:
                continue
            prefix, _, name = key.partition("/")
            if not prefix.startswith("member") or not prefix[6:].isdigit() or int(prefix[6:]) >= len(member_arrays):
                raise ValueError(f"{path}: unexpected array {key}")
            member_arrays[int(prefix[6:])][name] = data[key]
    return meta, build(meta, member_arrays)


def quantize_rowwise_int8(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row int8 (same arithmetic as the PyTorch tool)."""
    if weight.ndim != 2 or weight.dtype != np.float32:
        raise ValueError("Row-wise int8 needs a 2-D float32 array")
    scale = np.abs(weight).max(axis=1, keepdims=True) / np.float32(127.0)
    safe = np.where(scale > 0, scale, np.float32(1.0))
    values = np.where(scale > 0, weight / safe, np.float32(0.0))
    return np.clip(np.round(values), -127, 127).astype(np.int8), scale.astype(np.float32)
