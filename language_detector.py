"""A compact PyTorch language classifier using hashed character n-grams."""

import argparse
import csv
import hashlib
import re
import threading
import unicodedata
from pathlib import Path

import torch
from torch import nn

BUCKETS = 8192
_word_cache: dict[tuple[str, int], int] = {}
_word_cache_lock = threading.Lock()


def encode(
    text: str,
    *,
    buckets: int = BUCKETS,
    ngrams: tuple[int, ...] = (1, 2, 3),
    max_length: int = 2000,
    word_features: bool = False,
    preprocessing: str = "none",
) -> list[int]:
    if preprocessing == "ticket_v2":
        from ticket_text import normalize_ticket_text

        text = normalize_ticket_text(text)
    elif preprocessing != "none":
        raise ValueError(f"Unknown preprocessing: {preprocessing}")
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = text[:max_length]
    if not any(char.isalpha() for char in text):
        raise ValueError("Text must contain at least one letter.")
    text = " " + text + " "
    features = [
        int.from_bytes(
            hashlib.blake2b(text[i : i + n].encode(), digest_size=8).digest(), "little"
        )
        % buckets
        for n in ngrams
        for i in range(len(text) - n + 1)
    ]
    if word_features:
        words = re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*", text, flags=re.UNICODE)
        for word in words:
            key = (word, buckets)
            with _word_cache_lock:
                if key not in _word_cache:
                    if len(_word_cache) >= 10000:
                        _word_cache.clear()
                    _word_cache[key] = (
                        int.from_bytes(
                            hashlib.blake2b(("\x00W" + word).encode(), digest_size=8).digest(),
                            "little",
                        )
                        % buckets
                    )
                index = _word_cache[key]
            features.extend([index] * 4)
    return features


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


class LanguageDetector:
    def __init__(self, model_path: str | Path = "model.pt"):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.languages = checkpoint["languages"]
        config = checkpoint.get("config", {})
        self.encoding_options = {
            "buckets": config.get("buckets", BUCKETS),
            "ngrams": tuple(config.get("ngrams", (1, 2, 3))),
            "max_length": config.get("max_length", 2000),
            "word_features": config.get("word_features", False),
            "preprocessing": config.get("preprocessing", "none"),
        }
        self.model = build_model(
            len(self.languages),
            self.encoding_options["buckets"],
            config.get("embedding_dim", 32),
            config.get("hidden_dim", 128),
            config.get("arch", "linear"),
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.calibration = checkpoint.get("calibration", {})
        self._cache: dict[str, str] = {}
        self._cache_size = 1000
        self._cache_lock = threading.Lock()

    def _cache_get(self, text: str) -> str | None:
        with self._cache_lock:
            return self._cache.get(text)

    def _cache_put(self, text: str, result: str) -> None:
        with self._cache_lock:
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[text] = result

    def predict(self, text: str) -> str:
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        with torch.inference_mode():
            index = (
                self.model(*batch([text], **self.encoding_options)).argmax(dim=1).item()
            )
        result = self.languages[index]
        self._cache_put(text, result)
        return result

    def predict_many(self, texts: list[str], batch_size: int = 128) -> list[str]:
        """Predict a sequence of texts without reloading weights."""
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        results: list[str | None] = [self._cache_get(text) for text in texts]
        pending = [i for i, result in enumerate(results) if result is None]
        if pending:
            with torch.inference_mode():
                for start in range(0, len(pending), batch_size):
                    group = pending[start : start + batch_size]
                    chunk = [texts[i] for i in group]
                    indices = (
                        self.model(*batch(chunk, **self.encoding_options))
                        .argmax(dim=1)
                        .tolist()
                    )
                    for pos, index in zip(group, indices):
                        language = self.languages[index]
                        results[pos] = language
                        self._cache_put(texts[pos], language)
        return [result for result in results if result is not None]

    def predict_details(self, text: str) -> dict:
        """Return optional confidence details and the und code when uncertain."""
        with torch.inference_mode():
            logits = self.model(*batch([text], **self.encoding_options))
            temperature = self.calibration.get("temperature", 1.0)
            probabilities = (logits / temperature).softmax(dim=1)
            confidence, index = probabilities.max(dim=1)
            confidence = confidence.item()
            best = self.languages[index.item()]
        threshold = self.calibration.get("min_confidence")
        uncertain = threshold is not None and confidence < threshold
        return {
            "language": "und" if uncertain else best,
            "best_language": best,
            "confidence": confidence,
            "uncertain": uncertain,
            "calibrated": bool(self.calibration),
        }

    def predict_ticket(
        self, title: str, description: str = "", *, allow_uncertain: bool = False
    ) -> str:
        from ticket_text import extract_ticket_text

        text = extract_ticket_text(title, description)
        if allow_uncertain:
            if not self.calibration:
                raise ValueError("Uncertainty mode requires a calibrated model.")
            try:
                return self.predict_details(text)["language"]
            except ValueError:
                if not any(char.isalpha() for char in text):
                    return "und"
                raise
        return self.predict(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--data", type=Path, default=Path("data/demo.csv"))
    training.add_argument("--model", type=Path, default=Path("model.pt"))
    training.add_argument("--epochs", type=int, default=120)
    prediction = commands.add_parser("predict")
    prediction.add_argument("text")
    prediction.add_argument("--model", type=Path, default=Path("model.pt"))
    args = parser.parse_args()
    try:
        if args.command == "train":
            if args.epochs < 1:
                raise ValueError("The number of epochs must be positive.")
            train(args.data, args.model, args.epochs)
        else:
            print(LanguageDetector(args.model).predict(args.text))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
