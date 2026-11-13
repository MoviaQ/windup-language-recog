"""A compact language classifier using hashed character n-grams.

Inference runs on numpy (numpy_runtime.py). PyTorch is needed only to train
and to read legacy ``.pt`` checkpoints; ``.npz`` release files load without it.
"""

import argparse
import hashlib
import re
import threading
import unicodedata
from pathlib import Path

import numpy as np

import numpy_runtime

BUCKETS = 8192
DEFAULT_MODEL = Path("model.pt")
_word_cache: dict[tuple[str, int], int] = {}
_word_cache_lock = threading.Lock()
# Training-side names kept importable from here; they load PyTorch on first use.
_TORCH_EXPORTS = {
    "batch", "LanguageModel", "LanguageModelMLP", "build_model", "train",
    "LanguageModelEnsemble", "WEIGHT_STORAGES", "ROW_SCALE_SUFFIX", "quantize_rowwise_int8",
    "compress_state_dict", "decompress_state_dict", "model_from_checkpoint",
}


def __getattr__(name):
    if name in _TORCH_EXPORTS:
        import torch_models

        return getattr(torch_models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def encode(
    text: str,
    *,
    buckets: int = BUCKETS,
    ngrams: tuple[int, ...] = (1, 2, 3),
    max_length: int = 2000,
    word_features: bool = False,
    preprocessing: str = "none",
    word_repeat: int = 4,
) -> list[int]:
    if not isinstance(word_repeat, int) or isinstance(word_repeat, bool) or word_repeat < 1:
        raise ValueError("Word feature repetition must be a positive integer")
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
            features.extend([index] * word_repeat)
    return features



def encode_batch(texts: list[str], **encoding_options) -> tuple[np.ndarray, np.ndarray]:
    """Flat feature ids and per-text offsets (numpy, int64)."""
    ids: list[int] = []
    offsets: list[int] = []
    for text in texts:
        offsets.append(len(ids))
        ids.extend(encode(text, **encoding_options))
    return np.asarray(ids, dtype=np.int64), np.asarray(offsets, dtype=np.int64)


def load_model(model_path: str | Path):
    """(metadata, numpy model) from a .npz release file or a legacy .pt checkpoint."""
    path = Path(model_path)
    if path.suffix == ".npz":
        return numpy_runtime.load_npz(path)
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            f"{path} is a PyTorch checkpoint; reading it needs PyTorch "
            "(pip install -r requirements-training.txt). Release .npz files do not."
        ) from error
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return numpy_runtime.from_checkpoint(checkpoint)


def evidence_profile(text, *, case_sensitive=False):
    """Shared training/runtime profile for lexical evidence; no language labels."""
    from ticket_text import normalize_ticket_text

    text = normalize_ticket_text(text)
    words = re.findall(r"[^\W\d_]+", text)
    single = words[0] if len(words) == 1 and all("LATIN" in unicodedata.name(c, "") for c in words[0]) else None
    if single is not None and not case_sensitive:
        single = single.casefold()
    cjk = sum("CJK" in unicodedata.name(c, "") or "HIRAGANA" in unicodedata.name(c, "") or "KATAKANA" in unicodedata.name(c, "") or "HANGUL" in unicodedata.name(c, "") for c in text)
    return ("long" if len(words) > 3 or cjk >= 8 else "short"), single


class LanguageDetector:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL):
        meta, self.model = load_model(model_path)
        self.languages = meta["languages"]
        self.encoding_options = numpy_runtime.encoding_options(meta["members"][0]["config"])
        self.calibration = meta.get("calibration", {})
        self._ambiguous_single_words = set(self.calibration.get("ambiguous_single_words", []))
        self._single_word_language = self.calibration.get("single_word_language", {})
        self._cache: dict[str, str] = {}
        self._cache_size = 1000
        self._cache_lock = threading.Lock()

    def clear_cache(self) -> None:
        """Forget memoized predict() results (used by latency measurements)."""
        with self._cache_lock:
            self._cache.clear()

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
        index = int(self.model.logits(*encode_batch([text], **self.encoding_options)).argmax(axis=1)[0])
        result = self.languages[index]
        self._cache_put(text, result)
        return result

    def predict_many(self, texts: list[str], batch_size: int = 128) -> list[str]:
        """Predict a sequence of texts without reloading weights."""
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        results: list[str | None] = [self._cache_get(text) for text in texts]
        pending = [i for i, result in enumerate(results) if result is None]
        for start in range(0, len(pending), batch_size):
            group = pending[start : start + batch_size]
            chunk = [texts[i] for i in group]
            indices = self.model.logits(*encode_batch(chunk, **self.encoding_options)).argmax(axis=1).tolist()
            for pos, index in zip(group, indices):
                language = self.languages[index]
                results[pos] = language
                self._cache_put(texts[pos], language)
        return [result for result in results if result is not None]

    def predict_details(self, text: str) -> dict:
        """Return calibrated score details and abstain when evidence is insufficient."""
        logits, agreement = self.model.logits_with_agreement(*encode_batch([text], **self.encoding_options))
        agree = bool(agreement[0])
        temperature = self.calibration.get("temperature", 1.0)
        probabilities = numpy_runtime.softmax(logits / temperature)[0]
        index = int(probabilities.argmax())
        confidence = float(probabilities[index])
        best = self.languages[index]
        if probabilities.size >= 2:
            top2 = np.sort(probabilities)[-2:]
            margin = float(top2[1] - top2[0])
        else:
            margin = confidence
        threshold = self.calibration.get("min_confidence")
        threshold = self.calibration.get("min_confidence_by_language", {}).get(
            best, threshold
        )
        ambiguous = False
        length_group = None
        contextual = self.calibration.get("min_confidence_by_language_and_length", {})
        margin_floors = self.calibration.get("min_margin_by_length", {})
        if contextual or self._ambiguous_single_words or self._single_word_language or margin_floors:
            length_group, single_word = evidence_profile(text, case_sensitive=self.calibration.get("word_evidence_case_sensitive", False))
            lexical_language = self._single_word_language.get(single_word)
            ambiguous = single_word in self._ambiguous_single_words or (lexical_language is not None and lexical_language != best)
            threshold = contextual.get(best, {}).get(length_group, threshold)
        margin_ok = True
        if length_group is not None and margin_floors:
            floor = margin_floors.get(length_group)
            margin_ok = floor is None or margin >= floor
        uncertain = ambiguous or not agree or not margin_ok or (threshold is not None and confidence < threshold)
        return {
            "language": "und" if uncertain else best,
            "best_language": best,
            "confidence": confidence,
            "margin": margin,
            "uncertain": uncertain,
            "calibrated": bool(self.calibration),
        }

    def predict_guarded_ticket(
        self, title: str, description: str = "", *, cutoff: float = 0.65
    ) -> str:
        """Optional frozen ticket guard (opt-in only).

        Same input selection as the whole-ticket text (title/description
        via prepare_ticket), whole-ticket vote through predict_details
        acceptance, then one foreign-confident sentence veto at cutoff.
        No extra sentence-level heuristics are applied. It can abstain with
        und when a confident foreign sentence is detected; that is not
        reliable detection of all mixtures (a mixed ticket can still
        answer when no segment clears the cutoff). Works with any
        checkpoint (calibrated or raw); calibration, when present, still
        governs the whole-ticket vote. Letterless input returns und;
        other model errors propagate. Never touches fastText.
        """
        from ticket_guard import (
            RawPrediction,
            TicketPolicy,
            guarded_language,
            prepare_ticket,
        )

        TicketPolicy(name="ticket-guard", sentence_cutoff=cutoff)
        prepared = prepare_ticket(title or "", description or "")
        if not any(char.isalpha() for char in prepared.text):
            return "und"
        whole = self.predict_details(prepared.text)["language"]
        if whole == "und":
            return "und"

        def _raw(sentence: str) -> RawPrediction:
            details = self.predict_details(sentence)
            return RawPrediction(
                best=str(details.get("best_language", "und")),
                confidence=float(details.get("confidence", 0.0)),
                margin=float(details.get("margin", 0.0)),
                agreement=True,
            )

        return guarded_language(whole, prepared.sentences, _raw, float(cutoff))

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
    prediction.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    prediction.add_argument("--allow-uncertain", action="store_true", help="Return und when the calibrated acceptance policy rejects the input")
    args = parser.parse_args()
    try:
        if args.command == "train":
            if args.epochs < 1:
                raise ValueError("The number of epochs must be positive.")
            from torch_models import train

            train(args.data, args.model, args.epochs)
        else:
            detector = LanguageDetector(args.model)
            print(detector.predict_ticket(args.text, allow_uncertain=True) if args.allow_uncertain else detector.predict(args.text))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
