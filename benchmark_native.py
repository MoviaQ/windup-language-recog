"""Compare Python/native encoding on varied public text and warmed repeated input."""

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from pathlib import Path

import torch

import language_detector as detector_module
from dataset_utils import read_csv
from ticket_text import normalize_ticket_text


def measure(predict, texts):
    times = []
    for text in texts:
        start = time.perf_counter()
        predict(text)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": statistics.median(times),
        "p95_ms": sorted(times)[int(0.95 * (len(times) - 1))],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/synthetic.csv"))
    parser.add_argument("--model", type=Path, default=Path("model.pt"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--fasttext-bin", type=Path)
    parser.add_argument("--fasttext-ftz", type=Path)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    native = detector_module._native_char_features
    if native is None:
        parser.error("build the optional encoder first; see native/README.md")
    import _windup_native

    torch.set_num_threads(2)
    model = detector_module.LanguageDetector(args.model)
    rows = [
        row
        for row in read_csv(args.csv)
        if row["language"] != "und"
        and any(c.isalpha() for c in normalize_ticket_text(row["text"]))
    ]
    random.Random(913).shuffle(rows)
    texts = [row["text"] for row in rows[: args.samples]]
    if not texts:
        parser.error("CSV contains no usable examples")
    for text in texts:
        expected = detector_module.encode(text, **model.encoding_options)
        detector_module._native_char_features = None
        try:
            assert expected == detector_module.encode(text, **model.encoding_options)
        finally:
            detector_module._native_char_features = native
    backends = {"python": (None, model.predict), "native": (native, model.predict)}
    fasttext_models = {}
    for name, path in (
        ("fasttext_bin", args.fasttext_bin),
        ("fasttext_ftz", args.fasttext_ftz),
    ):
        if path is not None:
            import fasttext

            ft = fasttext.load_model(str(path))
            fasttext_models[name] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

            def predict(text, ft=ft):
                text = normalize_ticket_text(text)
                # Match the encoder's NFC/lowercase/2000-character cap.
                import unicodedata

                return ft.predict(
                    [unicodedata.normalize("NFC", text).lower().strip()[:2000]], k=1
                )[0][0][0]

            backends[name] = (native, predict)
    results = {}
    try:
        for name, (encoder, predict) in backends.items():
            detector_module._native_char_features = encoder
            for text in texts[:30]:
                predict(text)
            _windup_native.cache_clear()
            values = {"varied_pass_from_empty_fragment_cache": measure(predict, texts)}
            repeated = {}
            for length in (64, 256, 2000):
                text = (
                    "Nie mogę się zalogować po ostatnim update. Proszę o pomoc. " * 50
                )[:length]
                for _ in range(30):
                    predict(text)
                repeated[str(length)] = measure(predict, [text] * 1000)
            values["repeated_warm_fragment_cache"] = repeated
            results[name] = values
    finally:
        detector_module._native_char_features = native
    output = {
        "samples": len(texts),
        "encoding_equality_checked": len(texts),
        "csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "threads": 2,
        },
        "protocol": "Seed913. Varied CSV sample, one pass after clearing native fragment cache; repeated inputs 30warmup+1000calls. Cleanup+encoding+PyTorch included, imports/loading/extraction excluded. Cache stores exact <=20-byte fragment hashes in32768slots, about1MiB; never predictions. No full-text memoization.",
        "fasttext_models": fasttext_models,
        "results": results,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
