"""Calibrate on public validation, then evaluate fixed public/synthetic tests."""

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch

from dataset_utils import read_csv
from language_detector import LanguageDetector
from train_model import EncodedDataset, score

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "model.pt")
    args = parser.parse_args()
    path = args.model.resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    detector = LanguageDetector(path)
    config = checkpoint["config"]
    validation = EncodedDataset(
        [ROOT / "data/prepared/valid_short.csv"], detector.languages, config, 384
    )
    with torch.no_grad():
        logits = torch.cat(
            [
                detector.model(*validation.batch(indices))
                for indices in torch.arange(len(validation)).split(512)
            ]
        )
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], max_iter=35, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            logits / log_temperature.exp().clamp(0.1, 10), validation.targets
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = log_temperature.exp().clamp(0.1, 10).item()
    checkpoint["calibration"] = {
        "temperature": temperature,
        "min_confidence": 0.9,
        "rule": "Fixed operational cutoff after public-validation temperature scaling; not a universal accuracy guarantee.",
    }
    torch.save(checkpoint, path)
    detector = LanguageDetector(path)
    test = EncodedDataset(
        [ROOT / "data/prepared/test_short.csv"], detector.languages, config, 384
    )
    metrics = score(detector.model, test, 512, len(detector.languages))
    metrics["per_language"] = {
        detector.languages[int(i)]: v for i, v in metrics["per_language"].items()
    }
    rows = [r for r in read_csv(ROOT / "data/synthetic.csv") if r["language"] != "und"]
    predictions = detector.predict_many([r["text"] for r in rows])
    synthetic = {
        "samples": len(rows),
        "languages": len({r["language"] for r in rows}),
        "accuracy": sum(r["language"] == p for r, p in zip(rows, predictions))
        / len(rows),
        "kind": "Assistant-authored, held-out diagnostic; not a production benchmark",
    }
    benchmark = {}
    for length in (64, 256, 2000):
        text = ("Nie mogę się zalogować po ostatnim update. Proszę o pomoc. " * 50)[
            :length
        ]
        for _ in range(30):
            detector.predict(text)
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            detector.predict(text)
            times.append((time.perf_counter() - start) * 1000)
        benchmark[str(length)] = {
            "median_ms": statistics.median(times),
            "p95_ms": sorted(times)[math.ceil(0.95 * len(times)) - 1],
        }
    summary = {
        "language_count": 100,
        "model_bytes": path.stat().st_size,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "public_test": metrics,
        "synthetic_test": synthetic,
        "benchmark": benchmark,
        "calibration": checkpoint["calibration"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "threads": 2,
        },
        "protocol": "30 warmup+1000 predictions per length; encoding+network, excludes imports/loading/extraction; public/source and synthetic tests only.",
        "test_sha256": hashlib.sha256(
            (ROOT / "data/prepared/test_short.csv").read_bytes()
        ).hexdigest(),
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/public_evaluation.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "public_accuracy": metrics["accuracy"],
                "public_macro_recall": metrics["macro_recall"],
                "synthetic": synthetic,
                "benchmark": benchmark,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
