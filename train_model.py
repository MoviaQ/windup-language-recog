"""Reproducible training with feature caching and validation checkpoint selection."""

import argparse
import copy
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dataset_utils import read_csv
from language_detector import build_model, encode

ROOT = Path(__file__).resolve().parent


def _encode_ids(job: tuple[str, dict, int]) -> bytes:
    text, config, cap = job
    return np.asarray(
        encode(
            text,
            buckets=config["buckets"],
            ngrams=tuple(config["ngrams"]),
            max_length=cap,
            word_features=config.get("word_features", False),
            preprocessing=config.get("preprocessing", "none"),
            word_repeat=config.get("word_repeat", 4),
        ),
        dtype=np.int32,
    ).tobytes()


def encoding_config(config: dict) -> dict:
    """The config keys that change encode() output (not model size or arch)."""
    return {
        "buckets": config["buckets"],
        "ngrams": list(config["ngrams"]),
        "word_features": config.get("word_features", False),
        "preprocessing": config.get("preprocessing", "none"),
        "word_repeat": config.get("word_repeat", 4),
    }


def _encode_batch(job: tuple[list[str], dict, int]) -> tuple[np.ndarray, np.ndarray]:
    texts, config, cap = job
    pieces = [np.frombuffer(_encode_ids((text, config, cap)), dtype=np.int32) for text in texts]
    lengths = np.fromiter((len(piece) for piece in pieces), dtype=np.int64, count=len(pieces))
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    return np.concatenate(pieces), offsets


class StreamingDataset:
    """Encode every batch on the fly in worker processes; nothing is cached.

    For corpora whose feature cache would not fit on disk. Encoding is
    deterministic, so a given row order gives the same batches as
    EncodedDataset.
    """

    def __init__(self, files: list[Path], languages: list[str], config: dict, cap: int, workers: int):
        lookup = {language: index for index, language in enumerate(languages)}
        self.texts: list[str] = []
        targets: list[int] = []
        unknown: Counter = Counter()
        for file in files:
            with file.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                for row in reader:
                    index = lookup.get(row["language"])
                    if index is None:
                        unknown[row["language"]] += 1
                        continue
                    self.texts.append(row["text"])
                    targets.append(index)
        if unknown:
            raise ValueError(f"Labels outside the language list: {dict(unknown)}")
        if not self.texts:
            raise ValueError("Dataset is empty.")
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.counts = Counter({languages[i]: c for i, c in Counter(targets).items()})
        self.config, self.cap, self.workers = config, cap, workers

    def __len__(self) -> int:
        return len(self.targets)

    def iter_batches(self, index_batches, prefetch: int | None = None):
        """Yield batches in order, encoding at most `prefetch` ahead.

        Pool.imap is not used: its feeder submits every job at once, so when
        encoding outpaces training the finished batches pile up in memory
        (Exp D on 14.9M rows was OOM-killed at 13 GB).
        """
        import multiprocessing
        from collections import deque

        prefetch = prefetch or 2 * self.workers
        if prefetch < 1:
            raise ValueError("prefetch must be positive")
        batches = iter(index_batches)
        pending: deque = deque()
        self.max_in_flight = 0
        with multiprocessing.get_context("spawn").Pool(self.workers) as pool:

            def submit() -> None:
                indices = next(batches, None)
                if indices is not None:
                    job = ([self.texts[i] for i in indices.tolist()], self.config, self.cap)
                    pending.append((indices, pool.apply_async(_encode_batch, (job,))))
                    self.max_in_flight = max(self.max_in_flight, len(pending))

            for _ in range(prefetch):
                submit()
            while pending:
                indices, result = pending.popleft()
                ids, offsets = result.get()
                submit()
                yield indices, (torch.from_numpy(ids.astype(np.int64)), torch.from_numpy(offsets))


class EncodedDataset:
    """Cache features on disk instead of encoding them in every epoch."""

    def __init__(
        self,
        files: list[Path],
        languages: list[str],
        config: dict,
        cap: int,
        workers: int = 1,
    ):
        signature = hashlib.sha256(
            json.dumps(
                {"encoding": encoding_config(config), "cap": cap, "languages": languages},
                sort_keys=True,
            ).encode()
        )
        rows = []
        for file in files:
            signature.update(file.read_bytes())
            rows.extend(read_csv(file))
        if not rows:
            raise ValueError("Dataset is empty.")
        lookup = {language: index for index, language in enumerate(languages)}
        unknown = Counter(row["language"] for row in rows if row["language"] not in lookup)
        if unknown:
            raise ValueError(f"Labels outside the language list: {dict(unknown)}")
        self.targets = torch.tensor(
            [lookup[row["language"]] for row in rows], dtype=torch.long
        )
        self.counts = Counter(row["language"] for row in rows)
        prefix = ROOT / "data" / "cache" / signature.hexdigest()[:24]
        prefix.parent.mkdir(parents=True, exist_ok=True)
        ids_path, offsets_path = prefix.with_suffix(".bin"), prefix.with_suffix(".npy")
        if not ids_path.exists() or not offsets_path.exists():
            temporary = ids_path.with_suffix(".tmp")
            offsets = [0]
            jobs = ((row["text"], config, cap) for row in rows)
            pool = None
            if workers > 1:
                import multiprocessing

                pool = multiprocessing.get_context("spawn").Pool(workers)
                # imap preserves order, so the cache is identical to workers=1.
                encoded = pool.imap(_encode_ids, jobs, chunksize=256)
            else:
                encoded = map(_encode_ids, jobs)
            try:
                with temporary.open("wb") as stream:
                    for index, data in enumerate(encoded):
                        stream.write(data)
                        offsets.append(offsets[-1] + len(data) // 4)
                        if (index + 1) % 100000 == 0:
                            print(f"Encoding: {index + 1}/{len(rows)}", flush=True)
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()
            temporary.replace(ids_path)
            np.save(offsets_path, np.asarray(offsets, dtype=np.int64))
        self.offsets = np.load(offsets_path)
        self.ids = np.memmap(ids_path, mode="r", dtype=np.int32)
        if len(self.offsets) != len(rows) + 1 or self.offsets[-1] != len(self.ids):
            raise ValueError(f"Invalid feature cache: {prefix}")

    def __len__(self) -> int:
        return len(self.targets)

    def iter_batches(self, index_batches):
        for indices in index_batches:
            yield indices, self.batch(indices)

    def batch(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        selected = indices.numpy()
        starts, ends = self.offsets[selected], self.offsets[selected + 1]
        offsets = np.concatenate(([0], np.cumsum(ends - starts)[:-1]))
        ids = np.concatenate([self.ids[start:end] for start, end in zip(starts, ends)])
        return torch.from_numpy(ids.astype(np.int64)), torch.from_numpy(offsets)


def _jsonable(value):
    """Arguments for the JSON report: paths (also inside --valid-csv lists) as str."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def temperature_row_probabilities(
    targets: torch.Tensor, counts: list[int], alpha: float
) -> torch.Tensor:
    """Per-row draw probabilities giving language L a share proportional to n_L**alpha."""
    counts_tensor = torch.tensor([float(count) for count in counts])
    if (counts_tensor == 0).any():
        missing = [index for index, count in enumerate(counts) if count == 0]
        raise ValueError(f"Temperature sampling needs data for every class: {missing}")
    share = counts_tensor**alpha
    share /= share.sum()
    return (share / counts_tensor)[targets]


def score(
    model: nn.Module, data: EncodedDataset, batch_size: int, classes: int
) -> dict:
    model.eval()
    confusion = torch.zeros((classes, classes), dtype=torch.long)
    total_loss = 0.0
    with torch.inference_mode():
        for indices in torch.arange(len(data)).split(batch_size):
            logits = model(*data.batch(indices))
            targets = data.targets[indices]
            total_loss += nn.functional.cross_entropy(
                logits, targets, reduction="sum"
            ).item()
            predicted = logits.argmax(dim=1)
            confusion += torch.bincount(
                targets * classes + predicted, minlength=classes * classes
            ).reshape(classes, classes)
    support = confusion.sum(dim=1)
    present = support > 0
    correct = confusion.diag()
    recall = correct.double() / support.clamp(min=1)
    precision = correct.double() / confusion.sum(dim=0).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return {
        "samples": len(data),
        "accuracy": correct.sum().item() / len(data),
        "macro_recall": recall[present].mean().item(),
        "macro_f1": f1[present].mean().item(),
        "loss": total_loss / len(data),
        "per_language": {
            str(index): {
                "samples": support[index].item(),
                "recall": recall[index].item(),
                "precision": precision[index].item(),
                "f1": f1[index].item(),
            }
            for index in range(classes)
            if support[index]
        },
        "confusion": confusion.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--buckets", type=int, default=65536)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--arch", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--train-length", type=int, default=384)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path, default=ROOT / "model.pt")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--word-features", action="store_true")
    parser.add_argument("--ngram-max", type=int, default=4)
    parser.add_argument(
        "--preprocessing", choices=("none", "ticket_v2"), default="none"
    )
    parser.add_argument(
        "--selection", choices=("macro_f1", "accuracy"), default="macro_f1"
    )
    parser.add_argument("--uniform-loss", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--short-only", action="store_true")
    parser.add_argument(
        "--valid-csv",
        type=Path,
        action="append",
        help="Validation CSV(s) for checkpoint selection, replacing the "
        "data-dir valid*.csv files (e.g. FLORES dev from tools/export_flores_csv.py).",
    )
    parser.add_argument(
        "--encode-workers",
        type=int,
        default=1,
        help="Processes for building feature caches; output is identical.",
    )
    parser.add_argument(
        "--stream-workers",
        type=int,
        default=0,
        help="Encode training batches on the fly in N processes instead of "
        "building a disk cache (for corpora whose cache does not fit on disk).",
    )
    parser.add_argument(
        "--sampling-alpha",
        type=float,
        help="Temperature sampling: each epoch draws rows so that language L "
        "has probability proportional to n_L**alpha (0.3 as in NLLB/OpenLID). "
        "Requires --uniform-loss so classes are not rebalanced twice.",
    )
    args = parser.parse_args()
    if args.sampling_alpha is not None:
        if not 0 < args.sampling_alpha <= 1:
            parser.error("--sampling-alpha must be in (0, 1].")
        if not args.uniform_loss:
            parser.error(
                "--sampling-alpha already rebalances languages; pass --uniform-loss "
                "to disable the inverse-square-root loss weights."
            )
    if args.encode_workers < 1:
        parser.error("--encode-workers must be positive.")
    if args.stream_workers < 0:
        parser.error("--stream-workers must be zero (cache) or positive.")
    if (
        min(
            args.epochs,
            args.patience,
            args.batch_size,
            args.threads,
            args.buckets,
            args.embedding_dim,
            args.train_length,
        )
        < 1
        or args.lr <= 0
    ):
        parser.error("Training parameters must be positive.")
    if args.train_length > 384:
        parser.error(
            "--train-length cannot exceed the 384 characters checked for duplicates."
        )
    if not 0 <= args.dropout < 1 or args.ngram_max < 1:
        parser.error("Dropout must be in [0, 1), and --ngram-max must be positive.")
    start = time.perf_counter()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    languages = manifest["languages"]
    config = {
        "buckets": args.buckets,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "arch": args.arch,
        "ngrams": list(range(1, args.ngram_max + 1)),
        "max_length": 2000,
        "word_features": args.word_features,
        "preprocessing": args.preprocessing,
    }
    suffixes = (
        ("_short", "_mixed", "_ascii", "_chat")
        if args.short_only
        else ("", "_short", "_mixed", "_ascii", "_chat")
    )
    train_files = [
        args.data_dir / f"train{suffix}.csv"
        for suffix in suffixes
        if (args.data_dir / f"train{suffix}.csv").exists()
    ]
    valid_files = args.valid_csv or [
        args.data_dir / f"valid{suffix}.csv"
        for suffix in ("", "_short", "_mixed", "_ascii")
        if (args.data_dir / f"valid{suffix}.csv").exists()
    ]
    print(f"Languages: {len(languages)}; preparing feature caches...", flush=True)
    if args.stream_workers:
        training = StreamingDataset(
            train_files, languages, config, args.train_length, args.stream_workers
        )
    else:
        training = EncodedDataset(
            train_files, languages, config, args.train_length, args.encode_workers
        )
    validation = EncodedDataset(
        valid_files, languages, config, args.train_length, args.encode_workers
    )
    row_probabilities = None
    if args.sampling_alpha is not None:
        row_probabilities = temperature_row_probabilities(
            training.targets,
            [training.counts[language] for language in languages],
            args.sampling_alpha,
        )
    model = build_model(len(languages), args.buckets, args.embedding_dim, args.hidden_dim, args.arch)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Inverse-square-root frequency weights reduce large-language dominance.
    weights = torch.tensor(
        [training.counts[language] ** -0.5 for language in languages]
    )
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=None if args.uniform_loss else weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    best, best_epoch, stale = -1.0, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        if row_probabilities is None:
            order = torch.randperm(len(training))
        else:
            order = torch.multinomial(row_probabilities, len(training), replacement=True)
        for indices, inputs in training.iter_batches(order.split(args.batch_size)):
            optimizer.zero_grad(set_to_none=True)
            if args.dropout:
                keep = torch.rand(len(inputs[0])) >= args.dropout
                # Keep at least the first feature of each input.
                keep[inputs[1]] = True
                cumulative = keep.long().cumsum(0)
                inputs = (inputs[0][keep], cumulative[inputs[1]] - 1)
            logits = model(*inputs)
            loss = criterion(logits, training.targets[indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        metrics = score(model, validation, args.batch_size, len(languages))
        selection_score = metrics[args.selection]
        scheduler.step(selection_score)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / len(training),
            "valid_accuracy": metrics["accuracy"],
            "valid_macro_f1": metrics["macro_f1"],
            "seconds": time.perf_counter() - epoch_start,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if selection_score > best + 1e-5:
            best, best_epoch, stale = selection_score, epoch, 0
            args.model.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "languages": languages,
                    "config": config,
                    "training": {
                        "seed": args.seed,
                        "best_epoch": epoch,
                        "selection_metric": args.selection,
                        "selection_score": best,
                        "validation_macro_f1": metrics["macro_f1"],
                        "manifest": manifest,
                    },
                },
                args.model,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping: validation did not improve.", flush=True)
                break
    best_record = next(record for record in history if record["epoch"] == best_epoch)
    report = {
        "config": config,
        "seed": args.seed,
        "training_samples": len(training),
        "validation_samples": len(validation),
        "train_files": [str(path) for path in train_files],
        "valid_files": [str(path) for path in valid_files],
        "best_epoch": best_epoch,
        "best_valid_macro_f1": best_record["valid_macro_f1"],
        "seconds": time.perf_counter() - start,
        "history": history,
        "arguments": {key: _jsonable(value) for key, value in vars(args).items()},
    }
    report["selection_metric"] = args.selection
    report["best_selection_score"] = best
    report_path = args.report_dir / "training.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Best epoch: {best_epoch}; saved {args.model}", flush=True)


if __name__ == "__main__":
    main()
