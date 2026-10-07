"""Prepare a reproducible, 100-language corpus from WiLI and MASSIVE."""

import hashlib
import json
import random
import re
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

from dataset_utils import read_csv, text_key, write_csv
from ticket_text import normalize_ticket_text

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw"
OUT = ROOT / "data/prepared"
SOURCES = {
    "wili-2018.zip": "https://zenodo.org/records/841984/files/wili-2018.zip",
    "massive-1.1.tar.gz": "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz",
}


def download(name):
    path = RAW / name
    if not path.exists():
        with urlopen(SOURCES[name], timeout=300) as response:
            contents = response.read()
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(contents)
        temporary.replace(path)
    return path


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    languages = sorted(json.loads((ROOT / "data/languages.json").read_text()))
    rows = {split: [] for split in ("train", "valid", "test")}
    wili = download("wili-2018.zip")
    if (
        hashlib.sha256(wili.read_bytes()).hexdigest()
        != "727e52ca4e13400e6def1b1b594ca12c8b7fd49ad9d45fd5c9e57a1f6d3b7a3f"
    ):
        raise ValueError("WiLI checksum mismatch")
    import csv
    import io

    with zipfile.ZipFile(wili) as archive:
        table = list(
            csv.DictReader(
                io.StringIO(archive.read("labels.csv").decode()), delimiter=";"
            )
        )
        mapping = {r["Label"]: r["ISO 369-3"].strip() for r in table}
        import pycountry

        for key, iso in list(mapping.items()):
            language = pycountry.languages.get(alpha_3=iso) if len(iso) == 3 else None
            mapping[key] = getattr(language, "alpha_2", None) or iso
        for split in ("train", "test"):
            texts = archive.read(f"x_{split}.txt").decode().splitlines()
            labels = archive.read(f"y_{split}.txt").decode().splitlines()
            if len(texts) != len(labels):
                raise ValueError("WiLI labels and texts mismatch")
            grouped = defaultdict(list)
            for text, label in zip(texts, labels):
                code = mapping[label]
                if code in languages:
                    grouped[code].append({"text": text[:2000], "language": code})
            for code, samples in sorted(grouped.items()):
                rng = random.Random(42)
                rng.shuffle(samples)
                if split == "train":
                    rows["valid"].extend(samples[:50])
                    rows["train"].extend(samples[50:])
                else:
                    rows["test"].extend(samples)
    massive = download("massive-1.1.tar.gz")
    if (
        hashlib.sha256(massive.read_bytes()).hexdigest()
        != "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
    ):
        raise ValueError("MASSIVE checksum mismatch")
    with tarfile.open(massive) as archive:
        for member in archive:
            if not member.name.endswith(".jsonl"):
                continue
            code = member.name.rsplit("/", 1)[-1].split("-")[0]
            if code not in languages:
                continue
            for line in archive.extractfile(member):
                entry = json.loads(line)
                split = "valid" if entry["partition"] == "dev" else entry["partition"]
                if split not in rows:
                    raise ValueError("Unknown MASSIVE partition")
                rows[split].append({"text": entry["utt"], "language": code})
    print(
        "Sources loaded",
        {split: len(samples) for split, samples in rows.items()},
        flush=True,
    )
    labels = defaultdict(set)
    for samples in rows.values():
        for row in samples:
            row["text"] = normalize_ticket_text(row["text"])
            labels[text_key(row["text"])].add(row["language"])
    conflicts = {key for key, codes in labels.items() if len(codes) > 1}
    reserved = {text_key(r["text"]) for r in read_csv(ROOT / "data/synthetic.csv")}
    heldout = set(reserved)
    counts = {}
    for split in ("test", "valid", "train"):
        rng = random.Random(945)
        own = set()
        samples = []
        for row in rows[split]:
            text = row["text"]
            if len(text) < 12 or not any(c.isalpha() for c in text[:384]):
                continue
            key = text_key(text)
            if not key or key in conflicts or key in heldout:
                continue
            variants = 2 if split == "train" and len(text) > 160 else 1
            for _ in range(variants):
                short = text
                if len(text) > 160:
                    sentences = [
                        s
                        for s in re.split(r"(?<=[.!?。！？])\s+", text)
                        if 20 <= len(s) <= 240
                    ]
                    if sentences:
                        short = rng.choice(sentences)
                    elif " " not in text:
                        short = text[: rng.randint(16, 60)]
                    else:
                        words = text.split()
                        length = min(len(words), rng.randint(4, 16))
                        start = rng.randrange(max(1, len(words) - length + 1))
                        short = " ".join(words[start : start + length])
                short_key = text_key(short)
                if (
                    not short_key
                    or not any(c.isalpha() for c in short)
                    or short_key in heldout
                    or short_key in conflicts
                    or short_key in own
                ):
                    continue
                own.add(short_key)
                samples.append({"text": short, "language": row["language"]})
            own.add(key)
        write_csv(OUT / f"{split}_short.csv", samples)
        heldout.update(own)
        counts[split] = len(samples)
        print(split, len(samples), flush=True)
    manifest = {
        "languages": languages,
        "language_count": len(languages),
        "seed": 945,
        "sources": SOURCES,
        "licenses": {
            "WiLI": "ODbL-1.0, underlying Wikipedia content has its own attribution/share-alike terms",
            "MASSIVE": "CC-BY-4.0",
        },
        "sha256_raw": {
            name: hashlib.sha256((RAW / name).read_bytes()).hexdigest()
            for name in SOURCES
        },
        "counts": counts,
        "synthetic_test_sha256": hashlib.sha256(
            (ROOT / "data/synthetic.csv").read_bytes()
        ).hexdigest(),
        "note": "Source partitions retained; exact normalized conflicts/duplicates removed; held-out synthetic test reserved.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
