"""CSV I/O and stable, accent-insensitive keys for data separation."""

import csv
import unicodedata
from pathlib import Path

csv.field_size_limit(10_000_000)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["text", "language"])
        writer.writeheader()
        writer.writerows(rows)


def text_key(text: str) -> str:
    text = text.casefold().translate(str.maketrans({"ł": "l"}))
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )
    return " ".join("".join(c if c.isalpha() else " " for c in text[:384]).split())
