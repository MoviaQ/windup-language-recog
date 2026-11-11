"""Write FLORES-200 dev items (full sentences + 20/50/100-char prefixes) as a
text,language CSV for train_model.py --valid-csv checkpoint selection.

Only the dev split is allowed: devtest is reserved for reporting
(docs/flores-benchmark.md). Output belongs in an ignored directory.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_flores import build_items, check_coverage, load_languages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flores-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    languages = load_languages()
    missing = check_coverage(args.flores_dir, "dev", languages)
    if missing:
        parser.error(f"FLORES dev lacks supported codes: {missing}")
    items = build_items(args.flores_dir, "dev", languages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["text", "language"])
        writer.writeheader()
        for item in items:
            writer.writerow({"text": item["text"], "language": item["gold"]})
    print(f"Wrote {len(items)} dev items for {len(languages)} languages to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
