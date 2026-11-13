"""Write top-k fastText (lid.176) or NLLB (lid218e) predictions for FLORES-200 items.

Runs in any environment with fastText bindings (see requirements-benchmark.txt);
it never imports torch. The output feeds
``tools/evaluate_flores.py --fasttext-predictions``, which owns the label
mapping. Items include the --extra-variants files when they exist, so one
predictions file serves runs with and without that flag.

    python tools/predict_fasttext.py --model lid.176.bin --name ft176 \
        --flores-dir flores200_dataset --split devtest --output reports/ft176_devtest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_flores import (
    EXTRA_VARIANTS,
    SPLITS,
    build_items,
    check_coverage,
    file_sha256,
    load_languages,
    map_external_label,
    split_file,
    text_digest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="fastText .bin (lid.176.bin, lid218e.bin)")
    parser.add_argument("--name", help="system name in reports (default: model file stem)")
    parser.add_argument("--flores-dir", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, default="devtest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10,
                        help="labels kept per item; extended per item until one maps to our codes")
    parser.add_argument("--limit-per-language", type=int)
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    import fasttext

    languages = load_languages()
    language_set = frozenset(languages)
    missing = check_coverage(args.flores_dir, args.split, languages)
    codes = [code for code in languages if code not in missing]
    variants = all(split_file(args.flores_dir, args.split, f).is_file() for f in EXTRA_VARIANTS)
    items = build_items(args.flores_dir, args.split, codes, variants=variants,
                        limit_per_language=args.limit_per_language)
    model = fasttext.load_model(str(args.model))
    labels = list(model.get_labels())
    meta = {
        "name": args.name or args.model.stem,
        "model": str(args.model),
        "model_sha256": file_sha256(args.model),
        "split": args.split,
        "top_k": args.top_k,
        "limit_per_language": args.limit_per_language,
        "variants_included": variants,
        "missing_languages": missing,
        "items": len(items),
        "fasttext_version": getattr(fasttext, "__version__", "unknown"),
        "labels": labels,
    }
    started = time.time()
    extended = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"meta": meta}) + "\n")
        for item in items:
            # fastText rejects newlines; FLORES lines are stripped single lines.
            # The list form is NumPy-2 safe (see requirements-benchmark.txt).
            text = item["text"].replace("\n", " ")
            predicted, probabilities = model.predict([text], k=args.top_k)
            labels_k, probs_k = list(predicted[0]), list(probabilities[0])
            if not any(map_external_label(label, language_set) for label in labels_k):
                # Keep restricted argmax exact: extend this item's list up to
                # its first label that maps to one of our codes.
                extended += 1
                predicted, probabilities = model.predict([text], k=-1)
                labels_k, probs_k = list(predicted[0]), list(probabilities[0])
                last = next((i for i, l in enumerate(labels_k) if map_external_label(l, language_set)), len(labels_k) - 1)
                labels_k, probs_k = labels_k[: last + 1], probs_k[: last + 1]
            stream.write(json.dumps({
                "id": item["id"],
                "sha": text_digest(item["text"]),
                "labels": labels_k,
                "probs": [round(float(p), 6) for p in probs_k],
            }) + "\n")
    print(f"{meta['name']}: {len(items)} items in {time.time() - started:.1f}s; "
          f"{len(labels)} labels; {extended} items extended beyond top-{args.top_k}; missing languages {sorted(missing)} -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
