"""Report acceptance (coverage, accepted precision, false acceptances) on reporting-only data.

This tool fits nothing. It runs ``LanguageDetector.predict_details`` for one
or more checkpoints on the same items and reports, per model:

  * labelled CSVs (e.g. the public ``test_short.csv``): raw accuracy, coverage,
    accepted precision, accepted errors and accepted macro FPR, overall and by
    evidence_profile length group;
  * negative (non-label) texts: how many are accepted, and as which label.
    ``--negatives`` keeps only the reporting part of a negatives CSV (the
    complement of tools/fit_acceptance.py's fitting share); ``--flores-negative``
    uses FLORES devtest items of a non-label file such as nno_Latn;
  * the authored ticket fixture: whole-ticket policy vote, the guarded ticket
    API and ``predict_ticket(allow_uncertain=True)``;
  * FLORES devtest predictions written by tools/evaluate_flores.py
    ``--predictions-out``: accepted macro FPR and accepted errors per slice.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_flores import (
    TRUNCATION_CAPS, build_items, file_sha256, git_state, load_languages, read_lines, score,
    slice_items, split_file, truncate_prefix,
)
from tools.fit_acceptance import negative_partition, read_labeled_csv

_DETECTOR = None


def _init(model_path: str) -> None:
    global _DETECTOR
    import torch

    torch.set_num_threads(1)
    from language_detector import LanguageDetector

    _DETECTOR = LanguageDetector(model_path)


def _details(text: str) -> dict[str, Any]:
    try:
        d = _DETECTOR.predict_details(text)
    except ValueError as error:  # letterless text: recorded, counted as abstention
        return {"error": str(error)}
    return {"raw": d["best_language"], "accepted": None if d["uncertain"] else d["language"]}


def run_details(model: Path, texts: list[str], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        _init(str(model))
        return [_details(t) for t in texts]
    import multiprocessing

    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init, initargs=(str(model),)) as pool:
        return pool.map(_details, texts, chunksize=512)


def labelled_report(golds: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = [r.get("raw") or "<error>" for r in rows]
    accepted = [r.get("accepted") for r in rows]
    taken = [(g, a) for g, a in zip(golds, accepted) if a is not None]
    correct = sum(g == a for g, a in taken)
    gated = score(golds, [a if a is not None else "und" for a in accepted])
    return {
        "texts": len(golds),
        "raw_accuracy": score(golds, raw)["accuracy"],
        "coverage": len(taken) / len(golds),
        "accepted": len(taken),
        "accepted_correct": correct,
        "accepted_errors": len(taken) - correct,
        "accepted_precision": correct / len(taken) if taken else None,
        "accepted_macro_fpr": gated["macro_fpr"],
        "errors_raised": sum("error" in r for r in rows),
        "top_accepted_confusions": [[g, a, c] for (g, a), c in Counter(
            (g, a) for g, a in taken if g != a).most_common(10)],
    }


def negative_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = Counter(r["accepted"] for r in rows if r.get("accepted"))
    raw = Counter(r.get("raw") for r in rows if r.get("raw"))
    return {"texts": len(rows), "accepted": sum(accepted.values()),
            "accepted_rate": sum(accepted.values()) / len(rows) if rows else None,
            "accepted_as": dict(accepted.most_common(8)), "raw_top1": dict(raw.most_common(5))}


def length_groups(texts: list[str]) -> list[str]:
    from language_detector import evidence_profile

    return [evidence_profile(t)[0] for t in texts]


def flores_negative_texts(flores_dir: Path, split: str, code: str) -> list[str]:
    texts = []
    for text in read_lines(split_file(flores_dir, split, code), split):
        texts.append(text)
        seen = {text}
        for cap in TRUNCATION_CAPS:
            prefix = truncate_prefix(text, cap)
            if prefix and prefix not in seen:
                seen.add(prefix)
                texts.append(prefix)
    return texts


def flores_predictions_report(flores_dir: Path, path: Path) -> dict[str, Any]:
    """Per-slice acceptance from an evaluate_flores.py --predictions-out file."""
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows[row["id"]] = row
    items = build_items(flores_dir, "devtest", load_languages(), variants=any(
        r["id"].startswith(("zho_Hant", "prs_Arab", "ckb_Arab")) for r in rows.values()))
    if len(items) != len(rows) or any(item["id"] not in rows for item in items):
        raise ValueError(f"{path}: predictions do not cover the devtest items exactly")
    result = {}
    for name, indices in slice_items(items).items():
        golds = [items[i]["gold"] for i in indices]
        predictions = [rows[items[i]["id"]] for i in indices]
        result[name] = labelled_report(golds, [{"raw": p["ours"], "accepted": p["ours_accepted"]}
                                               for p in predictions])
    return result


def ticket_report(model: Path, fixture: Path) -> dict[str, Any]:
    from language_detector import LanguageDetector
    from ticket_guard import prepare_ticket
    from tools.benchmark_ticket_guard import load_cases, summarize

    detector = LanguageDetector(model)
    cases = load_cases(fixture)
    systems: dict[str, list[str]] = {"main_policy": [], "main_guarded": [], "predict_ticket_allow_uncertain": []}
    for case in cases:
        prepared = prepare_ticket(case["title"], case["description"])
        has_letters = any(c.isalpha() for c in prepared.text)
        systems["main_policy"].append(detector.predict_details(prepared.text)["language"] if has_letters else "und")
        systems["main_guarded"].append(detector.predict_guarded_ticket(case["title"], case["description"]))
        systems["predict_ticket_allow_uncertain"].append(
            detector.predict_ticket(case["title"], case["description"], allow_uncertain=True))
    return {name: summarize(cases, predictions) for name, predictions in systems.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--csv", action="append", default=[], metavar="NAME=PATH", help="labelled reporting CSV")
    parser.add_argument("--negatives", action="append", default=[], metavar="NAME=PATH",
                        help="negatives CSV; only the reporting part (partition >= --negatives-fit-share)")
    parser.add_argument("--negatives-fit-share", type=float, default=0.5)
    parser.add_argument("--flores-dir", type=Path)
    parser.add_argument("--flores-negative", action="append", default=[], help="devtest file code, e.g. nno_Latn")
    parser.add_argument("--flores-predictions", action="append", default=[], metavar="NAME=JSONL",
                        help="evaluate_flores.py --predictions-out file of the model NAME")
    parser.add_argument("--tickets", type=Path, help="authored ticket fixture JSONL")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from dataset_utils import text_key

    models = dict(spec.split("=", 1) for spec in args.model)
    datasets: dict[str, tuple[list[str], list[str] | None]] = {}
    for spec in args.csv:
        name, path = spec.split("=", 1)
        rows = read_labeled_csv(Path(path))
        datasets[name] = ([r["text"] for r in rows], [r["language"] for r in rows])
    for spec in args.negatives:
        name, path = spec.split("=", 1)
        rows = read_labeled_csv(Path(path))
        texts = [r["text"] for r in rows if negative_partition(text_key(r["text"])) >= args.negatives_fit_share]
        datasets[f"{name}_reporting_part"] = (texts, None)
    for code in args.flores_negative:
        if args.flores_dir is None:
            parser.error("--flores-negative needs --flores-dir")
        datasets[f"flores_devtest_{code}"] = (flores_negative_texts(args.flores_dir, "devtest", code), None)

    report: dict[str, Any] = {"meta": {
        "command": [Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])],
        "git": git_state(), "fits_nothing": True,
        "models": {name: {"path": path, "sha256": file_sha256(Path(path))} for name, path in models.items()},
        "datasets": {name: len(texts) for name, (texts, _) in datasets.items()},
    }, "results": {}}
    for name, (texts, golds) in datasets.items():
        groups = length_groups(texts) if golds is not None else None
        report["results"][name] = {}
        for model_name, path in models.items():
            rows = run_details(Path(path), texts, args.workers)
            if golds is None:
                report["results"][name][model_name] = negative_report(rows)
                continue
            entry = {"all": labelled_report(golds, rows)}
            for group in ("short", "long"):
                idx = [i for i, g in enumerate(groups) if g == group]
                if idx:
                    entry[group] = labelled_report([golds[i] for i in idx], [rows[i] for i in idx])
            report["results"][name][model_name] = entry
            print(f"{name} / {model_name}: {json.dumps({k: v for k, v in entry['all'].items() if k != 'top_accepted_confusions'})}", flush=True)
    for spec in args.flores_predictions:
        model_name, path = spec.split("=", 1)
        if args.flores_dir is None:
            parser.error("--flores-predictions needs --flores-dir")
        report["results"].setdefault("flores_devtest", {})[model_name] = flores_predictions_report(args.flores_dir, Path(path))
    if args.tickets:
        report["results"]["authored_tickets"] = {m: ticket_report(Path(p), args.tickets) for m, p in models.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
