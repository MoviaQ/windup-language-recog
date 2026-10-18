"""Offline, non-fitting comparison on synthetic ticket examples.

fastText is an optional benchmark dependency, never a runtime dependency.
Both systems receive the same selected text and the same sentence-veto rule.
The fixture's AI-authored labels are not production evidence.
"""

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from numbers import Real
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ticket_guard import RawPrediction, guarded_language, prepare_ticket

DEFAULT_INPUT = ROOT / "tests/fixtures/authored_tickets.jsonl"
METADATA = ROOT / "tests/fixtures/authored_tickets.metadata.json"
SENTENCE_CUTOFF = 0.65
# fastText adds a small epsilon before taking log probabilities; exp can
# therefore return slightly above one. Normalize only that numerical range.
FT_PROBABILITY_TOLERANCE = 0.0001


def file_digest(path: Path) -> str:
    """Fingerprint local inputs without retaining model files in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Validate JSONL boundaries without disclosing ticket contents."""
    cases, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at line {number}") from error
            keys = ("id", "title", "description", "language")
            if not isinstance(case, dict) or any(
                not isinstance(case.get(key), str) for key in keys
            ):
                raise ValueError(f"invalid ticket fields at line {number}")
            if not case["id"] or not case["language"] or case["id"] in seen:
                raise ValueError(f"empty or duplicate identifier at line {number}")
            seen.add(case["id"])
            cases.append(case)
    if not cases:
        raise ValueError("the evaluation population is empty")
    return cases


def summarize(
    cases: list[dict[str, Any]], predictions: list[str]
) -> dict[str, Any]:
    """C/E/U sum to n; desired abstentions are never accepted-correct."""
    if len(cases) != len(predictions):
        raise ValueError("case/prediction population drift")
    pairs = list(zip(cases, predictions))
    correct = sum(pred != "und" and pred == case["language"] for case, pred in pairs)
    wrong = sum(pred != "und" and pred != case["language"] for case, pred in pairs)
    withheld = predictions.count("und")
    mixed = [(case, pred) for case, pred in pairs if case["language"] == "und"]
    answered = correct + wrong
    return {
        "n": len(cases), "C": correct, "E": wrong, "U": withheld,
        "answered": answered,
        "precision": correct / answered if answered else None,
        "correct_answer_rate": correct / len(cases) if cases else None,
        "mixed_correct_withheld": sum(pred == "und" for _, pred in mixed),
        "mixed_false_acceptances": sum(pred != "und" for _, pred in mixed),
    }


def ft_vote(model: Any, text: str) -> RawPrediction:
    """Native top-1; list binding works with the pinned wheel and NumPy 2."""
    if not any(char.isalpha() for char in text):
        return RawPrediction("und", 0.0)
    labels, scores = model.predict([text], k=1)
    try:
        label, score = labels[0][0], scores[0][0]
    except (IndexError, TypeError) as error:
        raise ValueError("invalid fastText prediction shape") from error
    prefix = "__label__"
    if not isinstance(label, str) or not label.startswith(prefix):
        raise ValueError("invalid fastText label")
    code = label[len(prefix):]
    if not code or code == "und":
        raise ValueError("invalid fastText language code")
    if isinstance(score, bool) or not isinstance(score, Real):
        raise ValueError("invalid fastText probability type")
    if (
        not math.isfinite(float(score))
        or not 0.0 <= score <= 1.0 + FT_PROBABILITY_TOLERANCE
    ):
        raise ValueError("invalid fastText probability")
    return RawPrediction(code, min(float(score), 1.0))


def matched_predictions(rows: list[dict[str, Any]], count: int):
    """Top-N after the same guard, on the original cohort, without gold."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("answer count must be a nonnegative integer")
    eligible = [row for row in rows if row["ft_eligible"]]
    info = {"matched": count <= len(eligible), "eligible": len(eligible),
            "requested": count}
    if not info["matched"]:
        return None, info
    ordered = sorted(eligible, key=lambda row: (-row["ft_score"], row["id"]))
    chosen = {row["id"] for row in ordered[:count]}
    predictions = [
        row["ft_label"] if row["id"] in chosen else "und" for row in rows
    ]
    info.update({"answered": count,
                 "cutoff": ordered[count - 1]["ft_score"] if count else None})
    return predictions, info


def predict_row(detector: Any, fasttext_model: Any, case: dict[str, Any]):
    """Inference only consumes title/description, never the gold language."""
    title, description = case["title"], case["description"]
    prepared = prepare_ticket(title, description)
    whole = ft_vote(fasttext_model, prepared.text)
    segments = {text: ft_vote(fasttext_model, text) for text in prepared.sentences}
    ft_guarded = guarded_language(
        whole.best, prepared.sentences, segments.__getitem__, SENTENCE_CUTOFF
    )
    has_letters = any(char.isalpha() for char in prepared.text)
    main = detector.predict_details(prepared.text)["language"] if has_letters else "und"
    return {
        "main_guarded": detector.predict_guarded_ticket(title, description),
        "main_policy": main,
        "fasttext_065": ft_guarded if whole.confidence >= 0.65 else "und",
        "fasttext_080": ft_guarded if whole.confidence >= 0.80 else "und",
        "ft_label": whole.best, "ft_score": whole.confidence,
        "ft_eligible": ft_guarded != "und",
    }


def dominates(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
    """Never claim that withholding every answer is a win."""
    return (
        candidate["answered"] > 0 and candidate["C"] >= reference["C"]
        and candidate["E"] <= reference["E"]
        and (candidate["C"] > reference["C"] or candidate["E"] < reference["E"])
    )


def cohort_report(cases: list[dict[str, Any]], rows: list[dict[str, Any]]):
    """Every system is evaluated on exactly this cohort, including abstentions."""
    systems = {
        key: summarize(cases, [row[key] for row in rows])
        for key in ("main_guarded", "main_policy", "fasttext_065", "fasttext_080")
    }
    ours = systems["main_guarded"]
    matched, matching = matched_predictions(rows, ours["answered"])
    comparison = summarize(cases, matched) if matched is not None else None
    return {
        "n": len(cases), "systems": systems, "exact_matching": matching,
        "fasttext_matched": comparison,
        "strict_win_primary": dominates(ours, systems["fasttext_065"]),
        "strict_win_matched": dominates(ours, comparison) if comparison else False,
    }


def evaluate_cases(
    detector: Any, ft_model: Any, cases: list[dict[str, Any]], excluded: set[str]
):
    rows = [{"id": case["id"], **predict_row(detector, ft_model, case)} for case in cases]
    report = {"full": cohort_report(cases, rows)}
    indices = [index for index, case in enumerate(cases) if case["id"] not in excluded]
    if excluded:
        clean_cases, clean_rows = [cases[i] for i in indices], [rows[i] for i in indices]
        report["conservative_no_repeated_titles"] = cohort_report(clean_cases, clean_rows)
    report["by_language"] = {}
    for language in sorted({case["language"] for case in cases}):
        indices = [i for i, case in enumerate(cases) if case["language"] == language]
        report["by_language"][language] = {
            key: summarize([cases[i] for i in indices], [rows[i][key] for i in indices])
            for key in ("main_guarded", "fasttext_065")
        }
    return report


def validate_output(args: argparse.Namespace) -> None:
    """A report must never overwrite evaluation data or model weights."""
    if args.out is None:
        return
    protected = {args.input, args.model, args.ft_model, DEFAULT_INPUT, METADATA}
    if args.out.resolve() in {path.resolve() for path in protected}:
        raise ValueError("output cannot replace an input, metadata or model file")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    validate_output(args)
    import fasttext
    import torch
    from language_detector import LanguageDetector

    if args.threads < 1:
        raise ValueError("threads must be positive")
    input_sha = file_digest(args.input)
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if (
        args.input.resolve() == DEFAULT_INPUT.resolve()
        and input_sha != metadata["file"]["sha256"]
    ):
        raise ValueError("the committed synthetic fixture has changed")
    cases = load_cases(args.input)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(args.threads)
        detector = LanguageDetector(args.model)
        supported = set(detector.languages) | {"und"}
        if any(case["language"] not in supported for case in cases):
            raise ValueError("gold language is outside the model's supported scope")
        ft_model = fasttext.load_model(str(args.ft_model))
        excluded = (
            set(metadata["benchmark"]["clean_cohort_ids_excluded"])
            if input_sha == metadata["file"]["sha256"] else set()
        )
        start = time.perf_counter()
        report = evaluate_cases(detector, ft_model, cases, excluded)
        report["evaluation_seconds"] = time.perf_counter() - start
    finally:
        torch.set_num_threads(previous_threads)
    report["provenance"] = {
        "input_sha256": input_sha, "model_sha256": file_digest(args.model),
        "fasttext_sha256": file_digest(args.ft_model), "python": platform.python_version(),
        "torch": str(torch.__version__), "threads": args.threads,
        "sentence_cutoff": SENTENCE_CUTOFF, "fitted_on_input": False,
        "warning": (
            "Small AI-authored fixture, human-unverified; not production evidence."
            if input_sha == metadata["file"]["sha256"]
            else "Caller-supplied labels: verify provenance and annotation quality."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=ROOT / "model.pt")
    parser.add_argument("--ft-model", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        report = run_benchmark(args)
    except ImportError as error:
        parser.exit(
            2, f"Missing optional benchmark dependency: {error}. "
            "Install requirements-benchmark.txt.\n"
        )
    except (ValueError, OSError) as error:
        parser.exit(2, f"Benchmark failed: {error}\n")
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    sys.stdout.write(output)


if __name__ == "__main__":
    main()
