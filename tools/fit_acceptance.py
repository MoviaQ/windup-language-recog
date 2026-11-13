"""Fit a release acceptance policy ("und" gate) for a single-network checkpoint.

The output checkpoint is the base checkpoint plus ``calibration`` in the
schema ``LanguageDetector.predict_details`` reads, and ``acceptance_fit`` with
the provenance (data files, digests, counts, rule, ablation table).

Split policy (enforced, not documented only):
  * Fitting data may be FLORES-200 ``dev``, in-corpus validation CSVs, public
    *validation* CSVs and negative (non-label) texts that must be rejected.
  * FLORES ``devtest`` and every file whose name says ``test`` are reporting
    only. They are refused as fitting inputs, and any fitting text whose
    ``text_key`` also occurs in a reporting set is refused (or, with
    ``--drop-reporting-overlap``, removed and counted in the provenance).

Method:
  1. Logits of the base checkpoint for every fitting text (torch API, the same
     forward pass predict_details uses).
  2. Temperature by NLL on the labelled fitting texts.
  3. Every threshold maximises ``correct - lambda * wrong`` over the texts it
     accepts (negatives always count as wrong, weighted so the whole negative
     set weighs as much as one average label). The marginal accepted text then
     has precision of at least lambda / (1 + lambda).
  4. Candidate policies (components) are compared by two-fold cross-fitting:
     fit on one half of the fitting data, score the objective on the other
     half, and swap. Halves are formed by FLORES sentence line (all prefixes
     and all translations of a sentence stay together) or by text key. The
     candidate with the highest summed held-out objective wins; a component
     is kept only if it raises that objective. The winner is refitted on all
     fitting data.
  5. The written checkpoint is reloaded and predict_details is compared with
     the vectorised simulation on a sample; any disagreement that is not a
     float tie at a threshold aborts the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keys LanguageDetector.predict_details reads, plus the human-readable rule
# string that v21 also stores. Nothing else is written into ``calibration``.
POLICY_KEYS = frozenset({
    "temperature",
    "min_confidence",
    "min_confidence_by_language",
    "min_confidence_by_language_and_length",
    "min_margin_by_length",
})
CALIBRATION_KEYS = POLICY_KEYS | {"rule"}
LENGTH_GROUPS = ("short", "long")
# A threshold no softmax value can reach: the cell rejects every input.
REJECT_ALL = 2.0
REPORTING_NAME = re.compile(r"(^|[^a-z])(dev)?test([^a-z]|$)", re.IGNORECASE)

CANDIDATES: dict[str, tuple[str, ...]] = {
    "none": (),
    "global": ("global",),
    "global+margin": ("global", "margin"),
    "length": ("global", "length"),
    "length+margin": ("global", "length", "margin"),
    "language": ("global", "language"),
    "language+length": ("global", "language", "language_length"),
    "language+length+margin": ("global", "language", "language_length", "margin"),
}


class SplitPolicyError(ValueError):
    """Raised when reporting-only data would reach the fit."""


# --- data ---------------------------------------------------------------------

@dataclass
class FitData:
    texts: list[str] = field(default_factory=list)
    gold: list[str | None] = field(default_factory=list)  # None: negative, must be rejected
    group: list[str] = field(default_factory=list)  # cross-fit unit
    source: list[str] = field(default_factory=list)

    def extend(self, texts, gold, group, source):
        self.texts += texts
        self.gold += gold
        self.group += group
        self.source += [source] * len(texts)

    def subset(self, keep: Iterable[int]) -> "FitData":
        keep = list(keep)
        return FitData([self.texts[i] for i in keep], [self.gold[i] for i in keep],
                       [self.group[i] for i in keep], [self.source[i] for i in keep])


def assert_fitting_path(path: Path) -> None:
    """Refuse reporting-only files by name (devtest, test, *_test.csv)."""
    for part in path.resolve().parts[-2:]:
        if REPORTING_NAME.search(part):
            raise SplitPolicyError(f"{path}: name marks a reporting-only split; never fit on it")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_unit(key: str) -> float:
    """Deterministic number in [0, 1) for a key."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / 2**64


def read_labeled_csv(path: Path) -> list[dict[str, str]]:
    import csv

    csv.field_size_limit(10_000_000)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"text", "language"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: needs text and language columns")
        return list(reader)


def load_csv_source(path: Path, languages: set[str]) -> tuple[list[str], list[str], list[str]]:
    assert_fitting_path(path)
    from dataset_utils import text_key

    rows = read_labeled_csv(path)
    unknown = sorted({row["language"] for row in rows} - languages)
    if unknown:
        raise ValueError(f"{path}: labels outside the checkpoint: {unknown[:10]}")
    texts = [row["text"] for row in rows]
    return texts, ["key:" + text_key(text) for text in texts], [row["language"] for row in rows]


def load_negatives(path: Path, languages: set[str], share: float) -> tuple[list[str], list[str], dict]:
    """Negative texts; with share < 1 only the fitting part (stable hash < share)."""
    assert_fitting_path(path)
    from dataset_utils import text_key

    rows = read_labeled_csv(path)
    labels = {row["language"] for row in rows}
    if labels & languages:
        raise ValueError(f"{path}: negatives carry model labels {sorted(labels & languages)}")
    keys = [text_key(row["text"]) for row in rows]
    kept = [i for i, key in enumerate(keys) if negative_partition(key) < share]
    return ([rows[i]["text"] for i in kept], ["key:" + keys[i] for i in kept],
            {"rows": len(rows), "fitting_rows": len(kept), "share": share})


def negative_partition(key: str) -> float:
    """Fitting negatives are those below the share; the rest are reporting-only."""
    return stable_unit("negatives:" + key)


def load_flores(flores_dir: Path, split: str, codes: list[str]):
    """FLORES items through tools/evaluate_flores.py, so the item set is identical."""
    from tools.evaluate_flores import build_items, check_coverage

    missing = check_coverage(flores_dir, split, codes)
    if missing:
        raise ValueError(f"FLORES {split} lacks {len(missing)} languages: {sorted(missing)[:10]}")
    return build_items(flores_dir, split, codes)


def flores_fit_items(flores_dir: Path, split: str, codes: list[str]):
    if split != "dev":
        raise SplitPolicyError(f"FLORES {split!r} is reporting-only; fitting uses dev")
    items = load_flores(flores_dir, "dev", codes)
    texts = [item["text"] for item in items]
    # Translations share a line number; keep a sentence and its translations together.
    groups = ["flores-line:" + item["id"].split(":")[1] for item in items]
    return texts, groups, [item["gold"] for item in items]


def flores_negative_items(flores_dir: Path, split: str, flores_code: str):
    """Full sentences and prefixes of a non-label FLORES file (e.g. nno_Latn)."""
    if split != "dev":
        raise SplitPolicyError(f"FLORES {split!r} is reporting-only; fitting uses dev")
    from tools.evaluate_flores import TRUNCATION_CAPS, read_lines, split_file, truncate_prefix

    texts, groups = [], []
    for line_number, text in enumerate(read_lines(split_file(flores_dir, split, flores_code), split)):
        seen = {text}
        texts.append(text)
        groups.append(f"flores-line:{line_number}")
        for cap in TRUNCATION_CAPS:
            prefix = truncate_prefix(text, cap)
            if prefix and prefix not in seen:
                seen.add(prefix)
                texts.append(prefix)
                groups.append(f"flores-line:{line_number}")
    return texts, groups


def reporting_keys(flores_dir: Path | None, codes: list[str], negative_codes: list[str],
                   reporting_csvs: list[Path]) -> tuple[set[str], list[str]]:
    from dataset_utils import text_key

    keys: set[str] = set()
    names = []
    if flores_dir is not None:
        for item in load_flores(flores_dir, "devtest", codes):
            keys.add(text_key(item["text"]))
        names.append("FLORES devtest (all model codes)")
        from tools.evaluate_flores import read_lines, split_file

        for code in negative_codes:
            for text in read_lines(split_file(flores_dir, "devtest", code), "devtest"):
                keys.add(text_key(text))
            names.append(f"FLORES devtest {code}")
    for path in reporting_csvs:
        for row in read_labeled_csv(path):
            keys.add(text_key(row["text"]))
        names.append(str(path))
    keys.discard("")
    return keys, names


# --- model outputs ----------------------------------------------------------------

_DETECTOR = None


def _init_worker(model_path: str) -> None:
    global _DETECTOR
    from language_detector import LanguageDetector

    _DETECTOR = LanguageDetector(model_path)


def _logits_chunk(texts: list[str]) -> tuple[np.ndarray, list[str | None]]:
    """Logits as predict_details computes them (encode -> mean-pooled bags -> numpy model).

    A text that encode() refuses (no letters) gets NaN logits and its error
    message; predict_details raises on such texts, so no threshold applies.
    """
    from language_detector import encode

    detector = _DETECTOR
    out = np.full((len(texts), len(detector.languages)), np.nan, dtype=np.float32)
    errors: list[str | None] = [None] * len(texts)
    ids: list[int] = []
    offsets: list[int] = []
    valid = []
    for i, text in enumerate(texts):
        try:
            features = encode(text, **detector.encoding_options)
        except ValueError as error:
            errors[i] = f"{type(error).__name__}: {error}"
            continue
        offsets.append(len(ids))
        ids.extend(features)
        valid.append(i)
    if valid:
        out[valid] = detector.model.logits(np.asarray(ids, dtype=np.int64), np.asarray(offsets, dtype=np.int64))
    return out, errors


def compute_logits(model_path: Path, texts: list[str], workers: int, chunk: int = 256):
    chunks = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    if workers <= 1:
        _init_worker(str(model_path))
        results = [_logits_chunk(c) for c in chunks]
    else:
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        with context.Pool(workers, initializer=_init_worker, initargs=(str(model_path),)) as pool:
            results = pool.map(_logits_chunk, chunks, chunksize=4)
    logits = np.concatenate([r[0] for r in results]) if results else np.zeros((0, 0), np.float32)
    errors = [e for r in results for e in r[1]]
    return logits, errors


def cached_logits(model_path: Path, model_sha: str, texts: list[str], workers: int,
                  cache_dir: Path | None):
    digest = hashlib.sha256(model_sha.encode())
    digest.update(b"numpy-runtime-logits-v1")  # torch-era caches differ in the last bits
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    key = digest.hexdigest()[:24]
    path = cache_dir / f"logits-{key}.npz" if cache_dir else None
    if path is not None and path.is_file():
        stored = np.load(path, allow_pickle=False)
        return stored["logits"], [e if e else None for e in stored["errors"].tolist()]
    logits, errors = compute_logits(model_path, texts, workers)
    if path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(path, logits=logits, errors=np.array([e or "" for e in errors]))
    return logits, errors


def evidence_profiles(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """evidence_profile per text: length group (0 short, 1 long) and single-Latin-word flag."""
    from language_detector import evidence_profile

    profiles = [evidence_profile(t) for t in texts]
    return (np.array([LENGTH_GROUPS.index(p[0]) for p in profiles], dtype=np.int8),
            np.array([p[1] is not None for p in profiles], dtype=bool))


def softmax_stats(logits: np.ndarray, temperature: float):
    """best index, top-1 probability and top1-top2 margin at a temperature.

    Same float32 arithmetic as LanguageDetector.predict_details (numpy_runtime.softmax).
    """
    import numpy_runtime

    probabilities = numpy_runtime.softmax(logits.astype(np.float32) / temperature)
    best = probabilities.argmax(axis=1)
    top = np.sort(probabilities, axis=1)[:, -2:]
    return best.astype(np.int64), top[:, 1].astype(np.float64), (top[:, 1] - top[:, 0]).astype(np.float64)


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    import torch

    x = torch.from_numpy(logits)
    y = torch.from_numpy(targets)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(x / log_t.exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    value = float(log_t.exp().item())
    if not (0.05 <= value <= 20.0) or not math.isfinite(value):
        raise ValueError(f"temperature fit diverged: {value}")
    return value


# --- policy fitting ---------------------------------------------------------------

def best_threshold(scores: np.ndarray, gains: np.ndarray) -> float:
    """Threshold t maximising sum(gains[scores >= t]); REJECT_ALL if accepting nothing is best.

    Cuts are only placed between distinct score values, so ties are always
    accepted or rejected together, as ``>=`` does at inference.
    """
    if len(scores) == 0:
        raise ValueError("cannot fit a threshold on an empty cell")
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_gains = scores[order], gains[order]
    cumulative = np.cumsum(sorted_gains)
    # Valid cut after position i only where the next score differs.
    boundary = np.ones(len(scores), dtype=bool)
    boundary[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    candidates = np.where(boundary, cumulative, -np.inf)
    best = int(np.argmax(candidates))
    if candidates[best] <= 0:
        return REJECT_ALL
    return float(sorted_scores[best])


@dataclass
class Arrays:
    best: np.ndarray
    confidence: np.ndarray
    margin: np.ndarray
    length: np.ndarray
    correct: np.ndarray
    weight: np.ndarray


def gains(arrays: Arrays, lam: float) -> np.ndarray:
    return np.where(arrays.correct, arrays.weight, -lam * arrays.weight)


def simulate(policy: dict[str, Any], languages: list[str], arrays: Arrays) -> np.ndarray:
    """Accepted mask; mirrors LanguageDetector.predict_details for single networks
    without lexical tables (agreement is always true for one network)."""
    base = policy.get("min_confidence")
    threshold = np.full(len(arrays.best), -np.inf if base is None else base)
    by_language = policy.get("min_confidence_by_language", {})
    contextual = policy.get("min_confidence_by_language_and_length", {})
    floors = policy.get("min_margin_by_length", {})
    for index, language in enumerate(languages):
        mask = arrays.best == index
        if language in by_language:
            threshold[mask] = by_language[language]
        for g, group in enumerate(LENGTH_GROUPS):
            if group in contextual.get(language, {}):
                threshold[mask & (arrays.length == g)] = contextual[language][group]
    accepted = arrays.confidence >= threshold
    if floors:
        for g, group in enumerate(LENGTH_GROUPS):
            if group in floors:
                accepted &= ~((arrays.length == g) & (arrays.margin < floors[group]))
    return accepted


def fit_policy(components: tuple[str, ...], languages: list[str], arrays: Arrays,
               lam: float, min_samples: int, rounds: int = 3) -> dict[str, Any]:
    """Fit thresholds for the named components; coordinate ascent between
    confidence thresholds and margin floors."""
    policy: dict[str, Any] = {}
    if not components:
        return policy
    gain = gains(arrays, lam)
    margin_pass = np.ones(len(gain), dtype=bool)
    for _ in range(rounds if "margin" in components else 1):
        policy = {k: v for k, v in policy.items() if k == "min_margin_by_length"}
        rows = margin_pass
        policy["min_confidence"] = best_threshold(arrays.confidence[rows], gain[rows])
        if "length" in components:
            cells = {}
            for g, group in enumerate(LENGTH_GROUPS):
                mask = rows & (arrays.length == g)
                if mask.sum() >= min_samples:
                    value = best_threshold(arrays.confidence[mask], gain[mask])
                    cells[group] = value
            policy["min_confidence_by_language_and_length"] = {
                language: dict(cells) for language in languages} if cells else {}
        if "language" in components:
            by_language = {}
            for index, language in enumerate(languages):
                mask = rows & (arrays.best == index)
                if mask.sum() >= min_samples:
                    by_language[language] = best_threshold(arrays.confidence[mask], gain[mask])
            policy["min_confidence_by_language"] = by_language
        if "language_length" in components:
            contextual: dict[str, dict[str, float]] = {}
            for index, language in enumerate(languages):
                for g, group in enumerate(LENGTH_GROUPS):
                    mask = rows & (arrays.best == index) & (arrays.length == g)
                    if mask.sum() >= min_samples:
                        contextual.setdefault(language, {})[group] = best_threshold(
                            arrays.confidence[mask], gain[mask])
            policy["min_confidence_by_language_and_length"] = contextual
        if "margin" in components:
            no_floor = {k: v for k, v in policy.items() if k != "min_margin_by_length"}
            confident = simulate(no_floor, languages, arrays)
            floors = {}
            for g, group in enumerate(LENGTH_GROUPS):
                mask = confident & (arrays.length == g)
                if mask.sum() >= min_samples:
                    floors[group] = best_threshold(arrays.margin[mask], gain[mask])
            policy["min_margin_by_length"] = floors
            margin_pass = np.ones(len(gain), dtype=bool)
            for g, group in enumerate(LENGTH_GROUPS):
                if group in floors:
                    margin_pass &= ~((arrays.length == g) & (arrays.margin < floors[group]))
    return {k: v for k, v in policy.items() if v != {}}


def objective(accepted: np.ndarray, arrays: Arrays, lam: float) -> float:
    return float(gains(arrays, lam)[accepted].sum())


def metrics(accepted: np.ndarray, arrays: Arrays, positive: np.ndarray) -> dict[str, Any]:
    pos = positive
    taken = accepted & pos
    correct = int((taken & arrays.correct).sum())
    total = int(pos.sum())
    neg = ~pos
    return {
        "labelled": total,
        "accepted": int(taken.sum()),
        "accepted_correct": correct,
        "accepted_errors": int(taken.sum()) - correct,
        "coverage": taken.sum() / total if total else None,
        "accepted_precision": correct / taken.sum() if taken.sum() else None,
        "raw_accuracy": int((arrays.correct & pos).sum()) / total if total else None,
        "negatives": int(neg.sum()),
        "negatives_accepted": int((accepted & neg).sum()),
    }


# --- driver -----------------------------------------------------------------

def build_arrays(logits, temperature, data: FitData, languages, lengths, negative_weight) -> Arrays:
    best, confidence, margin = softmax_stats(logits, temperature)
    index = {language: i for i, language in enumerate(languages)}
    gold = np.array([index[g] if g is not None else -1 for g in data.gold])
    weight = np.where(gold >= 0, 1.0, negative_weight)
    return Arrays(best, confidence, margin, lengths, best == gold, weight)


def negative_weight_for(data: FitData, n_labels: int, scale: float) -> float:
    """Per-text weight so that the whole negative set weighs ``scale`` average labels."""
    positives = sum(g is not None for g in data.gold)
    negatives = len(data.gold) - positives
    return scale * (positives / n_labels) / negatives if negatives else 0.0


def per_language_coverage(accepted: np.ndarray, arrays: Arrays, gold: list[str | None],
                          languages: list[str]) -> dict[str, float]:
    """Share of each gold language's labelled texts that are accepted (correct or not)."""
    index = {language: i for i, language in enumerate(languages)}
    gold_index = np.array([index[g] if g is not None else -1 for g in gold])
    result = {}
    for i, language in enumerate(languages):
        mask = gold_index == i
        if mask.any():
            result[language] = round(float((accepted & mask).sum() / mask.sum()), 5)
    return result


def cross_fit(data: FitData, logits, lengths, languages, lam, min_samples, negative_scale, folds=2):
    """Two-fold held-out objective per candidate; temperature refitted per fold."""
    index = {language: i for i, language in enumerate(languages)}
    fold_of = np.array([int(stable_unit("fold:" + g) * folds) for g in data.group])
    table = {name: {"objective": 0.0, "folds": []} for name in CANDIDATES}
    for k in range(folds):
        fit_rows, check_rows = np.where(fold_of != k)[0], np.where(fold_of == k)[0]
        fit_data, check_data = data.subset(fit_rows), data.subset(check_rows)
        positives = [i for i, g in enumerate(fit_data.gold) if g is not None]
        temperature = fit_temperature(logits[fit_rows][positives],
                                      np.array([index[fit_data.gold[i]] for i in positives]))
        fit_arrays = build_arrays(logits[fit_rows], temperature, fit_data, languages, lengths[fit_rows],
                                  negative_weight_for(fit_data, len(languages), negative_scale))
        check_arrays = build_arrays(logits[check_rows], temperature, check_data, languages, lengths[check_rows],
                                    negative_weight_for(check_data, len(languages), negative_scale))
        check_positive = np.array([g is not None for g in check_data.gold])
        for name, components in CANDIDATES.items():
            policy = fit_policy(components, languages, fit_arrays, lam, min_samples)
            accepted = simulate(policy, languages, check_arrays)
            value = objective(accepted, check_arrays, lam)
            table[name]["objective"] += value
            table[name]["folds"].append({"fold": k, "temperature": temperature, "held_out_objective": value,
                                         **metrics(accepted, check_arrays, check_positive),
                                         "per_language_coverage": per_language_coverage(
                                             accepted, check_arrays, check_data.gold, languages)})
    return table


def select_candidate(table: dict[str, Any]) -> str:
    """Highest held-out objective; on equal objective the simpler candidate (fewer components)."""
    return max(CANDIDATES, key=lambda name: (table[name]["objective"], -len(CANDIDATES[name])))


def describe_rule(selected: str, lam: float, min_samples: int, sources: list[dict], negative_weight: float,
                  negative_scale: float) -> str:
    names = ", ".join(f"{s['name']} ({s['used_rows']} rows)" for s in sources)
    return (
        f"Single-network acceptance policy '{selected}' ({' + '.join(CANDIDATES[selected]) or 'no gate'}); "
        f"temperature by NLL and every threshold by max(correct - {lam:g} x wrong) over accepted fitting texts; "
        f"cells with fewer than {min_samples} texts fall back to the coarser threshold; negatives count as wrong "
        f"with weight {negative_weight:.4g} each (whole negative set = {negative_scale:g} x one average label); "
        f"components chosen by two-fold cross-fitted objective. Fitting data only: {names}. "
        "No devtest or test data used for fitting or selection."
    )


def git_state() -> dict[str, Any]:
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": run("rev-parse", "HEAD"), "dirty_files": run("status", "--porcelain").splitlines()}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def verify_checkpoint(path: Path, policy: dict[str, Any], languages: list[str], texts: list[str],
                      arrays: Arrays, sample: int, tolerance: float = 1e-5) -> dict[str, int]:
    """Reload with LanguageDetector; predict_details must agree with the simulation."""
    from language_detector import LanguageDetector

    detector = LanguageDetector(path)
    if detector.languages != languages:
        raise ValueError("written checkpoint changed the language order")
    rng = np.random.default_rng(0)
    picks = rng.choice(len(texts), size=min(sample, len(texts)), replace=False)
    expected = simulate(policy, languages, arrays)
    floors = policy.get("min_margin_by_length", {})
    checked = ties = 0
    for i in picks.tolist():
        details = detector.predict_details(texts[i])
        if details["best_language"] != languages[arrays.best[i]]:
            if details["margin"] < tolerance:  # top-2 tie: batch and single passes may order it differently
                ties += 1
                continue
            raise ValueError(f"verification: best language differs for fitting item {i}")
        thresholds = [policy.get("min_confidence")]
        thresholds += [policy.get("min_confidence_by_language", {}).get(details["best_language"])]
        thresholds += list(policy.get("min_confidence_by_language_and_length", {}).get(details["best_language"], {}).values())
        near = any(t is not None and abs(details["confidence"] - t) < tolerance for t in thresholds)
        near |= any(abs(details["margin"] - f) < tolerance for f in floors.values())
        if near:
            ties += 1
            continue
        if (not details["uncertain"]) != bool(expected[i]):
            raise ValueError(f"verification: predict_details disagrees with the fitted simulation on item {i}")
        checked += 1
    return {"checked": checked, "float_ties_skipped": ties}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="uncalibrated single-network checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="calibrated checkpoint to write")
    parser.add_argument("--report", type=Path, required=True, help="JSON provenance and ablation report")
    parser.add_argument("--flores-dir", type=Path, help="flores200_dataset; dev is fitted, devtest is excluded")
    parser.add_argument("--flores-negative", action="append", default=[],
                        help="FLORES file code that is not a label (e.g. nno_Latn); its dev items are negatives")
    parser.add_argument("--csv", action="append", default=[], metavar="NAME=PATH",
                        help="labelled validation CSV (text,language); repeatable")
    parser.add_argument("--negatives", action="append", default=[], metavar="NAME=PATH",
                        help="CSV of non-label texts that must be rejected; repeatable")
    parser.add_argument("--negatives-fit-share", type=float, default=0.5,
                        help="share of each negatives CSV used for fitting (stable hash); the rest is reporting-only")
    parser.add_argument("--reporting-csv", action="append", type=Path, default=[],
                        help="reporting-only CSV whose texts must not be fitted (overlap check)")
    parser.add_argument("--drop-reporting-overlap", action="store_true",
                        help="remove (and count) fitting texts that also occur in reporting data instead of refusing")
    parser.add_argument("--lambda", dest="lam", type=float, default=25.0,
                        help="cost of one accepted error in correct answers (v21 used 25)")
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--negative-scale", type=float, default=1.0,
                        help="weight of the whole negative set in average labels (0: measure negatives only)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, help="optional logits cache (keyed by model and texts)")
    parser.add_argument("--verify-sample", type=int, default=3000)
    args = parser.parse_args(argv)

    import torch

    if args.lam <= 0 or args.min_samples < 1 or not 0 < args.negatives_fit_share <= 1 or args.negative_scale < 0:
        parser.error("--lambda and --min-samples must be positive, --negatives-fit-share in (0, 1], "
                     "--negative-scale non-negative")
    if args.out.resolve() == args.model.resolve():
        parser.error("--out must differ from --model")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    if checkpoint.get("calibration"):
        parser.error(f"{args.model} already has a calibration; fit from an uncalibrated checkpoint")
    ensemble = checkpoint.get("ensemble")
    if ensemble and ensemble.get("require_agreement"):
        parser.error("ensemble agreement gates are not fitted by this tool")
    languages = list(checkpoint["languages"])
    language_set = set(languages)
    model_sha = file_sha256(args.model)

    data = FitData()
    sources = []
    try:
        if args.flores_dir:
            texts, groups, gold = flores_fit_items(args.flores_dir, "dev", sorted(languages))
            data.extend(texts, gold, groups, "flores_dev")
            sources.append({"name": "flores_dev", "kind": "labelled", "path": str(args.flores_dir / "dev"),
                            "rows": len(texts)})
            for code in args.flores_negative:
                texts, groups = flores_negative_items(args.flores_dir, "dev", code)
                data.extend(texts, [None] * len(texts), groups, f"flores_dev_{code}")
                sources.append({"name": f"flores_dev_{code}", "kind": "negative",
                                "path": str(args.flores_dir / "dev" / f"{code}.dev"), "rows": len(texts)})
        for spec in args.csv:
            name, _, path = spec.partition("=")
            path = Path(path)
            texts, groups, gold = load_csv_source(path, language_set)
            data.extend(texts, gold, groups, name)
            sources.append({"name": name, "kind": "labelled", "path": str(path), "sha256": file_sha256(path),
                            "rows": len(texts)})
        for spec in args.negatives:
            name, _, path = spec.partition("=")
            path = Path(path)
            texts, groups, info = load_negatives(path, language_set, args.negatives_fit_share)
            data.extend(texts, [None] * len(texts), groups, name)
            sources.append({"name": name, "kind": "negative", "path": str(path), "sha256": file_sha256(path),
                            "rows": info["fitting_rows"], "file_rows": info["rows"],
                            "fit_share": info["share"],
                            "partition": "stable_unit('negatives:' + text_key) < fit_share"})
        for path in args.reporting_csv:
            if not path.is_file():
                raise ValueError(f"reporting CSV {path} not found")
        if not data.texts:
            raise ValueError("no fitting data given")
        report_keys, report_names = reporting_keys(
            args.flores_dir, sorted(languages), args.flores_negative, args.reporting_csv)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    from dataset_utils import text_key

    overlap = [i for i, text in enumerate(data.texts) if text_key(text) in report_keys]
    overlap_by_source: dict[str, int] = {}
    for i in overlap:
        overlap_by_source[data.source[i]] = overlap_by_source.get(data.source[i], 0) + 1
    if overlap and not args.drop_reporting_overlap:
        print(f"Error: {len(overlap)} fitting texts also occur in reporting-only data {overlap_by_source}; "
              "pass --drop-reporting-overlap to remove them", file=sys.stderr)
        return 2
    if overlap:
        dropped = set(overlap)
        data = data.subset(i for i in range(len(data.texts)) if i not in dropped)
        print(f"dropped {len(overlap)} fitting texts that occur in reporting data: {overlap_by_source}")

    print(f"fitting texts: {len(data.texts)}; computing logits with {args.workers} workers", flush=True)
    logits, errors = cached_logits(args.model, model_sha, data.texts, args.workers, args.cache_dir)
    unencodable = [i for i, e in enumerate(errors) if e]
    unencodable_by_source: dict[str, int] = {}
    for i in unencodable:
        unencodable_by_source[data.source[i]] = unencodable_by_source.get(data.source[i], 0) + 1
    if unencodable:
        # predict_details raises on these texts, so no threshold can apply to them.
        print(f"{len(unencodable)} fitting texts raise in encode (no letters) and are excluded: "
              f"{unencodable_by_source}; first: {errors[unencodable[0]]}")
        keep = [i for i in range(len(data.texts)) if not errors[i]]
        data, logits = data.subset(keep), logits[keep]
    lengths, single_word = evidence_profiles(data.texts)
    for source in sources:
        source["used_rows"] = sum(1 for s in data.source if s == source["name"])

    table = cross_fit(data, logits, lengths, languages, args.lam, args.min_samples, args.negative_scale)
    selected = select_candidate(table)
    print("| candidate | held-out objective | coverage | accepted precision | accepted errors | negatives accepted |")
    for name, entry in table.items():
        acc = sum(f["accepted"] for f in entry["folds"])
        cor = sum(f["accepted_correct"] for f in entry["folds"])
        lab = sum(f["labelled"] for f in entry["folds"])
        neg = sum(f["negatives_accepted"] for f in entry["folds"])
        negn = sum(f["negatives"] for f in entry["folds"])
        mark = " (selected)" if name == selected else ""
        print(f"| {name}{mark} | {entry['objective']:.1f} | {100 * acc / lab:.2f} | "
              f"{100 * cor / acc if acc else 0:.3f} | {acc - cor} | {neg}/{negn} |")

    index = {language: i for i, language in enumerate(languages)}
    positives = [i for i, g in enumerate(data.gold) if g is not None]
    temperature = fit_temperature(logits[positives], np.array([index[data.gold[i]] for i in positives]))
    negative_weight = negative_weight_for(data, len(languages), args.negative_scale)
    arrays = build_arrays(logits, temperature, data, languages, lengths, negative_weight)
    policy = fit_policy(CANDIDATES[selected], languages, arrays, args.lam, args.min_samples)
    calibration = {"temperature": temperature, **policy}
    calibration["rule"] = describe_rule(selected, args.lam, args.min_samples, sources, negative_weight,
                                        args.negative_scale)
    if set(calibration) - CALIBRATION_KEYS:
        raise AssertionError(f"unexpected calibration keys {set(calibration) - CALIBRATION_KEYS}")
    is_positive = np.array([g is not None for g in data.gold])
    accepted = simulate(policy, languages, arrays)
    fitted = {"all": metrics(accepted, arrays, is_positive)}
    for source in sources:
        rows = np.array([s == source["name"] for s in data.source])
        sub = Arrays(*(getattr(arrays, f)[rows] for f in ("best", "confidence", "margin", "length", "correct", "weight")))
        fitted[source["name"]] = metrics(accepted[rows], sub, is_positive[rows])

    # Lexical single-word tables (ambiguous_single_words, single_word_language)
    # are not fitted: report how much fitting evidence exists for them.
    single_word_evidence = metrics(accepted[single_word], Arrays(*(getattr(arrays, f)[single_word] for f in (
        "best", "confidence", "margin", "length", "correct", "weight"))), is_positive[single_word])
    provenance = {
        "tool": "tools/fit_acceptance.py",
        "git": git_state(),
        "base_model": str(args.model),
        "base_model_sha256": model_sha,
        "sources": sources,
        "reporting_sets_excluded": report_names,
        "reporting_overlap_dropped": overlap_by_source,
        "unencodable_excluded": unencodable_by_source,
        "lambda": args.lam,
        "min_samples": args.min_samples,
        "negative_weight": negative_weight,
        "negative_scale": args.negative_scale,
        "per_language_coverage_in_sample": per_language_coverage(accepted, arrays, data.gold, languages),
        "negatives_fit_share": args.negatives_fit_share,
        "selected": selected,
        "components": list(CANDIDATES[selected]),
        "cross_fit": table,
        "fitted_data_metrics_in_sample": fitted,
        "single_latin_word_texts_in_sample": single_word_evidence,
    }
    output = dict(checkpoint)
    output["calibration"] = calibration
    output["acceptance_fit"] = json.loads(json.dumps(provenance))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    torch.save(output, temporary)
    try:
        verification = verify_checkpoint(temporary, policy | {"temperature": temperature}, languages,
                                         data.texts, arrays, args.verify_sample)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    output["acceptance_fit"]["verification"] = verification
    torch.save(output, temporary)
    os.replace(temporary, args.out)
    report = {"calibration": calibration, **output["acceptance_fit"], "output": str(args.out),
              "output_sha256": file_sha256(args.out)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"temperature {temperature:.4f}; selected {selected}; verification {verification}")
    print(f"in-sample fitting metrics: {json.dumps(fitted['all'])}")
    print(f"wrote {args.out} and {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
