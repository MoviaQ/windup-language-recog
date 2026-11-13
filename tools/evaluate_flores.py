"""Evaluate the bundled checkpoint (and optional fastText/NLLB predictions) on FLORES-200.

FLORES-200 is a public, parallel benchmark (CC BY-SA 4.0) that is never
committed here; see docs/flores-benchmark.md for the download.

Split policy:
  * ``dev``     may be used for checkpoint selection and threshold decisions.
  * ``devtest`` is for reporting only. Never select checkpoints,
    hyperparameters or thresholds on devtest numbers.

This module is importable without torch: our model is loaded lazily, so
tools/predict_fasttext.py can reuse the item builder in a fastText-only env.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES_FILE = ROOT / "data" / "languages.json"
SPLITS = ("dev", "devtest")
# FLORES-200 is sentence-parallel; every file of a split has this many lines.
EXPECTED_LINES = {"dev": 997, "devtest": 1012}
TRUNCATION_CAPS = (20, 50, 100)
LENGTH_BUCKETS = (("len<=20", 0, 20), ("len21-50", 21, 50), ("len51-100", 51, 100), ("len>100", 101, None))
TOP_CONFUSIONS = 20

# Our code -> FLORES-200 file code used as gold. Ambiguous choices:
FLORES_FOR_CODE: dict[str, str] = {
    "af": "afr_Latn", "am": "amh_Ethi",
    "ar": "arb_Arab",  # Modern Standard Arabic; dialect files are not gold
    "as": "asm_Beng",
    "az": "azj_Latn",  # North Azerbaijani (Latin); azb_Arab is South Azerbaijani
    "be": "bel_Cyrl", "bg": "bul_Cyrl", "bn": "ben_Beng", "bo": "bod_Tibt", "bs": "bos_Latn",
    "ca": "cat_Latn", "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn", "de": "deu_Latn",
    "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn",
    "et": "est_Latn", "eu": "eus_Latn",
    "fa": "pes_Arab",  # Iranian Persian; Dari (prs_Arab) only via --extra-variants
    "fi": "fin_Latn", "fr": "fra_Latn", "ga": "gle_Latn", "gl": "glg_Latn",
    "gn": "grn_Latn", "gu": "guj_Gujr", "ha": "hau_Latn", "he": "heb_Hebr", "hi": "hin_Deva",
    "hr": "hrv_Latn", "ht": "hat_Latn", "hu": "hun_Latn", "hy": "hye_Armn", "id": "ind_Latn",
    "ig": "ibo_Latn", "is": "isl_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "jv": "jav_Latn",
    "ka": "kat_Geor", "kk": "kaz_Cyrl", "km": "khm_Khmr", "kn": "kan_Knda", "ko": "kor_Hang",
    "ku": "kmr_Latn",  # Kurmanji (Latin); Sorani (ckb_Arab) only via --extra-variants
    "ky": "kir_Cyrl", "lb": "ltz_Latn", "lo": "lao_Laoo", "lt": "lit_Latn",
    "lv": "lvs_Latn",  # Standard Latvian
    "mg": "plt_Latn",  # Plateau Malagasy, the standard variety
    "mk": "mkd_Cyrl", "ml": "mal_Mlym",
    "mn": "khk_Cyrl",  # Halh Mongolian (Cyrillic)
    "mr": "mar_Deva",
    "ms": "zsm_Latn",  # Standard Malay
    "mt": "mlt_Latn", "my": "mya_Mymr",
    "nb": "nob_Latn",  # Bokmal; nno_Latn (Nynorsk) is not one of our codes
    "ne": "npi_Deva", "nl": "nld_Latn",
    "om": "gaz_Latn",  # West Central Oromo
    "or": "ory_Orya", "pa": "pan_Guru", "pl": "pol_Latn",
    "ps": "pbt_Arab",  # Southern Pashto
    "pt": "por_Latn",
    "qu": "quy_Latn",  # Ayacucho Quechua, the only Quechua file in FLORES-200
    "ro": "ron_Latn", "ru": "rus_Cyrl", "rw": "kin_Latn", "sd": "snd_Arab", "si": "sin_Sinh",
    "sk": "slk_Latn", "sl": "slv_Latn", "sn": "sna_Latn", "so": "som_Latn",
    "sq": "als_Latn",  # Tosk Albanian, the standard variety
    "sr": "srp_Cyrl",  # FLORES has Serbian only in Cyrillic
    "su": "sun_Latn", "sv": "swe_Latn",
    "sw": "swh_Latn",  # Swahili (individual language)
    "ta": "tam_Taml", "te": "tel_Telu", "tg": "tgk_Cyrl", "th": "tha_Thai", "tk": "tuk_Latn",
    "tl": "tgl_Latn", "tn": "tsn_Latn", "tr": "tur_Latn", "tt": "tat_Cyrl", "ug": "uig_Arab",
    "uk": "ukr_Cyrl", "ur": "urd_Arab",
    "uz": "uzn_Latn",  # Northern Uzbek (Latin)
    "vi": "vie_Latn",
    "yi": "ydd_Hebr",  # Eastern Yiddish
    "yo": "yor_Latn",
    "zh": "zho_Hans",  # Simplified; Traditional (zho_Hant) only via --extra-variants
}

# Extra FLORES files whose gold is one of our codes, reported separately.
EXTRA_VARIANTS: dict[str, str] = {"zho_Hant": "zh", "prs_Arab": "fa", "ckb_Arab": "ku"}

# NLLB (lid218e) prediction labels that count as our code, beyond the inverse
# of FLORES_FOR_CODE. Arabic dialect labels collapse to ar because we have one
# Arabic code; every other NLLB label is unmapped (counted wrong when top-1).
NLLB_EXTRA_LABELS: dict[str, str] = {
    "ars_Arab": "ar", "acm_Arab": "ar", "acq_Arab": "ar", "aeb_Arab": "ar",
    "ajp_Arab": "ar", "apc_Arab": "ar", "ary_Arab": "ar", "arz_Arab": "ar",
    **EXTRA_VARIANTS,
}
NLLB_LABELS: dict[str, str] = {flores: code for code, flores in FLORES_FOR_CODE.items()} | NLLB_EXTRA_LABELS

# fastText lid.176 labels are mostly ISO 639-1 and equal our codes. These are
# the only non-identical labels mapped. Deliberately unmapped: sh
# (Serbo-Croatian, cannot be split into bs/hr/sr), nn (Nynorsk), azb, yue, ...
FASTTEXT_EXTRA_LABELS: dict[str, str] = {"no": "nb", "ckb": "ku", "arz": "ar"}

NLLB_LABEL_PATTERN = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")


def load_languages(path: Path = LANGUAGES_FILE) -> list[str]:
    with path.open(encoding="utf-8") as stream:
        return sorted(json.load(stream))


def map_external_label(label: str, languages: set[str] | frozenset[str]) -> str | None:
    """Map a fastText lid.176 or NLLB lid218e label to one of ``languages``, or None."""
    code = label.removeprefix("__label__")
    if NLLB_LABEL_PATTERN.match(code):
        return NLLB_LABELS.get(code)
    if code in FASTTEXT_EXTRA_LABELS:
        return FASTTEXT_EXTRA_LABELS[code]
    return code if code in languages else None


def truncate_prefix(text: str, cap: int) -> str | None:
    """Longest word prefix of at most ``cap`` chars; None if text already fits.

    Scripts without spaces (ja, zh, th, ...) yield a word prefix shorter than
    half the cap; those fall back to a character prefix.
    """
    if len(text) <= cap:
        return None
    prefix = ""
    for word in text.split(" "):
        candidate = f"{prefix} {word}" if prefix else word
        if len(candidate) > cap:
            break
        prefix = candidate
    if len(prefix) < cap / 2:
        prefix = text[:cap].strip()
    return prefix or None


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def split_file(flores_dir: Path, split: str, flores_code: str) -> Path:
    return flores_dir / split / f"{flores_code}.{split}"


def check_coverage(flores_dir: Path, split: str, languages: list[str]) -> dict[str, str]:
    """Return {code: reason} for every language without a mapping or file."""
    missing = {}
    for code in languages:
        flores_code = FLORES_FOR_CODE.get(code)
        if flores_code is None:
            missing[code] = "no FLORES mapping"
        elif not split_file(flores_dir, split, flores_code).is_file():
            missing[code] = f"missing file {split_file(flores_dir, split, flores_code)}"
    return missing


def read_lines(path: Path, split: str) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != EXPECTED_LINES[split]:
        raise ValueError(f"{path}: {len(lines)} lines, expected {EXPECTED_LINES[split]}")
    return [line.strip() for line in lines]


def build_items(
    flores_dir: Path,
    split: str,
    codes: Iterable[str],
    *,
    variants: bool = False,
    limit_per_language: int | None = None,
) -> list[dict[str, Any]]:
    """Full sentences plus de-duplicated word-prefix truncations at 20/50/100 chars."""
    items = []
    for code in sorted(codes):
        flores_code = FLORES_FOR_CODE[code]
        lines = read_lines(split_file(flores_dir, split, flores_code), split)[:limit_per_language]
        for line_number, text in enumerate(lines):
            items.append(_item("primary", flores_code, line_number, "full", code, text))
            seen = {text}
            for cap in TRUNCATION_CAPS:
                prefix = truncate_prefix(text, cap)
                if prefix and prefix not in seen:
                    seen.add(prefix)
                    items.append(_item("primary", flores_code, line_number, f"p{cap}", code, prefix))
    if variants:
        for flores_code, code in sorted(EXTRA_VARIANTS.items()):
            path = split_file(flores_dir, split, flores_code)
            if not path.is_file():
                raise ValueError(f"--extra-variants needs missing file {path}")
            for line_number, text in enumerate(read_lines(path, split)[:limit_per_language]):
                items.append(_item("variant", flores_code, line_number, "full", code, text))
    return items


def _item(group: str, src: str, line: int, variant: str, gold: str, text: str) -> dict[str, Any]:
    return {"id": f"{src}:{line}:{variant}", "group": group, "src": src, "variant": variant,
            "gold": gold, "text": text, "n": len(text)}


def slice_items(items: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Index lists per reported slice. Length buckets use the actual item length
    over full sentences and truncations together."""
    slices: dict[str, list[int]] = {"full": []}
    for name, _, _ in LENGTH_BUCKETS:
        slices[name] = []
    for index, item in enumerate(items):
        if item["group"] != "primary":
            continue
        if item["variant"] == "full":
            slices["full"].append(index)
        for name, low, high in LENGTH_BUCKETS:
            if item["n"] >= low and (high is None or item["n"] <= high):
                slices[name].append(index)
    variant_items = [i for i, item in enumerate(items) if item["group"] == "variant"]
    if variant_items:
        slices["variants_all"] = variant_items
        for src in sorted({items[i]["src"] for i in variant_items}):
            slices[f"variant:{src}"] = [i for i in variant_items if items[i]["src"] == src]
    return {name: indices for name, indices in slices.items() if indices}


def score(golds: list[str], predictions: list[str]) -> dict[str, Any]:
    """Accuracy, macro P/R/F1/FPR over gold languages, per-language rows, confusions.

    A prediction that is not a gold language of the slice (an unmapped label,
    an error marker, or another language) is wrong and is a false positive only
    for the language it names.
    """
    if len(golds) != len(predictions) or not golds:
        raise ValueError("score needs equally long, non-empty gold and prediction lists")
    total = len(golds)
    support, true_pos, false_pos = Counter(golds), Counter(), Counter()
    confusions = Counter()
    for gold, predicted in zip(golds, predictions):
        if predicted == gold:
            true_pos[gold] += 1
        else:
            false_pos[predicted] += 1
            confusions[(gold, predicted)] += 1
    per_language = {}
    for code in sorted(support):
        tp, fp, fn = true_pos[code], false_pos[code], support[code] - true_pos[code]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        negatives = total - support[code]
        per_language[code] = {
            "support": support[code],
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "fpr": fp / negatives if negatives else None,
        }
    rows = per_language.values()
    fprs = [row["fpr"] for row in rows if row["fpr"] is not None]
    return {
        "texts": total,
        "languages": len(per_language),
        "accuracy": sum(true_pos.values()) / total,
        "macro_precision": sum(row["precision"] for row in rows) / len(rows),
        "macro_recall": sum(row["recall"] for row in rows) / len(rows),
        "macro_f1": sum(row["f1"] for row in rows) / len(rows),
        "macro_fpr": sum(fprs) / len(fprs) if fprs else None,
        "top_confusions": [[g, p, c] for (g, p), c in confusions.most_common(TOP_CONFUSIONS)],
        "per_language": per_language,
    }


def abstention(golds: list[str], accepted: list[str | None]) -> dict[str, Any]:
    """Coverage and precision of accepted answers; None means abstained or error."""
    taken = [(g, a) for g, a in zip(golds, accepted) if a is not None]
    correct = sum(g == a for g, a in taken)
    return {
        "coverage": len(taken) / len(golds),
        "accepted": len(taken),
        "accepted_correct": correct,
        "accepted_errors": len(taken) - correct,
        "accepted_precision": correct / len(taken) if taken else None,
        "correct_accepted_rate": correct / len(golds),
    }


# --- our checkpoint -------------------------------------------------------

_DETECTOR = None


def _init_worker(model_path: str) -> None:
    global _DETECTOR
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from language_detector import LanguageDetector

    _DETECTOR = LanguageDetector(model_path)


def _predict_one(text: str) -> dict[str, Any]:
    try:
        details = _DETECTOR.predict_details(text)
    except Exception as error:  # recorded per item and counted wrong, never dropped
        return {"error": f"{type(error).__name__}: {error}"}
    return {"raw": details["best_language"], "accepted": None if details["uncertain"] else details["language"]}


def predict_ours(model_path: Path, texts: list[str], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        _init_worker(str(model_path))
        return [_predict_one(text) for text in texts]
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    with context.Pool(workers, initializer=_init_worker, initargs=(str(model_path),)) as pool:
        return pool.map(_predict_one, texts, chunksize=512)


def checkpoint_languages(model_path: Path) -> list[str]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from language_detector import load_model

    return sorted(load_model(model_path)[0]["languages"])


# --- external predictions ---------------------------------------------------

NO_MAPPED_LABEL = "<no-mapped-label-in-top-k>"


def load_external(path: Path, split: str, items: list[dict[str, Any]], languages: list[str]) -> dict[str, Any]:
    """Read predict_fasttext.py JSONL and map labels; refuse stale or partial files."""
    with path.open(encoding="utf-8") as stream:
        meta = json.loads(stream.readline()).get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"{path}: first line must be a meta record")
        if meta.get("split") != split:
            raise ValueError(f"{path}: predictions are for split {meta.get('split')!r}, not {split!r}")
        rows = {}
        for line in stream:
            row = json.loads(line)
            rows[row["id"]] = row
    language_set = frozenset(languages)
    restricted, unrestricted, outside_top_k = [], [], 0
    for item in items:
        row = rows.get(item["id"])
        if row is None:
            raise ValueError(f"{path}: no prediction for item {item['id']}")
        if row["sha"] != text_digest(item["text"]):
            raise ValueError(f"{path}: text digest mismatch for item {item['id']}")
        mapped = [map_external_label(label, language_set) for label in row["labels"]]
        top = mapped[0]
        unrestricted.append(top if top else f"<unmapped:{row['labels'][0].removeprefix('__label__')}>")
        first = next((code for code in mapped if code), None)
        if first is None:
            outside_top_k += 1
        restricted.append(first or NO_MAPPED_LABEL)
    label_set = meta.get("labels", [])
    reachable = sorted({code for code in (map_external_label(l, language_set) for l in label_set) if code})
    return {
        "name": meta.get("name", path.stem),
        "meta": {k: v for k, v in meta.items() if k != "labels"} | {"file": str(path)},
        "reachable_codes": len(reachable),
        "unreachable_codes": sorted(set(languages) - set(reachable)),
        "restricted_no_mapped_label_in_top_k": outside_top_k,
        "restricted": restricted,
        "unrestricted": unrestricted,
    }


# --- driver -------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty_files": run("status", "--porcelain").splitlines()}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def evaluate_system(items, slices, predictions, accepted=None) -> dict[str, Any]:
    result = {}
    for name, indices in slices.items():
        golds = [items[i]["gold"] for i in indices]
        entry = score(golds, [predictions[i] for i in indices])
        if accepted is not None:
            entry["abstention"] = abstention(golds, [accepted[i] for i in indices])
        result[name] = entry
    return result


def paired_comparison(
    golds: list[str], ours: list[str], other: list[str], indices: list[int],
    groups: list[str], resamples: int = 1000, seed: int = 0,
) -> dict[str, Any]:
    """Paired accuracy difference (ours - other) on identical items.

    Discordant counts give McNemar's test (normal approximation, continuity
    correction). The 95% interval is a bootstrap over FLORES sentences, not
    items, because the prefixes of one sentence are not independent.
    """
    import math
    import random

    ours_only = sum(1 for i in indices if ours[i] == golds[i] and other[i] != golds[i])
    other_only = sum(1 for i in indices if other[i] == golds[i] and ours[i] != golds[i])
    discordant = ours_only + other_only
    if discordant:
        z = (abs(ours_only - other_only) - 1) / math.sqrt(discordant)
        p_value = math.erfc(max(z, 0.0) / math.sqrt(2))
    else:
        p_value = 1.0
    by_group: dict[str, list[int]] = {}
    for i in indices:
        by_group.setdefault(groups[i], []).append(i)
    keys = sorted(by_group)
    diff_per_group = [
        sum((ours[i] == golds[i]) - (other[i] == golds[i]) for i in by_group[k]) for k in keys
    ]
    size_per_group = [len(by_group[k]) for k in keys]
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        picks = [rng.randrange(len(keys)) for _ in keys]
        estimates.append(sum(diff_per_group[j] for j in picks) / sum(size_per_group[j] for j in picks))
    estimates.sort()
    return {
        "texts": len(indices),
        "accuracy_difference": (ours_only - other_only) / len(indices),
        "ci95": [estimates[int(0.025 * resamples)], estimates[int(0.975 * resamples) - 1]],
        "ours_only_correct": ours_only,
        "other_only_correct": other_only,
        "mcnemar_p": p_value,
        "resampled_unit": "FLORES sentence (all its prefixes together)",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flores-dir", type=Path, required=True, help="extracted flores200_dataset directory")
    parser.add_argument("--split", choices=SPLITS, default="devtest",
                        help="dev: selection; devtest: reporting only (default)")
    parser.add_argument("--model", type=Path, default=ROOT / "model.pt", help="checkpoint, opened read-only")
    parser.add_argument("--fasttext-predictions", type=Path, action="append", default=[],
                        help="JSONL from tools/predict_fasttext.py; repeatable")
    parser.add_argument("--output", type=Path, help="default: reports/flores_<split>.json")
    parser.add_argument("--allow-missing", action="store_true",
                        help="evaluate languages that have a FLORES file and list the rest")
    parser.add_argument("--extra-variants", action="store_true",
                        help="also report " + ", ".join(f"{k}->{v}" for k, v in EXTRA_VARIANTS.items()))
    parser.add_argument("--limit-per-language", type=int, help="first N sentences per file (smoke runs)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="processes for our model")
    parser.add_argument("--predictions-out", type=Path,
                        help="also write per-item gold and predictions of every system as JSONL")
    args = parser.parse_args(argv)
    if args.limit_per_language is not None and args.limit_per_language < 1:
        parser.error("--limit-per-language must be positive")
    output = args.output or ROOT / "reports" / f"flores_{args.split}.json"

    languages = load_languages()
    model_languages = checkpoint_languages(args.model)
    if model_languages != languages:
        print(f"Error: checkpoint languages differ from {LANGUAGES_FILE}: "
              f"{sorted(set(model_languages) ^ set(languages))}", file=sys.stderr)
        return 2
    missing = check_coverage(args.flores_dir, args.split, languages)
    if missing:
        listing = "\n".join(f"  {code}: {reason}" for code, reason in sorted(missing.items()))
        if not args.allow_missing:
            print(f"Error: {len(missing)} languages lack a FLORES mapping or file "
                  f"(pass --allow-missing to evaluate the rest):\n{listing}", file=sys.stderr)
            return 2
        print(f"Warning: skipping {len(missing)} languages:\n{listing}", file=sys.stderr)
    codes = [code for code in languages if code not in missing]
    try:
        items = build_items(args.flores_dir, args.split, codes, variants=args.extra_variants,
                            limit_per_language=args.limit_per_language)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    slices = slice_items(items)

    model_hash = file_sha256(args.model)
    started = time.time()
    ours = predict_ours(args.model, [item["text"] for item in items], args.workers)
    seconds = time.time() - started
    if file_sha256(args.model) != model_hash:
        print("Error: checkpoint changed during evaluation", file=sys.stderr)
        return 2
    errors = [(items[i]["id"], row["error"]) for i, row in enumerate(ours) if "error" in row]
    raw = [row.get("raw") or "<error>" for row in ours]
    accepted = [row.get("accepted") for row in ours]
    print(f"our model: {len(items)} items in {seconds:.1f}s with {args.workers} workers; "
          f"{len(errors)} raised (counted wrong)")

    systems = {"ours": evaluate_system(items, slices, raw, accepted)}
    external_info = {}
    external_predictions: dict[str, list[str]] = {}
    for path in args.fasttext_predictions:
        try:
            external = load_external(path, args.split, items, languages)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 2
        name = external["name"]
        if f"{name}_restricted" in systems:
            print(f"Error: duplicate external system name {name!r}", file=sys.stderr)
            return 2
        systems[f"{name}_restricted"] = evaluate_system(items, slices, external["restricted"])
        systems[f"{name}_unrestricted"] = evaluate_system(items, slices, external["unrestricted"])
        external_info[name] = {k: v for k, v in external.items() if k not in ("restricted", "unrestricted")}
        external_predictions[name] = external["restricted"]
        print(f"{name}: {external['restricted_no_mapped_label_in_top_k']} items without a mapped label "
              f"in top-k (counted wrong when restricted); unreachable codes {external['unreachable_codes']}")

    golds = [item["gold"] for item in items]
    groups = [item["id"].rsplit(":", 1)[0] for item in items]
    paired = {
        f"ours_vs_{name}_restricted": {
            slice_name: paired_comparison(golds, raw, predictions, indices, groups)
            for slice_name, indices in slices.items()
        }
        for name, predictions in external_predictions.items()
    }
    if args.predictions_out:
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_out.open("w", encoding="utf-8") as stream:
            for i, item in enumerate(items):
                row = {"id": item["id"], "gold": item["gold"], "ours": raw[i], "ours_accepted": accepted[i]}
                row.update({f"{name}_restricted": p[i] for name, p in external_predictions.items()})
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "meta": {
            "split": args.split,
            "split_policy": "dev may be used for selection; devtest is for reporting only",
            "command": [Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])],
            "git": git_state(),
            "model": str(args.model),
            "model_sha256": model_hash,
            "flores_dir": str(args.flores_dir),
            "limit_per_language": args.limit_per_language,
            "extra_variants": args.extra_variants,
            "workers": args.workers,
            "seconds_ours": round(seconds, 1),
            "python": sys.version.split()[0],
        },
        "missing_languages": missing,
        "mapping": {code: FLORES_FOR_CODE[code] for code in codes},
        "slice_texts": {name: len(indices) for name, indices in slices.items()},
        "ours_errors": {"count": len(errors), "items": [list(e) for e in errors]},
        "external": external_info,
        "systems": systems,
        "paired": paired,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"| system | slice | texts | accuracy | macro F1 | macro FPR | coverage / accepted precision |")
    for system, result in systems.items():
        for name, entry in result.items():
            ab = entry.get("abstention")
            extra = f"{100 * ab['coverage']:.2f} / {100 * (ab['accepted_precision'] or 0):.2f}" if ab else "-"
            fpr = "-" if entry["macro_fpr"] is None else f"{100 * entry['macro_fpr']:.4f}"
            print(f"| {system} | {name} | {entry['texts']} | {100 * entry['accuracy']:.2f} | "
                  f"{100 * entry['macro_f1']:.2f} | {fpr} | {extra} |")
    for comparison, by_slice in paired.items():
        print(f"| {comparison} | slice | accuracy diff (pp) | 95% CI (pp) | ours-only / other-only | McNemar p |")
        for name, entry in by_slice.items():
            low, high = entry["ci95"]
            print(f"| | {name} | {100 * entry['accuracy_difference']:+.2f} | {100 * low:+.2f} .. {100 * high:+.2f} | "
                  f"{entry['ours_only_correct']} / {entry['other_only_correct']} | {entry['mcnemar_p']:.2g} |")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
