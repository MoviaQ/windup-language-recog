"""FLORES evaluation accounting, testable without FLORES data or fastText."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from tools.evaluate_flores import (
    EXPECTED_LINES,
    EXTRA_VARIANTS,
    FLORES_FOR_CODE,
    NLLB_LABELS,
    abstention,
    build_items,
    check_coverage,
    load_external,
    load_languages,
    main,
    map_external_label,
    score,
    slice_items,
    text_digest,
    truncate_prefix,
)

LANGUAGES = frozenset(load_languages())


class MappingTests(unittest.TestCase):
    def test_every_supported_code_has_a_distinct_flores_file(self):
        self.assertEqual(set(FLORES_FOR_CODE), set(LANGUAGES))
        self.assertEqual(len(set(FLORES_FOR_CODE.values())), len(FLORES_FOR_CODE))

    def test_documented_ambiguous_choices(self):
        expected = {"fa": "pes_Arab", "ku": "kmr_Latn", "zh": "zho_Hans", "nb": "nob_Latn",
                    "ms": "zsm_Latn", "sq": "als_Latn", "mg": "plt_Latn", "om": "gaz_Latn",
                    "sr": "srp_Cyrl", "ar": "arb_Arab"}
        self.assertEqual({k: FLORES_FOR_CODE[k] for k in expected}, expected)

    def test_variants_target_supported_codes_and_differ_from_primary(self):
        self.assertEqual(EXTRA_VARIANTS, {"zho_Hant": "zh", "prs_Arab": "fa", "ckb_Arab": "ku"})
        self.assertFalse(set(EXTRA_VARIANTS) & set(FLORES_FOR_CODE.values()))

    def test_fasttext_labels(self):
        self.assertEqual(map_external_label("__label__no", LANGUAGES), "nb")
        self.assertEqual(map_external_label("__label__en", LANGUAGES), "en")
        self.assertEqual(map_external_label("__label__ckb", LANGUAGES), "ku")
        self.assertIsNone(map_external_label("__label__sh", LANGUAGES))
        self.assertIsNone(map_external_label("__label__nn", LANGUAGES))

    def test_nllb_labels(self):
        self.assertEqual(map_external_label("__label__arb_Arab", LANGUAGES), "ar")
        self.assertEqual(map_external_label("arz_Arab", LANGUAGES), "ar")
        self.assertEqual(map_external_label("__label__zho_Hant", LANGUAGES), "zh")
        self.assertEqual(map_external_label("__label__nob_Latn", LANGUAGES), "nb")
        self.assertIsNone(map_external_label("__label__nno_Latn", LANGUAGES))
        # A three-letter NLLB-style code never falls through to ISO matching.
        self.assertIsNone(map_external_label("__label__eng_Cyrl", LANGUAGES))
        self.assertTrue(set(NLLB_LABELS.values()) <= LANGUAGES)


class TruncationTests(unittest.TestCase):
    def test_short_text_is_not_truncated(self):
        self.assertIsNone(truncate_prefix("short text", 20))

    def test_word_prefix(self):
        text = "The quick brown fox jumps over the lazy dog"
        self.assertEqual(truncate_prefix(text, 20), "The quick brown fox")
        self.assertLessEqual(len(truncate_prefix(text, 20)), 20)

    def test_scripts_without_spaces_use_character_prefix(self):
        text = "これは日本語の文章でスペースがありませんので文字で切ります"
        self.assertEqual(truncate_prefix(text, 20), text[:20])

    def test_long_first_word_falls_back_to_characters(self):
        text = "Donaudampfschifffahrtsgesellschaftskapitän fährt"
        self.assertEqual(truncate_prefix(text, 20), text[:20])


class MetricTests(unittest.TestCase):
    def test_tiny_example(self):
        golds = ["de", "de", "en", "en"]
        preds = ["de", "en", "en", "<unmapped:sh>"]
        result = score(golds, preds)
        self.assertEqual(result["accuracy"], 0.5)
        de, en = result["per_language"]["de"], result["per_language"]["en"]
        self.assertEqual((de["precision"], de["recall"], de["fpr"]), (1.0, 0.5, 0.0))
        self.assertEqual((en["precision"], en["recall"], en["fpr"]), (0.5, 0.5, 0.5))
        self.assertAlmostEqual(de["f1"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 0.5) / 2)
        self.assertAlmostEqual(result["macro_fpr"], 0.25)
        self.assertEqual(result["top_confusions"], [["de", "en", 1], ["en", "<unmapped:sh>", 1]])

    def test_single_language_slice_has_no_fpr(self):
        self.assertIsNone(score(["zh", "zh"], ["zh", "ja"])["macro_fpr"])

    def test_empty_or_misaligned_input_is_refused(self):
        with self.assertRaises(ValueError):
            score([], [])
        with self.assertRaises(ValueError):
            score(["de"], [])

    def test_abstention_and_errors_are_not_accepted(self):
        result = abstention(["de", "de", "en", "en"], ["de", None, "de", None])
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["accepted_precision"], 0.5)
        self.assertEqual(result["accepted_errors"], 1)
        self.assertEqual(result["correct_accepted_rate"], 0.25)


def write_split(root: Path, split: str, files: dict[str, list[str]]) -> None:
    (root / split).mkdir(parents=True, exist_ok=True)
    for code, lines in files.items():
        padded = lines + [f"filler {i}" for i in range(EXPECTED_LINES[split] - len(lines))]
        (root / split / f"{code}.{split}").write_text("\n".join(padded) + "\n", encoding="utf-8")


class SyntheticFloresTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_split(self.root, "dev", {
            "eng_Latn": ["The quick brown fox jumps over the lazy dog again and again"],
            "deu_Latn": ["Kurz"],
        })

    def tearDown(self):
        self.tmp.cleanup()

    def test_coverage_lists_every_missing_language(self):
        missing = check_coverage(self.root, "dev", sorted(LANGUAGES))
        self.assertEqual(set(missing), LANGUAGES - {"en", "de"})

    def test_cli_refuses_missing_languages(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(["--flores-dir", str(self.root), "--split", "dev", "--workers", "1",
                         "--output", str(self.root / "out.json")])
        self.assertEqual(code, 2)
        self.assertIn("98 languages lack", stderr.getvalue())
        self.assertFalse((self.root / "out.json").exists())

    def test_items_and_slices(self):
        items = build_items(self.root, "dev", ["en", "de"], limit_per_language=1)
        ids = [item["id"] for item in items]
        self.assertEqual(ids, ["deu_Latn:0:full", "eng_Latn:0:full", "eng_Latn:0:p20", "eng_Latn:0:p50"])
        slices = slice_items(items)
        self.assertEqual(len(slices["full"]), 2)
        self.assertEqual([items[i]["id"] for i in slices["len<=20"]], ["deu_Latn:0:full", "eng_Latn:0:p20"])
        with self.assertRaises(ValueError):
            build_items(self.root, "dev", ["en"], variants=True)

    def test_external_predictions_are_mapped_and_validated(self):
        items = build_items(self.root, "dev", ["en", "de"], limit_per_language=1)
        rows = {
            "deu_Latn:0:full": ["__label__sh", "__label__de"],
            "eng_Latn:0:full": ["__label__en"],
            "eng_Latn:0:p20": ["__label__nn"],
            "eng_Latn:0:p50": ["__label__no", "__label__en"],
        }
        path = self.root / "pred.jsonl"

        def write(split="dev", drop=None):
            with path.open("w", encoding="utf-8") as stream:
                stream.write(json.dumps({"meta": {"name": "ft", "split": split, "labels": ["__label__en"]}}) + "\n")
                for item in items:
                    if item["id"] != drop:
                        stream.write(json.dumps({"id": item["id"], "sha": text_digest(item["text"]),
                                                 "labels": rows[item["id"]], "probs": []}) + "\n")

        write()
        result = load_external(path, "dev", items, sorted(LANGUAGES))
        self.assertEqual(result["unrestricted"], ["<unmapped:sh>", "en", "<unmapped:nn>", "nb"])
        self.assertEqual(result["restricted"], ["de", "en", "<no-mapped-label-in-top-k>", "nb"])
        self.assertEqual(result["restricted_no_mapped_label_in_top_k"], 1)
        self.assertEqual(result["reachable_codes"], 1)
        with self.assertRaises(ValueError):
            load_external(path, "devtest", items, sorted(LANGUAGES))
        write(drop="eng_Latn:0:p20")
        with self.assertRaises(ValueError):
            load_external(path, "dev", items, sorted(LANGUAGES))


if __name__ == "__main__":
    unittest.main()
