"""Offline tests for prepare_multidomain_data.py (no network)."""

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

import prepare_multidomain_data as prep
from dataset_utils import text_key

LABELS = sorted(json.loads((Path(__file__).resolve().parents[1] / "data/languages.json").read_text()))

FLORES_LINE = (
    "On Monday, scientists from the Stanford University School of Medicine announced "
    "the invention of a new diagnostic tool that can sort cells by type."
)


def make_flores(directory: Path, lines: list[str], files: int = 3) -> Path:
    for split in ("dev", "devtest"):
        (directory / split).mkdir(parents=True)
        for index in range(files):
            content = lines if index == 0 else [f"filler sentence number {index} in {split}"]
            (directory / split / f"x{index}_Latn.{split}").write_text("\n".join(content) + "\n", encoding="utf-8")
    return directory


JA_LINE = "東京の研究者たちは月曜日に、細胞を種類ごとに分類できる新しい診断ツールの発明を発表しました。"


class ScriptFilterTest(unittest.TestCase):
    def test_expected_scripts_pass(self):
        self.assertIsNone(prep.script_check("Ово је реченица на српском.", "sr"))
        self.assertIsNone(prep.script_check("Ovo je rečenica na srpskom.", "sr"))
        self.assertIsNone(prep.script_check("Bu oʻzbekcha gap.", "uz"))
        self.assertIsNone(prep.script_check("Бу ўзбекча гап.", "uz"))
        self.assertIsNone(prep.script_check("Ez dixwazim biçim malê.", "ku"))
        self.assertIsNone(prep.script_check("ئەمە ڕستەیەکی کوردییە.", "ku"))
        self.assertIsNone(prep.script_check("這是繁體中文。", "zh"))
        self.assertIsNone(prep.script_check("这是简体中文。", "zh"))
        self.assertIsNone(prep.script_check("これは日本語の文です。", "ja"))
        self.assertIsNone(prep.script_check("यह हिन्दी वाक्य है।", "hi"))

    def test_wrong_script_dropped(self):
        self.assertEqual(prep.script_check("This is plain English text.", "ru"), "script")
        self.assertEqual(prep.script_check("Ово је реченица.", "hr"), "script")  # hr is Latin only
        self.assertEqual(prep.script_check("Это русский текст.", "bs"), "script")
        self.assertEqual(prep.script_check("这是中文。", "ja"), "script")  # no kana
        self.assertEqual(prep.script_check("12345 !!!", "en"), "no_letters")

    def test_half_threshold(self):
        # 6 Cyrillic letters vs 4 Latin letters: passes for ru, fails for en.
        self.assertIsNone(prep.script_check("привет abcd", "ru"))
        self.assertEqual(prep.script_check("привет abcd", "en"), "script")

    def test_every_label_has_a_known_script(self):
        for label in LABELS + list(prep.NEGATIVE_LABELS):
            for name in prep.expected_scripts(label):
                self.assertIn(name, prep.SCRIPT_RE)


class LabelMappingTest(unittest.TestCase):
    def test_owner_decisions(self):
        self.assertEqual(prep.resolve_label("ckb", LABELS)[0], "ku")
        self.assertEqual(prep.resolve_label("ckb_Arab", LABELS)[0], "ku")
        self.assertEqual(prep.resolve_label("kmr_Latn", LABELS)[0], "ku")
        self.assertEqual(prep.resolve_label("fil", LABELS)[0], "tl")
        self.assertEqual(prep.resolve_label("srp_Latn", LABELS)[0], "sr")
        self.assertEqual(prep.resolve_label("uzn_Cyrl", LABELS)[0], "uz")
        self.assertEqual(prep.resolve_label("cmn_Hant", LABELS)[0], "zh")
        self.assertEqual(prep.resolve_label("no", LABELS)[0], "nb")
        for raw in ("bs", "hr", "sr"):
            self.assertEqual(prep.resolve_label(raw, LABELS)[0], raw)

    def test_dropped_and_negative(self):
        for raw in ("azb", "azb_Arab", "sh", "yue"):
            label, reason = prep.resolve_label(raw, LABELS)
            self.assertIsNone(label)
            self.assertTrue(reason)
        self.assertEqual(prep.resolve_label("nno", LABELS)[0], "nn")
        self.assertNotIn("nn", LABELS)  # negatives never become a training label
        self.assertIsNone(prep.resolve_label("xyz", LABELS)[0])

    def test_source_tables_only_use_known_labels(self):
        known = set(LABELS) | set(prep.NEGATIVE_LABELS)
        for table in (prep.OPENLID_FILES, prep.TATOEBA_CODES, prep.MADLAD_CODES,
                      prep.WIKI_CODES, prep.FINEWEB2_SLICES):
            self.assertLessEqual(set(table), known)
        self.assertEqual(set(prep.OPENLID_FILES), set(LABELS))
        self.assertNotIn("sh", sum(prep.WIKI_CODES.values(), []))
        self.assertFalse(set(prep.OPENLID_ALLOWED_TAGS) & set(prep.OPENLID_EXCLUDED_TAGS))
        for tag in ("xlsum", "leipzig", "leipzigwiki", "til", "afriqa", "lti", "tep", "madar", "wiki"):
            self.assertIn(tag, prep.OPENLID_EXCLUDED_TAGS)
        self.assertIn("mt560", prep.OPENLID_ALLOWED_TAGS)


class ContaminationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.flores = make_flores(Path(self.directory.name), [FLORES_LINE, JA_LINE])
        self.index = prep.ShingleIndex(prep.flores_lines(self.flores, expected_per_split=3))

    def tearDown(self):
        self.directory.cleanup()

    def test_positive_control_exact(self):
        self.assertEqual(self.index.check(text_key(FLORES_LINE)), "exact")
        # Case, punctuation and digits do not hide a match.
        self.assertEqual(self.index.check(text_key(FLORES_LINE.upper().replace(",", ""))), "exact")

    def test_partial_overlap_dropped(self):
        variant = FLORES_LINE.replace("by type.", "by type, a spokesperson said.")
        self.assertEqual(self.index.check(text_key(variant)), "shingle")

    def test_unspaced_char_shingles(self):
        variant = JA_LINE.replace("発表しました。", "発表したと報じられた。")
        self.assertEqual(self.index.check(text_key(variant)), "shingle")
        # Below 30 normalized characters only the exact check applies.
        self.assertIsNone(self.index.check(text_key(JA_LINE[:20])))

    def test_unrelated_kept(self):
        self.assertIsNone(self.index.check(text_key("The weather in the mountains was cold and windy all week long.")))
        self.assertIsNone(self.index.check(text_key("Short line")))

    def test_minority_overlap_kept(self):
        # Under half of the 8-word shingles come from FLORES.
        words = FLORES_LINE.split()[:9]
        text = " ".join(words + "and then the committee met again on a rainy afternoon to discuss other matters".split())
        self.assertIsNone(self.index.check(text_key(text)))

    def test_refuses_incomplete_flores(self):
        with self.assertRaises(ValueError):
            list(prep.flores_lines(self.flores))  # expects 204 files per split


class CropTest(unittest.TestCase):
    def test_word_crops(self):
        rng = random.Random(1)
        text = " ".join(f"w{i}" for i in range(30))
        for _ in range(200):
            crop = prep.make_crop(text, "en", rng)
            words = crop.split()
            self.assertTrue(4 <= len(words) <= 16)
            self.assertIn(crop, text)

    def test_crop_is_strictly_shorter(self):
        rng = random.Random(2)
        text = "one two three four five"
        for _ in range(50):
            self.assertEqual(len(prep.make_crop(text, "en", rng).split()), 4)
        self.assertIsNone(prep.make_crop("one two three four", "en", rng))

    def test_unspaced_crops(self):
        rng = random.Random(3)
        text = "我们今天在北京召开了一次非常重要的会议讨论了很多关于经济发展和环境保护的问题大家都积极发言" * 2
        for _ in range(200):
            crop = prep.make_crop(text, "zh", rng)
            self.assertTrue(16 <= len(crop) <= 60)
            self.assertIn(crop, text)
        self.assertIsNone(prep.make_crop("太短了的句子", "zh", rng))


class DedupTest(unittest.TestCase):
    def test_cross_label_rule(self):
        a, b, c = (prep.key_hash(text_key(t)) for t in ("Dobar dan svima", "Hvala lijepa", "God dag alle"))
        hashes = np.array([a, b, a, c, c], dtype=np.int64)
        owners = np.array([0, 0, 1, 2, 2], dtype=np.int64)  # a: hr+sr, c: twice in one label
        conflicts = prep.cross_label_conflicts(hashes, owners)
        self.assertEqual(conflicts.tolist(), [a])

    def test_keys_ignore_case_accents_digits(self):
        self.assertEqual(prep.key_hash(text_key("Čaj 2024!")), prep.key_hash(text_key("caj")))

    def test_near_signatures(self):
        base = "the quarterly report of the regional council was published today in full"
        other = base + " with an appendix"
        self.assertTrue(set(prep.near_signatures(text_key(base))) & set(prep.near_signatures(text_key(other))))
        self.assertEqual(prep.near_signatures("short key"), ())

    def test_allocate_redistributes(self):
        mix = {"web": 0.4, "wiki": 0.3, "short": 0.15, "other": 0.15}
        alloc = prep.allocate(mix, {"web": 1000, "wiki": 1000, "short": 10, "other": 1000}, 100)
        self.assertEqual(sum(alloc.values()), 100)
        self.assertEqual(alloc["short"], 10)
        alloc = prep.allocate(mix, {"web": 5, "wiki": 0, "short": 0, "other": 0}, 100)
        self.assertEqual(alloc["web"], 5)


class FilterTest(unittest.TestCase):
    def setUp(self):
        self.check = prep.Filter(min_chars=10, max_chars=300, min_words=3)

    def test_normalization_and_reasons(self):
        text, key = self.check("  Café au   lait est bon.  ", "fr", need_words=True)
        self.assertEqual(text, "Café au lait est bon.")
        self.assertEqual(self.check("short", "en", False), "too_short")
        self.assertEqual(self.check("see https://example.com for more", "en", False), "url_or_email")
        self.assertEqual(self.check("12345 67890 12345 a", "en", False), "too_few_letters")
        self.assertEqual(self.check("Heading only", "en", need_words=True), "too_few_words")

    def test_segments(self):
        parts = prep.split_segments("Prva rečenica. Druga rečenica! 第一句。第二句。", 300)
        self.assertEqual(parts, ["Prva rečenica.", "Druga rečenica!", "第一句。", "第二句。"])
        long = " ".join(["word"] * 200)
        self.assertTrue(all(len(p) <= 300 for p in prep.split_segments(long, 300)))


if __name__ == "__main__":
    unittest.main()
