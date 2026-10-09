"""Small API regression suite; no downloaded corpus is required."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from language_detector import LanguageDetector, build_model, encode
from ticket_text import extract_ticket_text, normalize_ticket_text

ROOT = Path(__file__).resolve().parents[1]


class DetectorTests(unittest.TestCase):
    def test_unicode_and_feature_options(self):
        self.assertEqual(encode(" ŻÓŁĆ "), encode("z\u0307o\u0301łc\u0301"))
        self.assertNotEqual(
            encode("hello world"), encode("hello world", word_features=True)
        )
        self.assertEqual(
            encode("ZaÅ¼Ã³Å‚Ä‡", preprocessing="ticket_v2"),
            encode("Zażółć", preprocessing="ticket_v2"),
        )

    def test_invalid_inputs_and_cap(self):
        for text in ("", "   ", "123 !!!", "1" * 2000 + "letters"):
            with self.assertRaises(ValueError):
                encode(text)
        self.assertEqual(encode("abc" * 1000), encode(("abc" * 1000)[:2000]))

    def test_checkpoint_batch_and_confidence_contract(self):
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as directory:
            model = build_model(2)
            path = Path(directory) / "model.pt"
            checkpoint = {
                "state_dict": model.state_dict(),
                "languages": ["en", "pl"],
                "calibration": {"temperature": 2.0, "min_confidence": 1.000001},
            }
            torch.save(checkpoint, path)
            detector = LanguageDetector(path)
            texts = ["Please help", "Proszę o pomoc"]
            self.assertEqual(
                detector.predict_many(texts, batch_size=1),
                [detector.predict(t) for t in texts],
            )
            self.assertEqual(detector.predict_many([]), [])
            self.assertFalse(detector.model.training)
            self.assertEqual(detector.predict_details(texts[0])["language"], "und")
            self.assertEqual(
                detector.predict_ticket("123 !!!", allow_uncertain=True), "und"
            )
            with self.assertRaises(ValueError):
                detector.predict_many(texts, batch_size=0)

    def test_cleaning_preserves_description_language(self):
        self.assertEqual(
            normalize_ticket_text("Pomocy! https://example.com INC123"), "Pomocy!"
        )
        self.assertEqual(
            extract_ticket_text("English title", "请帮助我恢复账户访问权限谢谢您"),
            "请帮助我恢复账户访问权限谢谢您",
        )
        self.assertEqual(
            extract_ticket_text(
                "System title",
                "Proszę o pomoc z kontem. Best regards, English department",
            ),
            "Proszę o pomoc z kontem.",
        )
        self.assertEqual(
            extract_ticket_text(
                "System title",
                "Following Service Request has not been fulfilled correctly. Users Comment: Ich brauche Hilfe mit meinem Konto.",
            ),
            "Ich brauche Hilfe mit meinem Konto.",
        )

    def test_language_list(self):
        languages = json.loads((ROOT / "data/languages.json").read_text())
        self.assertEqual(len(languages), 100)
        self.assertIn("ja", languages)
        self.assertNotIn("jp", languages)

    def test_prepared_splits_if_present(self):
        from dataset_utils import read_csv, text_key

        directory = ROOT / "data/prepared"
        if not (directory / "manifest.json").exists():
            self.skipTest("Prepare public corpora first")
        keys = {
            split: {
                text_key(row["text"])
                for row in read_csv(directory / f"{split}_short.csv")
            }
            for split in ("train", "valid", "test")
        }
        for first, second in (("train", "valid"), ("train", "test"), ("valid", "test")):
            self.assertFalse(keys[first] & keys[second])
        authored = {
            text_key(row["text"]) for row in read_csv(ROOT / "data/synthetic.csv")
        }
        self.assertFalse(keys["train"] & authored)

    def test_bundled_model_if_present(self):
        path = ROOT / "model.pt"
        if not path.exists():
            self.skipTest("Train or download the released checkpoint first")
        detector = LanguageDetector(path)
        self.assertEqual(len(detector.languages), 100)
        self.assertEqual(detector.predict("Proszę o pomoc z dostępem do konta."), "pl")
        self.assertEqual(detector.predict("Ich kann mich nicht mehr anmelden."), "de")


if __name__ == "__main__":
    unittest.main()
