"""Bundled release model (model.npz, v22): contents and smoke behaviour."""

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from language_detector import DEFAULT_MODEL, LanguageDetector

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model.npz"


class ReleaseModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL.exists():
            raise unittest.SkipTest("model.npz is absent; download the complete repository")
        cls.detector = LanguageDetector(MODEL)

    def test_default_path_and_contents(self):
        self.assertEqual(DEFAULT_MODEL, Path("model.npz"))
        self.assertEqual(len(self.detector.languages), 100)
        self.assertEqual(self.detector.model.embedding.weight.dtype, np.int8)
        self.assertEqual(self.detector.model.embedding.shape, (131072, 64))
        # Single network: the policy keys predict_details reads, nothing copied blindly.
        self.assertEqual(sorted(self.detector.calibration), [
            "min_confidence", "min_confidence_by_language",
            "min_confidence_by_language_and_length", "rule", "temperature"])

    def test_clear_requests_are_accepted(self):
        d = self.detector
        self.assertEqual(d.predict_details("Proszę o pomoc z dostępem do konta.")["language"], "pl")
        self.assertEqual(d.predict_details("Ich kann mich nicht mehr anmelden.")["language"], "de")
        self.assertEqual(d.predict_ticket("My VPN stopped working today. Please reinstall it.",
                                          allow_uncertain=True), "en")
        self.assertEqual(d.predict_ticket("123", "!!!", allow_uncertain=True), "und")

    def test_guarded_ticket_clear_cases(self):
        d = self.detector
        self.assertEqual(d.predict_guarded_ticket("Konto", "Proszę o pomoc z dostępem do konta."), "pl")
        self.assertEqual(d.predict_guarded_ticket("Konto", "Ich kann mich nicht mehr anmelden."), "de")
        self.assertEqual(d.predict_guarded_ticket("123", "!!!"), "und")

    def test_bokmal_is_answered(self):
        # Owner decision for v22: Bokmål is answered; Nynorsk (not a label) may be answered as nb.
        self.assertEqual(self.detector.predict_details(
            "Jeg trenger hjelp med kontoen min, jeg får ikke logget inn.")["language"], "nb")

    def test_runs_without_pytorch(self):
        script = (
            "import sys; sys.modules['torch'] = None\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from language_detector import LanguageDetector\n"
            f"print(LanguageDetector({str(MODEL)!r}).predict('Ich kann mich nicht mehr anmelden.'))\n"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "de")


if __name__ == "__main__":
    unittest.main()
