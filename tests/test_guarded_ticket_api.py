"""Optional ticket-guard API; mocked model, never holdout/FT."""

import math
import unittest

from language_detector import LanguageDetector


def _detector(votes):
    det = LanguageDetector.__new__(LanguageDetector)
    calls = []

    def predict_details(text):
        calls.append(text)
        return votes[text]

    det.predict_details = predict_details
    det.calls = calls
    return det


def _vote(best, confidence, accepted=None):
    return {"language": accepted if accepted is not None else best,
            "best_language": best, "confidence": confidence,
            "margin": 0.1, "uncertain": accepted == "und",
            "calibrated": True}


class GuardedTicketApiTests(unittest.TestCase):
    def _prepared(self, title, description):
        from ticket_guard import prepare_ticket
        return prepare_ticket(title, description)

    def test_mono_ticket_returns_whole_language(self):
        title, desc = "Drucker defekt", (
            "Im Drucker ist das Papier eingeklemmt. "
            "Bitte senden Sie bald einen Techniker vorbei.")
        prepared = self._prepared(title, desc)
        self.assertGreaterEqual(len(prepared.sentences), 2)
        votes = {prepared.text: _vote("de", 0.98)}
        votes.update({s: _vote("de", 0.97) for s in prepared.sentences})
        det = _detector(votes)
        self.assertEqual(det.predict_guarded_ticket(title, desc), "de")

    def test_confident_foreign_segment_withholds(self):
        title, desc = "Problem", (
            "Mein Drucker funktioniert nicht. "
            "The connection was refused hard.")
        prepared = self._prepared(title, desc)
        self.assertGreaterEqual(len(prepared.sentences), 2)
        votes = {prepared.text: _vote("de", 0.98)}
        votes.update({s: _vote("de", 0.97) for s in prepared.sentences})
        votes[prepared.sentences[-1]] = _vote("en", 0.95)
        det = _detector(votes)
        self.assertEqual(det.predict_guarded_ticket(title, desc), "und")

    def test_low_confidence_foreign_keeps_answer(self):
        title, desc = "Problem", (
            "Mein Drucker funktioniert nicht. "
            "The connection was refused hard.")
        prepared = self._prepared(title, desc)
        votes = {prepared.text: _vote("de", 0.98)}
        votes.update({s: _vote("de", 0.97) for s in prepared.sentences})
        votes[prepared.sentences[-1]] = _vote("en", 0.50)
        det = _detector(votes)
        self.assertEqual(det.predict_guarded_ticket(title, desc), "de")

    def test_cutoff_boundary_vetoes_at_equal_confidence(self):
        title, desc = "Problem", (
            "Mein Drucker funktioniert nicht. "
            "The connection was refused hard.")
        prepared = self._prepared(title, desc)
        votes = {prepared.text: _vote("de", 0.98)}
        votes.update({s: _vote("de", 0.97) for s in prepared.sentences})
        votes[prepared.sentences[-1]] = _vote("en", 0.65)
        det = _detector(votes)
        self.assertEqual(
            det.predict_guarded_ticket(title, desc, cutoff=0.65), "und")
        det = _detector(votes)
        self.assertEqual(
            det.predict_guarded_ticket(title, desc, cutoff=0.66), "de")

    def test_uncertain_whole_stays_und_without_segment_calls(self):
        title, desc = "Problem", (
            "Mein Drucker funktioniert nicht. "
            "The connection was refused hard.")
        prepared = self._prepared(title, desc)
        votes = {prepared.text: _vote("de", 0.40, accepted="und")}
        det = _detector(votes)
        self.assertEqual(det.predict_guarded_ticket(title, desc), "und")
        self.assertEqual(det.calls, [prepared.text])

    def test_short_ticket_without_segments_returns_whole(self):
        prepared = self._prepared("Drucker defekt", "")
        self.assertEqual(prepared.sentences, ())
        det = _detector({prepared.text: _vote("de", 0.90)})
        self.assertEqual(
            det.predict_guarded_ticket("Drucker defekt", ""), "de")

    def test_code_fenced_log_is_not_evidence(self):
        title, desc = "Problem", (
            "Mein Drucker funktioniert nicht.\n```\nAccess is denied\n```")
        prepared = self._prepared(title, desc)
        self.assertNotIn("Access", prepared.text)
        votes = {prepared.text: _vote("de", 0.96)}
        votes.update({s: _vote("de", 0.95) for s in prepared.sentences})
        det = _detector(votes)
        self.assertEqual(det.predict_guarded_ticket(title, desc), "de")

    def test_unused_subject_never_vetoes(self):
        title = "The connection was refused"
        desc = ("Mein Drucker druckt seit heute nur noch leere Seiten. "
                "Bitte senden Sie bald einen Techniker vorbei.")
        prepared = self._prepared(title, desc)
        self.assertNotIn("refused", " ".join(prepared.sentences))
        votes = {prepared.text: _vote("de", 0.98)}
        votes.update({s: _vote("de", 0.97) for s in prepared.sentences})
        det = _detector(votes)
        self.assertEqual(det.predict_guarded_ticket(title, desc), "de")

    def test_header_source_matches_whole(self):
        title = "Problem"
        desc = ("From: Max Mustermann Support\n\n"
                "Mein Drucker druckt seit heute nur noch leere Seiten. "
                "Bitte senden Sie bald einen Techniker vorbei.")
        prepared = self._prepared(title, desc)
        self.assertIn("Mustermann", prepared.text)
        self.assertIn("Mustermann", " ".join(prepared.sentences))
        votes = {prepared.text: _vote("de", 0.96)}
        votes.update({s: _vote("de", 0.95) for s in prepared.sentences})
        det = _detector(votes)
        self.assertEqual(det.predict_guarded_ticket(title, desc), "de")

    def test_letterless_input_returns_und_without_model(self):
        det = _detector({})
        self.assertEqual(det.predict_guarded_ticket("123", "!!!"), "und")
        self.assertEqual(det.calls, [])

    def test_long_input_uses_prefix_and_answers(self):
        desc = "Bitte helfen Sie mir sofort. " * 500
        prepared = self._prepared("Problem", desc)
        self.assertLess(
            sum(len(s) for s in prepared.sentences), len(desc))
        self.assertTrue(all(s in prepared.text for s in prepared.sentences))
        votes = {prepared.text: _vote("de", 0.96)}
        votes.update({s: _vote("de", 0.95) for s in prepared.sentences})
        det = _detector(votes)
        self.assertEqual(
            det.predict_guarded_ticket("Problem", desc), "de")

    def test_cutoff_must_be_a_finite_number_in_range(self):
        prepared = self._prepared("Drucker defekt", "")
        det = _detector({prepared.text: _vote("de", 0.90)})
        for bad in (True, False, "0.65", None, math.nan,
                    math.inf, -0.1, 1.1):
            with self.assertRaises(ValueError, msg=repr(bad)):
                det.predict_guarded_ticket("Drucker defekt", "",
                                           cutoff=bad)
        det = _detector({prepared.text: _vote("de", 0.90)})
        self.assertEqual(
            det.predict_guarded_ticket("Drucker defekt", "",
                                       cutoff=0.0), "de")
        det = _detector({prepared.text: _vote("de", 0.90)})
        self.assertEqual(
            det.predict_guarded_ticket("Drucker defekt", "",
                                       cutoff=1.0), "de")

    def test_raw_uncalibrated_model_still_guards(self):
        title, desc = "Problem", (
            "Mein Drucker funktioniert nicht. "
            "The connection was refused hard.")
        prepared = self._prepared(title, desc)
        votes = {prepared.text: _vote("de", 0.98)}
        votes.update({s: _vote("de", 0.97) for s in prepared.sentences})
        votes[prepared.sentences[-1]] = _vote("en", 0.95)
        for vote in votes.values():
            vote["calibrated"] = False
        det = _detector(votes)
        self.assertEqual(det.predict_guarded_ticket(title, desc), "und")

    def test_model_errors_propagate(self):
        prepared = self._prepared("Drucker defekt", "")
        det = _detector({})
        det.predict_details = lambda text: (_ for _ in ()).throw(
            RuntimeError("weights broken"))
        with self.assertRaises(RuntimeError):
            det.predict_guarded_ticket("Drucker defekt", "")
        votes = {prepared.text: _vote("de", 0.98)}

        def flaky(text):
            if text != prepared.text:
                raise ValueError("segment encoding broken")
            return votes[text]

        det.predict_details = flaky
        with self.assertRaises(ValueError):
            det.predict_guarded_ticket(
                "Drucker defekt", "Und noch ein langer Satz hier extra.")

    def test_beyond_prefix_foreign_single_line_never_vetoes(self):
        german = "Bitte senden Sie bald einen Techniker vorbei. "
        english = "The connection was refused hard and long. "
        desc = (german * 44 + english).strip()
        prepared = self._prepared("Problem", desc)
        self.assertNotIn("refused", " ".join(prepared.sentences))
        self.assertTrue(all(s in prepared.text for s in prepared.sentences))

        calls = []

        def predict_details(text):
            calls.append(text)
            if "refused" in text:
                return _vote("en", 0.95)
            return _vote("de", 0.97)

        det = LanguageDetector.__new__(LanguageDetector)
        det.predict_details = predict_details
        self.assertEqual(
            det.predict_guarded_ticket("Problem", desc), "de")
        self.assertFalse(any("refused" in c for c in calls))

    def test_straddling_second_line_clipped_at_prefix(self):
        line1 = "Mein Drucker druckt seit heute nur noch leere Seiten."
        german = "Bitte senden Sie bald einen Techniker vorbei. "
        english = "The connection was refused hard and long. "
        desc = line1 + "\n" + (german * 44 + english).strip()
        prepared = self._prepared("Problem", desc)
        self.assertNotIn("refused", " ".join(prepared.sentences))
        self.assertTrue(all(s in prepared.text for s in prepared.sentences))

        def predict_details(text):
            if "refused" in text:
                return _vote("en", 0.95)
            return _vote("de", 0.97)

        det = LanguageDetector.__new__(LanguageDetector)
        calls = []

        def tracked(text):
            calls.append(text)
            return predict_details(text)

        det.predict_details = tracked
        self.assertEqual(det.predict_guarded_ticket("Problem", desc), "de")
        self.assertFalse(any("refused" in c for c in calls))

    def test_no_fasttext_dependency_in_runtime_guard(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        source = (root / "ticket_guard.py").read_text(encoding="utf-8")
        self.assertNotIn("fasttext", source)
        self.assertNotIn("lid.176", source)


class RealModelSmokeTests(unittest.TestCase):
    def test_bundled_model_guarded_clear_cases(self):
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "model.pt"
        if not path.exists():
            self.skipTest("Train or download the released checkpoint first")
        detector = LanguageDetector(path)
        self.assertEqual(
            detector.predict_guarded_ticket(
                "Konto", "Proszę o pomoc z dostępem do konta."), "pl")
        self.assertEqual(
            detector.predict_guarded_ticket(
                "Konto", "Ich kann mich nicht mehr anmelden."), "de")
        self.assertEqual(detector.predict_guarded_ticket("123", "!!!"), "und")


if __name__ == "__main__":
    unittest.main()
