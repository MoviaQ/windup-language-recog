"""Benchmark accounting stays independent of optional fastText bindings."""

import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from tools.benchmark_ticket_guard import (
    ft_vote,
    load_cases,
    matched_predictions,
    summarize,
    validate_output,
)


class FakeFastText:
    def __init__(self, label="__label__de", score=0.9):
        self.label = label
        self.score = score
        self.inputs = []

    def predict(self, texts, k=1):
        self.inputs.append(texts)
        return [(self.label,)], [[self.score]]


class TicketBenchmarkTests(unittest.TestCase):
    def test_abstentions_are_not_correct_answers(self):
        cases = [{"language": code} for code in ("de", "und", "und", "en")]
        result = summarize(cases, ["de", "und", "fr", "und"])
        self.assertEqual((result["C"], result["E"], result["U"]), (1, 1, 2))
        self.assertEqual(result["mixed_correct_withheld"], 1)
        self.assertEqual(result["mixed_false_acceptances"], 1)
        self.assertEqual(result["n"], result["C"] + result["E"] + result["U"])

    def test_population_drift_fails(self):
        with self.assertRaises(ValueError):
            summarize([{"language": "de"}], [])

    def test_exact_matching_does_not_reintroduce_guard_vetoes(self):
        rows = [
            {"id": "a", "ft_label": "de", "ft_score": 0.98, "ft_eligible": True},
            {"id": "b", "ft_label": "fr", "ft_score": 0.60, "ft_eligible": True},
            {"id": "c", "ft_label": "en", "ft_score": 0.99, "ft_eligible": False},
        ]
        predictions, info = matched_predictions(rows, 2)
        self.assertEqual(predictions, ["de", "fr", "und"])
        self.assertEqual(info["answered"], 2)
        self.assertEqual(info["cutoff"], 0.60)
        self.assertEqual(len(predictions), len(rows))

    def test_matching_uses_scores_and_ids_not_gold(self):
        rows = [
            {"id": "b", "ft_label": "de", "ft_score": 0.9, "ft_eligible": True, "language": "en"},
            {"id": "a", "ft_label": "en", "ft_score": 0.9, "ft_eligible": True, "language": "de"},
        ]
        before = matched_predictions(rows, 1)
        altered = [{**row, "language": "und"} for row in rows]
        self.assertEqual(before, matched_predictions(altered, 1))
        self.assertEqual(before[0], ["und", "en"])

    def test_cannot_match_never_pads_vetoed_rows(self):
        rows = [{"id": "a", "ft_label": "de", "ft_score": 1.0, "ft_eligible": False}]
        predictions, info = matched_predictions(rows, 1)
        self.assertIsNone(predictions)
        self.assertFalse(info["matched"])

    def test_higher_cutoff_is_a_subset_at_fixed_guard(self):
        rows = [
            {"id": "a", "ft_label": "de", "ft_score": 0.9, "ft_eligible": True},
            {"id": "b", "ft_label": "en", "ft_score": 0.7, "ft_eligible": True},
        ]
        predictions, _ = matched_predictions(rows, 1)
        self.assertEqual(predictions, ["de", "und"])

    def test_native_codes_are_not_relabelled(self):
        model = FakeFastText("__label__als")
        self.assertEqual(ft_vote(model, "A synthetic sentence").best, "als")
        self.assertEqual(model.inputs, [["A synthetic sentence"]])

    def test_letterless_input_does_not_call_fasttext(self):
        model = FakeFastText()
        self.assertEqual(ft_vote(model, "123 !!!").best, "und")
        self.assertEqual(model.inputs, [])

    def test_fasttext_small_probability_overshoot_is_normalized(self):
        prediction = ft_vote(FakeFastText(score=1.000009), "A synthetic sentence")
        self.assertEqual(prediction.confidence, 1.0)

    def test_invalid_fasttext_output_fails(self):
        for label, score in (("de", 0.9), ("__label__", 0.9), ("__label__de", float("nan")), ("__label__de", 1.1)):
            with self.assertRaises(ValueError):
                ft_vote(FakeFastText(label, score), "A synthetic sentence")

    def test_duplicate_ids_fail_without_printing_ticket_text(self):
        case = {"id": "a", "title": "Example", "description": "Help needed", "language": "en"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(json.dumps(case) + "\n" + json.dumps(case) + "\n")
            with self.assertRaises(ValueError):
                load_cases(path)

    def test_report_cannot_replace_any_input(self):
        args = Namespace(input=Path("tickets.jsonl"), model=Path("model.pt"),
                         ft_model=Path("lid.176.bin"), out=None)
        for path in (args.input, args.model, args.ft_model):
            with self.assertRaises(ValueError):
                validate_output(Namespace(**{**vars(args), "out": path}))
        validate_output(Namespace(**{**vars(args), "out": Path("reports/result.json")}))

    def test_committed_fixture_is_complete_and_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "tests/fixtures/authored_tickets.jsonl"
        metadata = json.loads((root / "tests/fixtures/authored_tickets.metadata.json").read_text())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), metadata["file"]["sha256"])
        cases = load_cases(path)
        self.assertEqual(len(cases), 84)
        self.assertEqual(sum(case["language"] == "und" for case in cases), 13)
        self.assertTrue(all(case["authored_by_ai"] and not case["human_verified"] for case in cases))


if __name__ == "__main__":
    unittest.main()
