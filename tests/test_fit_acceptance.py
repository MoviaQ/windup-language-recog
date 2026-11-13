"""Acceptance-policy fitting tool: threshold rule, inference parity and split guards.

Small synthetic checkpoints only; no corpus, FLORES or download is needed.
"""

import csv
import io
import random
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch

from language_detector import LanguageDetector, build_model
from tools import fit_acceptance as fa

LANGUAGES = ["de", "en", "pl"]
WORDS = ["hallo", "hello", "dzien", "dobry", "guten", "morgen", "good", "morning",
         "prosze", "bitte", "please", "konto", "account", "zugang", "access", "dostep"]


def tiny_checkpoint(seed=0):
    torch.manual_seed(seed)
    model = build_model(len(LANGUAGES), 64, 8, hidden_dim=8, arch="mlp")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(4.0)
    return {"languages": LANGUAGES, "config": {"buckets": 64, "embedding_dim": 8, "hidden_dim": 8, "arch": "mlp"},
            "state_dict": model.state_dict()}


def random_texts(count, seed):
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 7))) for _ in range(count)]


def write_csv(path, texts, labels):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["text", "language"])
        writer.writerows(zip(texts, labels))


def arrays_for(detector, texts, gold, negative_weight=1.0):
    fa._DETECTOR = detector
    logits, errors = fa._logits_chunk(texts)
    assert not any(errors)
    lengths, _ = fa.evidence_profiles(texts)
    data = fa.FitData()
    data.extend(texts, gold, texts, "synthetic")
    return fa.build_arrays(logits, detector.calibration.get("temperature", 1.0), data, LANGUAGES, lengths,
                           negative_weight)


class BestThresholdTests(unittest.TestCase):
    def test_accepts_prefix_with_highest_gain(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        gains = np.array([1.0, 1.0, -25.0, 1.0])
        self.assertEqual(fa.best_threshold(scores, gains), 0.8)

    def test_ties_are_never_split(self):
        scores = np.array([0.9, 0.8, 0.8])
        gains = np.array([1.0, 1.0, -25.0])
        # Cutting between the two 0.8 values is impossible at inference (>=).
        self.assertEqual(fa.best_threshold(scores, gains), 0.9)

    def test_rejects_everything_when_no_prefix_pays(self):
        self.assertEqual(fa.best_threshold(np.array([0.9, 0.5]), np.array([-25.0, 1.0])), fa.REJECT_ALL)

    def test_lambda_sets_the_marginal_precision(self):
        # 20 correct then one error then 30 correct: worth accepting only if lambda < 30.
        scores = np.linspace(1.0, 0.5, 51)
        correct = np.ones(51, dtype=bool)
        correct[20] = False
        for lam, expected in ((10.0, scores[-1]), (40.0, scores[19])):
            gains = np.where(correct, 1.0, -lam)
            self.assertEqual(fa.best_threshold(scores, gains), expected)


class SimulationParityTests(unittest.TestCase):
    """The vectorised simulation must make the same decision as predict_details."""

    def test_simulate_matches_predict_details(self):
        texts = random_texts(400, seed=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m.pt"
            checkpoint = tiny_checkpoint()
            policy = {
                "temperature": 1.3,
                "min_confidence": 0.5,
                "min_confidence_by_language": {"de": 0.7, "en": 0.4},
                # contextual cells override the language threshold, as at inference
                "min_confidence_by_language_and_length": {"de": {"short": 0.45}, "pl": {"long": 0.8}},
                "min_margin_by_length": {"short": 0.2, "long": 0.05},
            }
            checkpoint["calibration"] = policy
            torch.save(checkpoint, path)
            detector = LanguageDetector(path)
            arrays = arrays_for(detector, texts, ["en"] * len(texts))
            expected = fa.simulate(policy, LANGUAGES, arrays)
            actual = np.array([not detector.predict_details(t)["uncertain"] for t in texts])
        self.assertGreater(expected.sum(), 0)
        self.assertLess(expected.sum(), len(texts))
        np.testing.assert_array_equal(expected, actual)

    def test_fitted_policy_keys_are_known_to_inference(self):
        rng = np.random.default_rng(0)
        n = 600
        arrays = fa.Arrays(rng.integers(0, 3, n), rng.uniform(0.3, 1, n), rng.uniform(0, 1, n),
                           rng.integers(0, 2, n).astype(np.int8), rng.uniform(0, 1, n) < 0.9, np.ones(n))
        for components in fa.CANDIDATES.values():
            policy = fa.fit_policy(components, LANGUAGES, arrays, 25.0, 20)
            self.assertLessEqual(set(policy), fa.POLICY_KEYS)


class SplitGuardTests(unittest.TestCase):
    def test_reporting_file_names_are_refused(self):
        for name in ("test_short.csv", "eng_Latn.devtest", "public_test.csv", "devtest/x.csv"):
            with self.assertRaises(fa.SplitPolicyError, msg=name):
                fa.assert_fitting_path(Path("/data") / name)
        for name in ("valid_short.csv", "valid.csv", "negatives/nn.csv", "latest/valid.csv"):
            fa.assert_fitting_path(Path("/data") / name)

    def test_flores_devtest_is_never_fitted(self):
        with self.assertRaises(fa.SplitPolicyError):
            fa.flores_fit_items(Path("/nonexistent"), "devtest", LANGUAGES)
        with self.assertRaises(fa.SplitPolicyError):
            fa.flores_negative_items(Path("/nonexistent"), "devtest", "nno_Latn")

    def test_negative_partitions_are_disjoint_and_complete(self):
        from dataset_utils import text_key

        texts = sorted(set(random_texts(600, seed=5)))[:400]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nn.csv"
            write_csv(path, texts, ["nn"] * len(texts))
            fitted, _, info = fa.load_negatives(path, set(LANGUAGES), 0.5)
        reporting = [t for t in texts if fa.negative_partition(text_key(t)) >= 0.5]
        self.assertFalse(set(fitted) & set(reporting))
        self.assertEqual(len(fitted) + len(reporting), len(texts))
        self.assertTrue(100 < len(fitted) < 300)
        self.assertEqual(info["fitting_rows"], len(fitted))

    def test_negatives_with_model_labels_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neg.csv"
            write_csv(path, ["hello"], ["en"])
            with self.assertRaises(ValueError):
                fa.load_negatives(path, set(LANGUAGES), 1.0)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.model = self.root / "base.pt"
        torch.save(tiny_checkpoint(), self.model)
        # Labels follow the base model with 15% noise, so the data carry signal
        # (random labels would drive the temperature to infinity).
        rng = random.Random(3)
        texts = sorted(set(random_texts(1200, seed=2)))[:900]
        predicted = LanguageDetector(self.model).predict_many(texts)
        labels = [p if rng.random() > 0.15 else rng.choice(LANGUAGES) for p in predicted]
        self.valid = self.root / "valid.csv"
        write_csv(self.valid, texts[:600], labels[:600])
        self.test = self.root / "test_short.csv"
        write_csv(self.test, texts[600:], labels[600:])

    def run_tool(self, *extra, out="out.pt"):
        argv = ["--model", str(self.model), "--out", str(self.root / out), "--report", str(self.root / (out + ".json")),
                "--min-samples", "20", "--workers", "1", "--verify-sample", "200", *extra]
        stream = io.StringIO()
        with redirect_stdout(stream), redirect_stderr(stream):
            code = fa.main(argv)
        return code, stream.getvalue()

    def test_fitting_on_a_test_split_is_refused(self):
        code, output = self.run_tool("--csv", f"public_test={self.test}")
        self.assertEqual(code, 2)
        self.assertIn("reporting-only", output)
        self.assertFalse((self.root / "out.pt").exists())

    def test_overlap_with_reporting_data_is_refused_unless_dropped(self):
        leaked = self.root / "valid_leak.csv"
        with self.test.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))[:5]
        write_csv(leaked, [r["text"] for r in rows] + ["unique words here"],
                  [r["language"] for r in rows] + ["en"])
        code, output = self.run_tool("--csv", f"leak={leaked}", "--csv", f"v={self.valid}",
                                     "--reporting-csv", str(self.test))
        self.assertEqual(code, 2)
        self.assertIn("also occur in reporting-only data", output)
        code, output = self.run_tool("--csv", f"leak={leaked}", "--csv", f"v={self.valid}",
                                     "--reporting-csv", str(self.test), "--drop-reporting-overlap")
        self.assertEqual(code, 0, output)
        fit = torch.load(self.root / "out.pt", weights_only=True)["acceptance_fit"]
        self.assertEqual(fit["reporting_overlap_dropped"], {"leak": 5})

    def test_reporting_data_cannot_move_thresholds(self):
        other = self.root / "other_test.csv"
        write_csv(other, ["zzz qqq"], ["en"])
        self.assertEqual(self.run_tool("--csv", f"v={self.valid}", "--reporting-csv", str(self.test), out="a.pt")[0], 0)
        self.assertEqual(self.run_tool("--csv", f"v={self.valid}", "--reporting-csv", str(other), out="b.pt")[0], 0)
        a = torch.load(self.root / "a.pt", weights_only=True)["calibration"]
        b = torch.load(self.root / "b.pt", weights_only=True)["calibration"]
        self.assertEqual(a, b)

    def test_output_checkpoint_records_policy_and_provenance(self):
        negatives = self.root / "nn.csv"
        write_csv(negatives, random_texts(300, seed=9), ["nn"] * 300)
        code, output = self.run_tool("--csv", f"v={self.valid}", "--negatives", f"nn={negatives}")
        self.assertEqual(code, 0, output)
        checkpoint = torch.load(self.root / "out.pt", weights_only=True)
        calibration, fit = checkpoint["calibration"], checkpoint["acceptance_fit"]
        self.assertLessEqual(set(calibration), fa.CALIBRATION_KEYS)
        self.assertIn("temperature", calibration)
        self.assertIn("No devtest or test data", calibration["rule"])
        self.assertEqual([s["name"] for s in fit["sources"]], ["v", "nn"])
        self.assertEqual(fit["sources"][0]["rows"], 600)
        self.assertLess(fit["sources"][1]["rows"], 300)  # only the fitting share of negatives
        self.assertEqual(set(fit["cross_fit"]), set(fa.CANDIDATES))
        self.assertGreater(fit["verification"]["checked"], 0)
        # The written checkpoint loads and answers through the normal API.
        detector = LanguageDetector(self.root / "out.pt")
        self.assertIn(detector.predict_details("hello good morning")["language"], LANGUAGES + ["und"])

    def test_negative_scale_zero_only_measures_negatives(self):
        negatives = self.root / "nn.csv"
        write_csv(negatives, random_texts(400, seed=11), ["nn"] * 400)
        self.assertEqual(self.run_tool("--csv", f"v={self.valid}", out="plain.pt")[0], 0)
        code, output = self.run_tool("--csv", f"v={self.valid}", "--negatives", f"nn={negatives}",
                                     "--negative-scale", "0", out="measured.pt")
        self.assertEqual(code, 0, output)
        plain = torch.load(self.root / "plain.pt", weights_only=True)
        measured = torch.load(self.root / "measured.pt", weights_only=True)
        strip = lambda c: {k: v for k, v in c.items() if k != "rule"}
        self.assertEqual(strip(plain["calibration"]), strip(measured["calibration"]))
        self.assertGreater(measured["acceptance_fit"]["fitted_data_metrics_in_sample"]["all"]["negatives"], 0)

    def test_calibrated_base_is_refused(self):
        checkpoint = tiny_checkpoint()
        checkpoint["calibration"] = {"temperature": 1.0}
        torch.save(checkpoint, self.model)
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                fa.main(["--model", str(self.model), "--out", str(self.root / "o.pt"),
                         "--report", str(self.root / "o.json"), "--csv", f"v={self.valid}"])


class SelectionTests(unittest.TestCase):
    def test_simpler_candidate_wins_ties(self):
        table = {name: {"objective": 10.0} for name in fa.CANDIDATES}
        self.assertEqual(fa.select_candidate(table), "none")
        table["language+length"]["objective"] = 10.5
        self.assertEqual(fa.select_candidate(table), "language+length")


if __name__ == "__main__":
    unittest.main()
