import unittest

import csv
import tempfile
import time
from pathlib import Path

try:
    import torch
    from train_model import (
        EncodedDataset,
        StreamingDataset,
        encoding_config,
        temperature_row_probabilities,
    )
except ImportError as error:  # numpy is a training-only dependency
    temperature_row_probabilities = None
    IMPORT_ERROR = error


@unittest.skipIf(
    temperature_row_probabilities is None,
    "train_model needs requirements-training.txt (numpy)",
)
class TemperatureSamplingTests(unittest.TestCase):
    def test_language_shares_follow_alpha(self):
        # 900 rows of class 0, 100 rows of class 1.
        targets = torch.tensor([0] * 900 + [1] * 100)
        probabilities = temperature_row_probabilities(targets, [900, 100], 0.5)
        self.assertAlmostEqual(probabilities.sum().item(), 1.0, places=6)
        share_0 = probabilities[targets == 0].sum().item()
        self.assertAlmostEqual(share_0, 30 / (30 + 10), places=6)

    def test_alpha_one_keeps_natural_distribution(self):
        targets = torch.tensor([0] * 3 + [1])
        probabilities = temperature_row_probabilities(targets, [3, 1], 1.0)
        self.assertTrue(torch.allclose(probabilities, torch.full((4,), 0.25)))

    def test_missing_class_is_refused(self):
        with self.assertRaises(ValueError):
            temperature_row_probabilities(torch.tensor([0]), [1, 0], 0.3)


@unittest.skipIf(
    temperature_row_probabilities is None,
    "train_model needs requirements-training.txt (numpy)",
)
class StreamingDatasetTests(unittest.TestCase):
    CONFIG = {
        "buckets": 4096, "ngrams": [1, 2, 3], "word_features": True,
        "preprocessing": "ticket_v2", "embedding_dim": 16, "arch": "mlp",
    }

    def test_streaming_batches_equal_cached_batches(self):
        rows = [("Dzień dobry, jak się masz?", "pl"), ("Good morning, how are you?", "en"),
                ("Guten Morgen, wie geht es?", "de"), ("Ahoj, jak se máš dnes?", "cs")] * 5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["text", "language"])
                writer.writerows(rows)
            languages = ["cs", "de", "en", "pl"]
            cached = EncodedDataset([path], languages, self.CONFIG, 384)
            streamed = StreamingDataset([path], languages, self.CONFIG, 384, workers=2)
            self.assertTrue(torch.equal(cached.targets, streamed.targets))
            self.assertEqual(cached.counts, streamed.counts)
            order = torch.randperm(len(cached), generator=torch.Generator().manual_seed(0)).split(6)
            expected, got = list(cached.iter_batches(order)), list(streamed.iter_batches(order))
            self.assertEqual(len(expected), len(got))
            for (i1, a), (i2, b) in zip(expected, got):
                self.assertTrue(torch.equal(i1, i2))
                self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))

    def test_streaming_prefetch_is_bounded(self):
        rows = [(f"Zdanie numer {i} jest po polsku.", "pl") for i in range(400)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["text", "language"])
                writer.writerows(rows)
            streamed = StreamingDataset([path], ["pl"], self.CONFIG, 384, workers=2)
            seen = []
            for indices, _ in streamed.iter_batches(torch.arange(len(streamed)).split(4), prefetch=3):
                time.sleep(0.002)  # consumer slower than the encoders
                seen.append(indices)
            self.assertEqual(torch.cat(seen).tolist(), list(range(len(rows))))
            self.assertEqual(streamed.max_in_flight, 3)
            with self.assertRaises(ValueError):
                next(streamed.iter_batches(torch.arange(4).split(2), prefetch=-1))

    def test_cache_key_ignores_model_size(self):
        bigger = dict(self.CONFIG, embedding_dim=64, hidden_dim=256, arch="linear")
        self.assertEqual(encoding_config(self.CONFIG), encoding_config(bigger))
        self.assertNotEqual(encoding_config(self.CONFIG), encoding_config(dict(self.CONFIG, buckets=8192)))


if __name__ == "__main__":
    unittest.main()
