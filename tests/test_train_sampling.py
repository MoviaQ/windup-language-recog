import unittest

try:
    import torch
    from train_model import temperature_row_probabilities
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


if __name__ == "__main__":
    unittest.main()
