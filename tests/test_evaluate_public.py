import unittest
from pathlib import Path

import torch

from evaluate_public import release_policy_keys

ROOT = Path(__file__).resolve().parents[1]


class EvaluatePublicGuardTests(unittest.TestCase):
    def test_bundled_release_checkpoint_is_refused(self):
        checkpoint = torch.load(ROOT / "model.pt", map_location="cpu", weights_only=True)
        self.assertIn("min_margin_by_length", release_policy_keys(checkpoint))

    def test_plain_retrained_checkpoint_is_accepted(self):
        checkpoint = {"calibration": {"temperature": 1.2, "min_confidence": 0.9}}
        self.assertEqual(release_policy_keys(checkpoint), [])


if __name__ == "__main__":
    unittest.main()
