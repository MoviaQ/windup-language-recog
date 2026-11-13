import unittest
from pathlib import Path

from evaluate_public import release_policy_keys

ROOT = Path(__file__).resolve().parents[1]


class EvaluatePublicGuardTests(unittest.TestCase):
    def test_release_policy_checkpoint_is_refused(self):
        # Same calibration keys as the v21 release checkpoint (no longer bundled).
        checkpoint = {"calibration": {"temperature": 1.0, "min_margin_by_length": {"short": 0.88},
                                      "min_confidence_by_language": {"en": 0.9}},
                      "ensemble": {"companions": []}}
        self.assertEqual(release_policy_keys(checkpoint),
                         ["ensemble", "min_confidence_by_language", "min_margin_by_length"])

    def test_plain_retrained_checkpoint_is_accepted(self):
        checkpoint = {"calibration": {"temperature": 1.2, "min_confidence": 0.9}}
        self.assertEqual(release_policy_keys(checkpoint), [])


if __name__ == "__main__":
    unittest.main()
