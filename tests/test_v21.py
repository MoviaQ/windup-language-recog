"""Portable v21 policy regression tests; no evaluation corpus is needed."""
import tempfile
import unittest
from pathlib import Path

import torch

from language_detector import LanguageDetector, build_model


def checkpoint(bias=(1., 0.), arch='linear'):
    model = build_model(2, 32, 4, hidden_dim=8, arch=arch)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor(bias))
    return dict(
        languages=['en', 'pl'],
        config=dict(buckets=32, embedding_dim=4, hidden_dim=8, arch=arch),
        state_dict=model.state_dict(),
        calibration=dict(temperature=1., min_confidence=.5),
    )


class V21Tests(unittest.TestCase):
    def load(self, c):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'model.pt'
        torch.save(c, path)
        return LanguageDetector(path)

    def test_details_include_margin(self):
        d = self.load(checkpoint())
        details = d.predict_details('Hello')
        self.assertIn('margin', details)
        # With two classes the top-2 margin is p_best - p_other.
        self.assertAlmostEqual(details['confidence'], 0.7310586, places=6)
        self.assertAlmostEqual(details['margin'], 0.4621172, places=6)

    def test_short_margin_floor_rejects_but_long_passes(self):
        c = checkpoint()
        c['calibration']['min_margin_by_length'] = {'short': 0.99, 'long': 0.0}
        d = self.load(c)
        short = d.predict_details('Hello')
        self.assertEqual(short['language'], 'und')
        self.assertLess(short['margin'], 0.99)
        self.assertEqual(
            d.predict_details('This is a complete request')['language'], 'en'
        )

    def test_long_margin_floor_rejects_when_above_input(self):
        c = checkpoint()
        c['calibration']['min_margin_by_length'] = {'short': 0.0, 'long': 0.99}
        d = self.load(c)
        self.assertEqual(d.predict_details('Hello')['language'], 'en')
        self.assertEqual(
            d.predict_details('This is a complete request')['language'], 'und'
        )

    def test_single_language_checkpoint_does_not_crash(self):
        model = build_model(1, 32, 4, hidden_dim=8, arch='linear')
        c = dict(
            languages=['en'],
            config=dict(buckets=32, embedding_dim=4, hidden_dim=8, arch='linear'),
            state_dict=model.state_dict(),
            calibration=dict(temperature=1., min_confidence=.5),
        )
        details = self.load(c).predict_details('Hello')
        self.assertEqual(details['language'], 'en')
        self.assertAlmostEqual(details['margin'], details['confidence'])

    def test_checkpoint_without_calibration(self):
        c = checkpoint()
        del c['calibration']
        details = self.load(c).predict_details('Hello')
        self.assertIn('margin', details)
        self.assertEqual(details['language'], details['best_language'])

    def test_backward_compatibility_without_margin_keys(self):
        baseline = self.load(checkpoint())
        self.assertNotIn('min_margin_by_length', baseline.calibration)
        self.assertEqual(baseline.predict_details('Hello')['language'], 'en')
        c = checkpoint()
        c['calibration']['min_margin_by_length'] = {'short': 0.99, 'long': 0.66}
        floored = self.load(c)
        # The margin gate only affects the calibrated detail API.
        self.assertEqual(floored.predict_details('Hello')['language'], 'und')
        self.assertEqual(floored.predict('Hello'), 'en')
        self.assertEqual(
            floored.predict_many(['Hello', 'Another text']), ['en', 'en']
        )

    def test_margin_composes_with_language_length_cutoff(self):
        c = checkpoint()
        c['calibration']['min_confidence_by_language_and_length'] = {'en': {'short': .9}}
        d = self.load(c)
        self.assertEqual(d.predict_details('Hello')['language'], 'und')
        self.assertEqual(
            d.predict_details('This is a complete request')['language'], 'en'
        )
        c = checkpoint()
        c['calibration']['min_margin_by_length'] = {'short': .9}
        d = self.load(c)
        self.assertEqual(d.predict_details('Hello')['language'], 'und')
        self.assertEqual(
            d.predict_details('This is a complete request')['language'], 'en'
        )
        c = checkpoint()
        c['calibration']['min_confidence_by_language_and_length'] = {'en': {'short': .9}}
        c['calibration']['min_margin_by_length'] = {'short': .9}
        d = self.load(c)
        self.assertEqual(d.predict_details('Hello')['language'], 'und')
        self.assertEqual(
            d.predict_details('This is a complete request')['language'], 'en'
        )

    def test_bundled_v21_acceptance(self):
        d = LanguageDetector(Path(__file__).resolve().parents[1] / 'model.pt')
        self.assertEqual(len(d.languages), 100)
        details = d.predict_details('Proszę o pomoc z dostępem do konta.')
        self.assertEqual(details['language'], 'pl')
        self.assertIn('margin', details)
        self.assertEqual(d.predict_ticket(
            'My VPN stopped working today. Please reinstall it.', allow_uncertain=True,
        ), 'en')


if __name__ == '__main__':
    unittest.main()
