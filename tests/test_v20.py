"""Portable v20 policy regression tests; no evaluation corpus is needed."""
import tempfile
import unittest
from pathlib import Path

import torch

from language_detector import (
    LanguageDetector, build_model, evidence_profile, model_from_checkpoint,
)


def checkpoint(bias=(4., 0.), arch='linear'):
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


class V20Tests(unittest.TestCase):
    def load(self, c):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'model.pt'
        torch.save(c, path)
        return LanguageDetector(path)

    def ensemble(self, companion_bias=(0., 2.)):
        c = checkpoint()
        c['ensemble'] = dict(
            companions=[checkpoint(companion_bias, arch='mlp')],
            weights=[3., 1.], temperatures=[2., 1.], require_agreement=True,
        )
        return c

    def test_weighted_logits_and_mixed_architecture(self):
        model = model_from_checkpoint(self.ensemble())
        logits, agreement = model.forward_with_agreement(
            torch.tensor([1, 2]), torch.tensor([0]),
        )
        torch.testing.assert_close(logits, torch.tensor([[1.5, .5]]))
        self.assertFalse(agreement.item())

    def test_disagreement_rejects_without_changing_raw_api(self):
        c = self.ensemble()
        c['state_dict']['classifier.bias'].copy_(torch.tensor([20., 0.]))
        d = self.load(c)
        self.assertGreater(d.predict_details('Hello support')['confidence'], .99)
        self.assertEqual(d.predict_details('Hello support')['language'], 'und')
        self.assertEqual(d.predict('Hello support'), 'en')
        self.assertEqual(d.predict_many(['Hello support', 'Another text']), ['en', 'en'])
        self.assertEqual(d.predict_ticket('Hello support', allow_uncertain=True), 'und')

    def test_agreement_and_empty_ticket(self):
        d = self.load(self.ensemble((2., 0.)))
        self.assertEqual(d.predict_details('Hello support')['language'], 'en')
        self.assertEqual(d.predict_ticket('123 !!!', allow_uncertain=True), 'und')

    def test_incompatible_encoding_and_language_order(self):
        for field, value in (
            ('languages', ['pl', 'en']),
            ('config', dict(buckets=64, embedding_dim=4, hidden_dim=8, arch='mlp')),
        ):
            c = self.ensemble()
            c['ensemble']['companions'][0][field] = value
            with self.assertRaises(ValueError):
                model_from_checkpoint(c)

    def test_invalid_ensemble_weights(self):
        for weights in ([0., 1.], [-1., 1.], [1.]):
            c = self.ensemble()
            c['ensemble']['weights'] = weights
            with self.assertRaises(ValueError):
                model_from_checkpoint(c)

    def test_language_threshold_fallback_and_raw_score(self):
        c = checkpoint((1., 0.))
        c['calibration'].update(min_confidence=.9, min_confidence_by_language={'pl': .6})
        d = self.load(c)
        self.assertEqual(d.predict_details('Hello')['language'], 'und')
        c['calibration']['min_confidence_by_language']['en'] = .7
        d = self.load(c)
        self.assertEqual(d.predict_details('Hello')['language'], 'en')
        self.assertAlmostEqual(d.predict_details('Hello')['confidence'], .7310586, places=6)

    def test_language_and_length_cutoffs(self):
        c = checkpoint((1., 0.))
        c['calibration']['min_confidence_by_language_and_length'] = {
            'en': {'short': .9, 'long': .7},
        }
        d = self.load(c)
        self.assertEqual(d.predict_details('Hello')['language'], 'und')
        self.assertEqual(d.predict_details('This is a complete request')['language'], 'en')
        self.assertEqual(d.predict('Hello'), 'en')

    def test_standalone_ambiguity_preserves_case_and_context(self):
        c = checkpoint()
        c['calibration'].update(ambiguous_single_words=['Widget'], word_evidence_case_sensitive=True)
        d = self.load(c)
        self.assertEqual(d.predict_details('Widget?')['language'], 'und')
        self.assertEqual(d.predict_details('widget')['language'], 'en')
        self.assertEqual(d.predict_details('My Widget stopped working')['language'], 'en')

    def test_distinctive_word_conflict_and_casefold_compatibility(self):
        c = checkpoint()
        c['calibration'].update(single_word_language={'Wort': 'pl'}, word_evidence_case_sensitive=True)
        d = self.load(c)
        self.assertEqual(d.predict_details('Wort')['language'], 'und')
        c['calibration'] = dict(ambiguous_single_words=['widget'], min_confidence=.5)
        d = self.load(c)
        self.assertEqual(d.predict_details('Widget?')['language'], 'und')

    def test_continuous_japanese_and_legacy_linear_mlp(self):
        group, word = evidence_profile('パスワードを変更しても接続できません。')
        self.assertEqual(group, 'long')
        self.assertIsNone(word)
        for arch in ('linear', 'mlp'):
            d = self.load(checkpoint(arch=arch))
            self.assertEqual(d.predict('Hello support'), 'en')
            self.assertFalse(d.model.training)

    def test_bundled_v20_acceptance(self):
        d = LanguageDetector(Path(__file__).resolve().parents[1] / 'model.pt')
        self.assertEqual(d.predict_ticket(
            'My VPN stopped working today. Please reinstall it.', allow_uncertain=True,
        ), 'en')
        self.assertEqual(d.predict_details('Proszę o pomoc z dostępem do konta.')['language'], 'pl')
        self.assertEqual(len(d.languages), 100)


if __name__ == '__main__':
    unittest.main()
