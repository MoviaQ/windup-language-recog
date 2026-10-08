"""Exact feature compatibility, including multilingual and error cases."""

import hashlib
import random
import unittest
from unittest.mock import patch

import language_detector as detector


@unittest.skipIf(detector._native_char_features is None, "Optional encoder not built")
class NativeEncoderTests(unittest.TestCase):
    def test_unicode_boundaries_and_digest_size(self):
        native = detector._native_char_features
        rng = random.Random(231)
        alphabet = " abcąłęЖЯ字語عمرحباéßΩ한😀e\u0301\x00"
        texts = ["", "A", "Zażółć gęślą jaźń", "字語", "مرحبا", "😀a"]
        texts += [
            "".join(rng.choices(alphabet, k=rng.randrange(1, 100))) for _ in range(300)
        ]
        for text in texts:
            for grams in ((1, 2, 3), (1, 2, 3, 4, 5), (2, 7)):
                for buckets in (1, 8192, 131072, 2**64 - 1):
                    expected = [
                        int.from_bytes(
                            hashlib.blake2b(
                                text[i : i + n].encode(), digest_size=8
                            ).digest(),
                            "little",
                        )
                        % buckets
                        for n in grams
                        for i in range(len(text) - n + 1)
                    ]
                    self.assertEqual(native(text, grams, buckets), expected)

    def test_full_encoding_preserves_preprocessing_and_words(self):
        texts = [
            "Nie mogę się zalogować po update.",
            "FranÃ§ais &amp; français",
            "مرحبا بالعالم",
            "字語 😀",
            "mail@example.com Hello world",
        ]
        for text in texts:
            for preprocessing in ("none", "ticket_v2"):
                options = {
                    "buckets": 131072,
                    "ngrams": (1, 2, 3, 4, 5),
                    "word_features": True,
                    "preprocessing": preprocessing,
                }
                expected = detector.encode(text, **options)
                with patch.object(detector, "_native_char_features", None):
                    self.assertEqual(expected, detector.encode(text, **options))

    def test_fragment_cache_eviction_is_exact(self):
        import _windup_native

        # These distinct fragments occupy the same direct-mapped cache slot.
        for text in ("word899", "word1454", "word899") * 20:
            expected = (
                int.from_bytes(
                    hashlib.blake2b(text.encode(), digest_size=8).digest(), "little"
                )
                % 131072
            )
            self.assertEqual(
                detector._native_char_features(text, (len(text),), 131072),
                [expected],
            )
        _windup_native.cache_clear()
        self.assertEqual(
            detector._native_char_features("word899", (7,), 131072),
            [
                int.from_bytes(
                    hashlib.blake2b(b"word899", digest_size=8).digest(), "little"
                )
                % 131072
            ],
        )

    def test_invalid_native_arguments(self):
        native = detector._native_char_features
        for buckets in (-1, 2**64):
            with self.assertRaises(OverflowError):
                native("abc", (1,), buckets)
        with self.assertRaises(ValueError):
            native("abc", (1,), 0)
        for grams in ((0,), (-1,)):
            with self.assertRaises(ValueError):
                native("abc", grams, 8192)
        with self.assertRaises(TypeError):
            native("abc", ("x",), 8192)
        with self.assertRaises(UnicodeEncodeError):
            native("a\ud800", (1,), 8192)


if __name__ == "__main__":
    unittest.main()
