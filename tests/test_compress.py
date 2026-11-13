"""Compressed weight storage: round trip, old checkpoints unchanged, refusals."""
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
import torch

from language_detector import (
    LanguageDetector,
    batch,
    build_model,
    compress_state_dict,
    decompress_state_dict,
    encode_batch,
    model_from_checkpoint,
)

from tools import compress_checkpoint

ROOT = Path(__file__).resolve().parents[1]

TEXTS = ["Hello, I need help signing in.", "Dzień dobry, nie mogę się zalogować.",
         "Bonjour à tous", "Привет, как дела?", "Hola amigo, ¿cómo estás?"]


def tiny_checkpoint(arch="mlp"):
    torch.manual_seed(0)
    model = build_model(5, 64, 8, hidden_dim=16, arch=arch)
    return {
        "languages": ["en", "es", "fr", "pl", "ru"],
        "config": {"buckets": 64, "embedding_dim": 8, "hidden_dim": 16, "arch": arch, "ngrams": [1, 2, 3]},
        "state_dict": model.state_dict(),
    }


def logits(model, texts=TEXTS, buckets=64):
    with torch.inference_mode():
        return model.eval()(*batch(texts, buckets=buckets))


def np_logits(detector, texts=TEXTS):
    return detector.model.logits(*encode_batch(texts, **detector.encoding_options))


class CompressTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)

    def test_round_trip_through_tool_and_detector(self):
        source = self.dir / "fp32.pt"
        torch.save(tiny_checkpoint(), source)
        reference = LanguageDetector(source)
        for storage, tolerance in (("fp16", 1e-3), ("int8_rowwise", 2e-2)):
            output = self.dir / f"{storage}.pt"
            self.assertEqual(compress_checkpoint.main(
                ["--input", str(source), "--output", str(output), "--storage", storage]), 0)
            stored = torch.load(output, map_location="cpu", weights_only=True)
            self.assertEqual(stored["config"]["weight_storage"], storage)
            expected_dtype = torch.float16 if storage == "fp16" else torch.int8
            self.assertEqual(stored["state_dict"]["embedding.weight"].dtype, expected_dtype)
            detector = LanguageDetector(output)
            # The numpy runtime keeps the stored dtype in memory (no float32 copy).
            self.assertEqual(detector.model.embedding.weight.dtype, {"fp16": np.float16, "int8_rowwise": np.int8}[storage])
            difference = abs(np_logits(detector) - np_logits(reference)).max()
            self.assertLess(difference, tolerance, storage)
            self.assertEqual([detector.predict(t) for t in TEXTS], [reference.predict(t) for t in TEXTS])

    def test_int8_rowwise_error_is_bounded_by_half_a_step(self):
        weight = tiny_checkpoint()["state_dict"]["embedding.weight"]
        weight[3].zero_()  # an all-zero row must survive
        stored = compress_state_dict({"embedding.weight": weight}, "int8_rowwise")
        restored = decompress_state_dict(stored, "int8_rowwise")["embedding.weight"]
        scale = stored["embedding.weight.__row_scale"]
        self.assertTrue(torch.all((restored - weight).abs() <= scale / 2 + 1e-7))
        self.assertTrue(torch.equal(restored[3], torch.zeros_like(restored[3])))

    def test_checkpoint_without_storage_key_loads_unchanged(self):
        checkpoint = tiny_checkpoint()
        model = model_from_checkpoint(checkpoint)
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, checkpoint["state_dict"][name]), name)

    def test_unknown_storage_is_refused(self):
        checkpoint = tiny_checkpoint()
        checkpoint["config"]["weight_storage"] = "int4"
        with self.assertRaisesRegex(ValueError, "Unknown weight storage"):
            model_from_checkpoint(checkpoint)
        with self.assertRaisesRegex(ValueError, "Unknown weight storage"):
            compress_state_dict(tiny_checkpoint()["state_dict"], "int4")

    def test_storage_dtype_mismatch_is_refused(self):
        checkpoint = tiny_checkpoint()
        checkpoint["config"]["weight_storage"] = "fp16"  # tensors are still float32
        with self.assertRaisesRegex(ValueError, "non-float16"):
            model_from_checkpoint(checkpoint)
        stored = compress_state_dict(tiny_checkpoint()["state_dict"], "int8_rowwise")
        del stored["embedding.weight.__row_scale"]
        with self.assertRaisesRegex(ValueError, "row scale"):
            decompress_state_dict(stored, "int8_rowwise")
        stored = compress_state_dict(tiny_checkpoint()["state_dict"], "int8_rowwise")
        stored["hidden.weight.__row_scale"] = torch.ones(16, 1)
        with self.assertRaisesRegex(ValueError, "without an int8 tensor"):
            decompress_state_dict(stored, "int8_rowwise")

    def test_tool_refuses_overwrite_ensemble_and_recompression(self):
        source = self.dir / "fp32.pt"
        torch.save(tiny_checkpoint(), source)
        before = source.read_bytes()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            compress_checkpoint.main(["--input", str(source), "--output", str(source), "--storage", "fp16"])
        self.assertEqual(source.read_bytes(), before)
        ensemble = tiny_checkpoint()
        ensemble["ensemble"] = {"companions": [tiny_checkpoint("linear")], "weights": [1, 1], "temperatures": [1, 1]}
        with self.assertRaisesRegex(ValueError, "Ensemble"):
            compress_checkpoint.compress_checkpoint(ensemble, "fp16")
        compressed = compress_checkpoint.compress_checkpoint(tiny_checkpoint(), "fp16")
        with self.assertRaisesRegex(ValueError, "already stored"):
            compress_checkpoint.compress_checkpoint(compressed, "int8_rowwise")


if __name__ == "__main__":
    unittest.main()
