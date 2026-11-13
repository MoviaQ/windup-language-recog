"""Torch-free runtime: equivalence with the PyTorch networks, .npz files, refusals."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import numpy_runtime
from language_detector import LanguageDetector, encode_batch
from tools import export_numpy
from torch_models import (
    LanguageModelEnsemble,
    batch,
    build_model,
    compress_state_dict,
    model_from_checkpoint,
    quantize_rowwise_int8,
)

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ["de", "en", "fr", "pl", "ru"]
TEXTS = ["Hello, I need help signing in.", "Dzień dobry, nie mogę się zalogować.",
         "Bonjour à tous", "Привет, как дела?", "Guten Morgen", "a", "Zażółć gęślą jaźń " * 40]


def tiny(arch="mlp", seed=0, storage="fp32", calibration=None):
    torch.manual_seed(seed)
    model = build_model(len(LANGUAGES), 64, 8, hidden_dim=16, arch=arch)
    config = {"buckets": 64, "embedding_dim": 8, "hidden_dim": 16, "arch": arch, "ngrams": [1, 2, 3],
              "word_features": True}
    state = model.state_dict()
    if storage != "fp32":
        config["weight_storage"] = storage
        state = compress_state_dict(state, storage)
    checkpoint = {"languages": LANGUAGES, "config": config, "state_dict": state}
    if calibration:
        checkpoint["calibration"] = calibration
    return checkpoint


def torch_logits(checkpoint, texts=TEXTS):
    model = model_from_checkpoint(checkpoint).eval()
    options = numpy_runtime.encoding_options(checkpoint["config"])
    with torch.inference_mode():
        inputs = batch(texts, **options)
        if isinstance(model, LanguageModelEnsemble):
            logits, agree = model.forward_with_agreement(*inputs)
            return logits.numpy(), agree.numpy()
        return model(*inputs).numpy(), np.ones(len(texts), dtype=bool)


def numpy_logits(detector, texts=TEXTS):
    return detector.model.logits_with_agreement(*encode_batch(texts, **detector.encoding_options))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)

    def save_pt(self, checkpoint, name="model.pt"):
        path = self.dir / name
        torch.save(checkpoint, path)
        return path

    def export(self, pt, storage="keep", name=None):
        out = self.dir / (name or f"{pt.stem}-{storage}.npz")
        self.assertEqual(export_numpy.main(["--input", str(pt), "--output", str(out), "--storage", storage]), 0)
        return out

    def test_matches_pytorch_for_each_architecture_and_storage(self):
        for arch in ("linear", "mlp"):
            for storage in ("fp32", "fp16", "int8_rowwise"):
                with self.subTest(arch=arch, storage=storage):
                    checkpoint = tiny(arch, storage=storage)
                    pt = self.save_pt(checkpoint, f"{arch}-{storage}.pt")
                    expected, _ = torch_logits(checkpoint)
                    from_pt = LanguageDetector(pt)
                    from_npz = LanguageDetector(self.export(pt))
                    got_pt, _ = numpy_logits(from_pt)
                    got_npz, _ = numpy_logits(from_npz)
                    np.testing.assert_allclose(got_pt, expected, rtol=0, atol=1e-5)
                    self.assertTrue(np.array_equal(got_pt.argmax(1), expected.argmax(1)))
                    self.assertTrue(np.array_equal(got_pt, got_npz))  # .pt and .npz: same bits

    def test_ensemble_agreement_and_policy_survive_export(self):
        calibration = {"temperature": 1.5, "min_confidence": 0.3, "min_margin_by_length": {"short": 0.1},
                       "min_confidence_by_language_and_length": {"en": {"long": 0.4}}}
        checkpoint = tiny("linear", seed=1, calibration=calibration)
        checkpoint["ensemble"] = {"companions": [tiny("mlp", seed=2, storage="int8_rowwise")],
                                  "weights": [3.0, 1.0], "temperatures": [2.0, 1.0], "require_agreement": True}
        checkpoint["training"] = {"note": "kept as source metadata", "epochs": [1, 2]}
        pt = self.save_pt(checkpoint)
        expected, expected_agree = torch_logits(checkpoint)
        detector = LanguageDetector(self.export(pt))
        got, agree = numpy_logits(detector)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-5)
        self.assertTrue(np.array_equal(agree, expected_agree))
        self.assertFalse(agree.all(), "fixture should contain a disagreement")
        self.assertEqual(detector.calibration, calibration)
        legacy = LanguageDetector(pt)
        self.assertEqual([detector.predict_details(t) for t in TEXTS], [legacy.predict_details(t) for t in TEXTS])
        with np.load(self.dir / "model-keep.npz") as data:
            meta = json.loads(bytes(data[numpy_runtime.META_KEY]).decode())
        self.assertEqual(meta["source_metadata"]["training"], checkpoint["training"])
        self.assertEqual(meta["export"]["source_file"], "model.pt")

    def test_int8_quantizer_matches_pytorch_bit_for_bit(self):
        weight = tiny()["state_dict"]["embedding.weight"].clone()
        weight[3].zero_()
        torch_values, torch_scale = quantize_rowwise_int8(weight)
        values, scale = numpy_runtime.quantize_rowwise_int8(weight.numpy())
        self.assertTrue(np.array_equal(values, torch_values.numpy()))
        self.assertTrue(np.array_equal(scale, torch_scale.numpy()))
        pt = self.save_pt(tiny())
        exported = LanguageDetector(self.export(pt, "int8_rowwise"))
        torch_tool = LanguageDetector(self.save_pt(tiny(storage="int8_rowwise"), "int8.pt"))
        self.assertTrue(np.array_equal(numpy_logits(exported)[0], numpy_logits(torch_tool)[0]))
        self.assertEqual(exported.model.embedding.weight.dtype, np.int8)

    def test_npz_runs_without_pytorch(self):
        npz = self.export(self.save_pt(tiny(calibration={"temperature": 1.0, "min_confidence": 0.1})))
        pt = self.dir / "model.pt"
        script = (
            "import sys; sys.modules['torch'] = None\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from language_detector import LanguageDetector\n"
            f"d = LanguageDetector({str(npz)!r})\n"
            "print(d.predict('Hello there'), d.predict_details('Bonjour')['calibrated'],"
            " d.predict_many(['Hello', 'Привет']))\n"
            "try:\n"
            f"    LanguageDetector({str(pt)!r})\n"
            "except ImportError as error:\n"
            "    print('pt-refused', 'requirements-training.txt' in str(error))\n"
            "print('torch' in sys.modules and sys.modules['torch'] is not None)\n"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=self.dir)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        reference = LanguageDetector(npz)
        self.assertEqual(lines[0], f"{reference.predict('Hello there')} True {reference.predict_many(['Hello', 'Привет'])}")
        self.assertEqual(lines[1], "pt-refused True")
        self.assertEqual(lines[2], "False")

    def test_refuses_malformed_files(self):
        meta, arrays = numpy_runtime.split_checkpoint(tiny())
        good = self.dir / "good.npz"
        numpy_runtime.save_npz(good, meta, arrays)
        numpy_runtime.load_npz(good)

        def write(name, meta_value, member_arrays, extra=None):
            payload = {numpy_runtime.META_KEY: np.frombuffer(json.dumps(meta_value).encode(), dtype=np.uint8)}
            for i, a in enumerate(member_arrays):
                payload.update({f"member{i}/{k}": v for k, v in a.items()})
            payload.update(extra or {})
            path = self.dir / name
            np.savez(path, **payload)
            return path

        versioned = dict(meta, format=numpy_runtime.FORMAT, format_version=numpy_runtime.FORMAT_VERSION)
        with self.assertRaisesRegex(ValueError, "unsupported format"):
            numpy_runtime.load_npz(write("v2.npz", dict(versioned, format_version=2), arrays))
        with self.assertRaisesRegex(ValueError, "unexpected array"):
            numpy_runtime.load_npz(write("extra.npz", versioned, arrays, {"member7/x": np.zeros(1)}))
        wrong_rows = [dict(arrays[0], **{"embedding.weight": arrays[0]["embedding.weight"][:10]})]
        with self.assertRaisesRegex(ValueError, "rows"):
            numpy_runtime.load_npz(write("rows.npz", versioned, wrong_rows))
        missing = [{k: v for k, v in arrays[0].items() if k != "hidden.bias"}]
        with self.assertRaisesRegex(ValueError, "missing"):
            numpy_runtime.load_npz(write("missing.npz", versioned, missing))
        with self.assertRaisesRegex(ValueError, "JSON round trip"):
            numpy_runtime.save_npz(self.dir / "tuple.npz", dict(meta, calibration={"x": (1, 2)}), arrays)
        bad_extra = tiny()
        bad_extra["training"] = {"tensor": torch.zeros(2)}
        with self.assertRaisesRegex(ValueError, "not plain JSON"):
            export_numpy.export(bad_extra, "keep", {})
        with self.assertRaisesRegex(ValueError, "at least one feature"):
            numpy_runtime.Embedding(np.zeros((4, 2), np.float32)).mean(np.array([1, 2]), np.array([0, 2]))
        with self.assertRaisesRegex(ValueError, "outside"):
            numpy_runtime.Embedding(np.zeros((4, 2), np.float32)).mean(np.array([9]), np.array([0]))


if __name__ == "__main__":
    unittest.main()
