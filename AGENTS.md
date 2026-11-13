# Wind-Up Language Recognition

**Version:** 0.1.0 (model v22) | **Port:** none (local CLI/library) | **Stack:** Python 3.10+, numpy, ftfy (PyTorch for training)

## What

A CPU language classifier for 100 language codes. It hashes character n-grams and whole words into a mean-pooled embedding, followed by an MLP. The v22 release (`model.npz`) is one int8 network trained on the multidomain corpus. Its acceptance policy is fitted per language and length group. Inference runs on numpy; PyTorch is used only for training and for reading legacy `.pt` checkpoints.

## Quick start

```bash
./setup.sh
.venv/bin/python language_detector.py predict "Hello, I need help signing in."
.venv/bin/python -m unittest discover -s tests
```

## Commands

```bash
# Runtime dependencies and syntax checks
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m compileall -q language_detector.py numpy_runtime.py torch_models.py ticket_text.py ticket_guard.py train_model.py prepare_public_data.py prepare_multidomain_data.py dataset_utils.py evaluate_public.py tools
# Training needs PyTorch; downloads data, then performs a substantial CPU job
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python prepare_public_data.py
.venv/bin/python train_model.py --short-only --epochs 12 --patience 4 --batch-size 512 --buckets 131072 --embedding-dim 64 --ngram-max 5 --word-features --preprocessing ticket_v2 --dropout 0.05 --lr 0.003 --model model-retrained.pt
.venv/bin/python evaluate_public.py --model model-retrained.pt
# Small educational exercise; do not overwrite release weights
.venv/bin/python language_detector.py train --data data/demo.csv --epochs 120 --model model-demo.pt
# v22 release pipeline (hours of CPU; see docs/experiments.md, Exp D/E and "v22 release")
.venv/bin/python prepare_multidomain_data.py --cap 150000 --workers 3 --out data/prepared_v3
.venv/bin/python train_model.py --data-dir data/prepared_v3 --valid-csv reports/flores_dev.csv --stream-workers 3 ...  # full flags in docs/experiments.md
.venv/bin/python tools/compress_checkpoint.py --input <trained.pt> --output <int8.pt> --storage int8_rowwise
.venv/bin/python tools/fit_acceptance.py --model <int8.pt> --out <calibrated.pt> ...  # full flags in docs/experiments.md
.venv/bin/python tools/export_numpy.py --input <calibrated.pt> --output model.npz
```

## Architecture

```text
language_detector.py      Encoding, inference API, CLI (no PyTorch import)
numpy_runtime.py          numpy networks, .npz release format, int8 storage
torch_models.py           PyTorch networks, demo trainer, .pt storage helpers
ticket_text.py            Local text cleanup and description extraction
ticket_guard.py           Optional frozen sentence veto for support tickets
train_model.py            Cached features, public validation, checkpoint selection
prepare_public_data.py    Public downloads, split handling, short-text preparation
evaluate_public.py        Validation temperature fit, public evaluation, latency
dataset_utils.py          CSV helpers and normalized duplicate keys
data/languages.json       Supported code-to-name map
prepare_multidomain_data.py  Multidomain corpus build (v22 training data)
tools/                    FLORES evaluation, acceptance fit/report, .npz export, compression
model.npz                 Bundled release model (v22, int8)
docs/                     User guides, tutorial, provenance, evaluation
assets/logo.png           Project logo
```

Inference cleans text, hashes features, averages embeddings, runs the MLP in numpy, and optionally applies the stored acceptance policy. Release files are produced from a trained `.pt` checkpoint with `tools/fit_acceptance.py`, then `tools/export_numpy.py`. Training uses disk-backed feature caches and selects checkpoints by validation macro F1. Test data must never determine hyperparameters or checkpoint selection.

## Key files

- `requirements.txt`: inference dependencies (numpy, ftfy; no PyTorch).
- `requirements-training.txt`: PyTorch and additional data/training dependencies.
- `setup.sh`: local virtual environment bootstrap.
- `MODEL_LICENSE.md` and `THIRD_PARTY_NOTICES.md`: weight terms and source attribution.
- `docs/model-card.md`: intended use and limitations.

## Configuration

No application environment file, network service, or API key is required. `setup.sh` accepts `PYTHON` (interpreter path) and `VENV_DIR` (virtual environment directory), defaulting to `python3` and `.venv`. Training settings use CLI flags; see `--help` and `docs/training.md`.

## Working rules

Preserve runtime and checkpoint compatibility. Keep inference offline and free of PyTorch (`tests/test_release.py` and `tests/test_numpy_runtime.py` load models with `torch` blocked). Do not publish downloaded corpora, caches, or generated training reports; `.gitignore` excludes them. Do not invent benchmark claims. Software and authored examples/logo use MIT; weights and upstream data have separate terms. See [CONTRIBUTING.md](CONTRIBUTING.md).
