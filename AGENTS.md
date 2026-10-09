# Wind-Up Language Recognition

**Version:** 0.1.0 | **Port:** none (local CLI/library) | **Stack:** Python 3.10+, PyTorch, ftfy

## What

A CPU language classifier for 100 language codes. Hashes character n-grams and whole words into an EmbeddingBag, then combines MLP and linear-network logits in the v20 release.

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
.venv/bin/python -m compileall -q language_detector.py ticket_text.py train_model.py prepare_public_data.py dataset_utils.py evaluate_public.py
# Public corpus and training; downloads data, then performs a substantial CPU job
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python prepare_public_data.py
.venv/bin/python train_model.py --short-only --epochs 12 --patience 4 --batch-size 512 --buckets 131072 --embedding-dim 64 --ngram-max 5 --word-features --preprocessing ticket_v2 --dropout 0.05 --lr 0.003 --model model-retrained.pt
.venv/bin/python evaluate_public.py --model model-retrained.pt
# Small educational exercise; do not overwrite release weights
.venv/bin/python language_detector.py train --data data/demo.csv --epochs 120 --model model-demo.pt
```

## Architecture

```text
language_detector.py      Encoding, model, inference API, demo trainer, CLI
ticket_text.py            Local text cleanup and description extraction
train_model.py            Cached features, public validation, checkpoint selection
prepare_public_data.py    Public downloads, split handling, short-text preparation
evaluate_public.py        Validation temperature fit, public evaluation, latency
dataset_utils.py          CSV helpers and normalized duplicate keys
data/languages.json       Supported code-to-name map
model.pt                  Bundled release checkpoint
docs/                     User guides, tutorial, provenance, evaluation
assets/logo.png           Project logo
```

Inference cleans text, hashes features, averages embeddings in two networks, combines logits, and optionally applies the stored acceptance policy. Training uses disk-backed feature caches and selects checkpoints by validation macro F1. Test data must never determine hyperparameters or checkpoint selection.

## Key files

- `requirements.txt`: inference dependencies.
- `requirements-training.txt`: additional data/training dependencies.
- `setup.sh`: local virtual environment bootstrap.
- `MODEL_LICENSE.md` and `THIRD_PARTY_NOTICES.md`: weight terms and source attribution.
- `docs/model-card.md`: intended use and limitations.

## Configuration

No application environment file, network service, or API key is required. `setup.sh` accepts `PYTHON` (interpreter path) and `VENV_DIR` (virtual environment directory), defaulting to `python3` and `.venv`. Training settings use CLI flags; see `--help` and `docs/training.md`.

## Working rules

Preserve runtime and checkpoint compatibility. Keep inference offline. Do not publish downloaded corpora, caches, or generated training reports; `.gitignore` excludes them. Do not invent benchmark claims. Software and authored examples/logo use MIT; weights and upstream data have separate terms. See [CONTRIBUTING.md](CONTRIBUTING.md).
