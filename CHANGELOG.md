# Changelog

## v20 checkpoint — 2026-10-09

- Replace bundled weights with a calibrated MLP/linear ensemble for all 100 languages.
- Add model-consensus, language/length, and case-sensitive single-word acceptance rules, while preserving raw prediction APIs and older checkpoints.
- Add CLI `--allow-uncertain` and portable policy regression tests.
- Document higher acceptance on application diagnostics, the general-text error tradeoff, and the two-network inference cost.

## 0.1.0 — 2026-10-07

Initial public release.

- Local CPU inference for 100 language codes with bundled PyTorch weights.
- Single-input, batch, request-field, and confidence-detail APIs.
- Hashed character n-grams and whole-word features with an EmbeddingBag classifier.
- Reproducible public WiLI-2018 and MASSIVE 1.1 preparation/training scripts.
- Setup script, tests, practical PyTorch tutorial, provenance and licensing notices.
