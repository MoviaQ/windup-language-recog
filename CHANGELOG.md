# Changelog

## v21 checkpoint — 2026-10-10

- Acceptance-policy-only release: the raw neural weights, ensemble members, weights, and temperatures are unchanged from v20.
- Refit `min_confidence_by_language_and_length` on the public validation split (100,175 rows; disjoint from train and test), capped so v21 never raises a confidence threshold above v20's; 94 (language, length) cells were lowered.
- Add per-length top-2 softmax margin floors (`short` 0.88, `long` 0.66): an accepted prediction must also have (top1 probability - top2 probability) at or above its length group's floor.
- Public test: accepted precision improves (99.6708% vs 99.6456%) with fewer accepted errors (595 vs 637) and higher coverage (95.34% vs 94.82%); accepted correct responses rise to 180,138.
- Ticket/application diagnostics are not regressed; the authored and difficult-language sets accept more correct responses.
- One-sample coverage regression on the locked 64-message sample (55 vs 56 correct accepted, 0 accepted errors), caused by the margin floor; all three ambiguous inputs are still rejected.
- Add portable v21 policy regression tests.

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
