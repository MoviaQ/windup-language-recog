# Changelog

## v22 — 2026-11-13

Weights, runtime and acceptance policy all change. Numbers are from reporting-only data (FLORES-200 devtest, public test split, authored tickets); nothing was selected or fitted on them. Details: [experiments](docs/experiments.md) (Exp D, Exp E and the v22 release section).

- **Runtime without PyTorch.** Inference runs on numpy (`numpy_runtime.py`). `requirements.txt` is now `numpy` + `ftfy`, and PyTorch moved to `requirements-training.txt`.
  - In one measured run: a numpy-only environment is 88 MB instead of 5.6 GB with PyTorch; process RSS after loading the model is 56 MB instead of 560 MB; import takes 0.06 s instead of 1.05 s. Median single-text latency is unchanged at about 0.4 ms.
  - Legacy `.pt` checkpoints still load (PyTorch then needed only to read the file) and run through the same numpy code.
  - On every FLORES dev and devtest item, numpy and the previous PyTorch code gave the same raw label and the same `und` decision, for the v21 ensemble and for Exp D in fp32 and int8. That is 749,222 items per model; the largest probability difference was 2.3e-6.
- **New bundled model `model.npz`, 8.7 MB, replacing the 67.9 MB `model.pt`.** It is a single MLP: 131,072 buckets × 64, hidden 128, n-grams 1–5 plus words, `ticket_v2`.
  - It is trained on the 150k-rows-per-language multidomain corpus (`prepare_multidomain_data.py --cap 150000`) and stored as row-wise int8.
  - FLORES-200 devtest raw accuracy: 99.27% on full sentences and 91.24% at ≤20 characters, against 98.51% and 84.08% for v21. Paired against NLLB lid218e (restricted to these 100 codes), the released int8 model is +0.14 pp on full sentences (95% CI +0.09..+0.20) and +1.86 pp at ≤20 characters (+1.67..+2.04), both significant.
  - The v21 weights remain available in git history.
- **New acceptance policy**, fitted by the committed `tools/fit_acceptance.py` on FLORES dev, in-corpus validation and public validation only.
  - It uses a temperature plus confidence cutoffs per language and per (language, length). Margin floors and single-word tables were dropped because cross-fitting showed no gain. There is no ensemble agreement gate (single network).
  - Owner decision: Bokmål is answered. As a result, Nynorsk, which is not a label, is mostly answered as `nb` (70.9% of FLORES devtest `nno_Latn` accepted; v21: 66.3%).
  - Acceptance on FLORES devtest (full sentences / ≤20 characters):
    - coverage 96.57% / 77.01% (v21: 93.63% / 71.09%);
    - accepted precision 99.99% / 99.50% (v21: 99.96% / 96.87%);
    - accepted errors 5 / 388 (v21: 38 / 2,252).
  - Public test (189,599 texts): coverage 93.10% (v21: 95.39%), accepted precision 99.77% (v21: 99.68%), accepted errors 409 (v21: 586). Short texts of 3 words or fewer are answered less often: coverage 74.4% (v21: 88.6%).
  - Authored tickets (84): whole-ticket policy 68 correct / 6 wrong / 10 withheld (v21: 70 / 5 / 9). Guarded API: 68 / 1 / 15 (v21: 70 / 0 / 14). v22 has no support-domain adaptation.
  - Remaining weak spots: `bs`/`hr` are almost always withheld (accepted coverage on FLORES full sentences 1.5% / 1.1%), and `en` coverage is 64.7%.
- **Licensing.** The weights are trained on the multidomain corpus, including OpenLID-v3 rows tagged `mt560` (the MT560 compilation, which contains JW300). The owner accepted that risk for this release; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **Tooling:**
  - `tools/export_numpy.py` writes `.npz` release files.
  - `tools/fit_acceptance.py` and `tools/evaluate_acceptance.py` fit and report acceptance policies.
  - `tools/benchmark_ticket_guard.py` drops `--threads` (it only set PyTorch threads).
  - `prepare_multidomain_data.py` records the OpenLID licence tag in each row's source, so later manifests count `mt560` rows.

## Unreleased (software only)

- Add opt-in `predict_guarded_ticket(title, description, cutoff=0.65)`: a frozen sentence-level veto for support tickets that can return `und` on confident foreign-sentence evidence. Default APIs, CLI, weights, and checkpoints are unchanged; no new training.
- Add runtime `ticket_guard.py` (shared input selection, 2,000-character prefix, >=3-word sentence filter) plus mocked and real-model regression tests.
- Add the 84-row AI-authored, human-unverified fixture (`tests/fixtures/authored_tickets.jsonl`, MIT) with an offline non-fitting benchmark (`tools/benchmark_ticket_guard.py`); confirmed guard 70 C / 0 E vs fastText-0.65 68 C / 0 E, exact-matched tie 68/68 on the conservative 82 rows. See `docs/ticket-guard.md`. Benchmark-only `requirements-benchmark.txt` pins the fastText wheel; runtime stays torch + ftfy.

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
