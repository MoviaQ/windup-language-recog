# Model card — bundled v22

## Intended use

A local CPU classifier for application messages and general text, with exploratory language routing. It covers 100 language codes, including `ja` for Japanese, and predicts one language per input (no spans, no translation).

- `predict` and `predict_many` always choose a supported code.
- `predict_details` and `predict_ticket(..., allow_uncertain=True)` can abstain with `und`.
- The optional `predict_guarded_ticket(title, description)` adds one sentence-level veto for support tickets and can also abstain; see [ticket guard](ticket-guard.md).

## Architecture and inference

The network is one mean EmbeddingBag: 131,072 buckets, 64 dimensions, character n-grams 1–5, and whole-word hashes repeated four times. It is followed by a 128-unit ReLU hidden layer and 100 logits. Cleanup uses ftfy and local regexes and keeps at most 2,000 cleaned characters.

- **Storage.** `model.npz` is 8.7 MB. The embedding table is row-wise int8 (one float32 scale per row); the small layers are float32. The int8 table stays int8 in memory, and only the rows a text uses are converted.
- **Runtime.** Inference uses numpy only (`numpy_runtime.py`); PyTorch is not needed.
- **Measured in one run** (single process, single thread, 2,000 FLORES texts):
  - process RSS after loading: 56 MB;
  - import: 0.06 s; model load: 0.07 s;
  - median latency: 0.43 ms per text.

  Latency depends on hardware and text length, and most of it is feature hashing in Python.

**Acceptance.** A text is accepted when its temperature-scaled softmax top-1 probability reaches the cutoff for the predicted language and the text's length group. Length groups come from `evidence_profile`: more than three words, or at least eight CJK/kana/Hangul characters, counts as long. Otherwise the per-language cutoff applies, then the global one.

- There is no ensemble agreement gate, margin floor or single-word table. Cross-fitting on fitting data showed no gain from them for this network (see Exp E in [experiments](experiments.md)).
- A high score is not a guarantee of correctness, and abstention is not an out-of-distribution detector.

## Provenance and model selection

**Training data.** 14,922,860 rows, at most 150,000 per language, from the multidomain corpus (`prepare_multidomain_data.py --cap 150000`; sources in [third-party notices](../THIRD_PARTY_NOTICES.md)).

- Domain mix: web 40%, Wikipedia 30%, short sentences 15%, other 15%. 30% of rows are short crops.
- Three labels have fewer rows: bo 107,756, gn 123,237, tn 141,867.
- Removed from the corpus:
  - every FLORES-200 dev and devtest sentence (exact match or at least 50% shingle overlap);
  - the public WiLI/MASSIVE evaluation splits;
  - the project's authored test sets.

**Training and selection.** Adam, learning rate 0.003, batch 512, dropout 0.05. The epoch (2 of 4) was selected by macro F1 on FLORES dev. No test data selected anything.

**Acceptance policy.** Fitted by `tools/fit_acceptance.py` on the int8 network, using:

- FLORES dev;
- in-corpus validation (`prepared_v3/valid.csv`);
- public validation (`valid_short.csv`);
- one half of the Nynorsk negatives (`nn` is not a label).

Thresholds maximise correct − 25 × wrong among accepted texts. Fitting texts that also occur in reporting data were dropped (2,427 rows, recorded). Owner decision: Bokmål answers are kept, so Nynorsk is not penalised during fitting (negative scale 0).

Provenance (data hashes, counts, the cross-fit table, the tool's git commit) is stored in the release file's `source_metadata.acceptance_fit`.

## Evaluation (reporting-only data)

**FLORES-200 devtest** (100 codes; prefixes are word-truncated at 20/50/100 characters):

| Slice | Raw accuracy, v22 | Raw accuracy, v21 | Coverage, v22 | Accepted precision, v22 | Accepted errors, v22 (v21) |
|---|---|---|---|---|---|
| full | 99.27% | 98.51% | 96.57% | 99.99% | 5 (38) |
| ≤20 chars | 91.24% | 84.08% | 77.01% | 99.50% | 388 (2,252) |
| 21–50 | 98.32% | 96.22% | 94.02% | 99.94% | 56 (398) |
| >100 | 99.37% | 98.69% | 96.81% | 100.00% | 3 (18) |

Against NLLB lid218e restricted to these 100 codes, raw accuracy is +0.14 pp on full sentences (95% CI +0.09..+0.20) and +1.86 pp at ≤20 characters (+1.67..+2.04), both from a paired bootstrap and McNemar's test.

**Public test** (189,599 WiLI/MASSIVE texts):

| | Coverage | Accepted precision | Accepted errors |
|---|---|---|---|
| v22 | 93.10% | 99.77% | 409 |
| v21 | 95.39% | 99.68% | 586 |

Texts of three words or fewer are answered less often: 74.4% coverage against 88.6% for v21.

**Authored tickets** (84, AI-authored and not human-verified):

| API | v22 correct / wrong / withheld | v21 |
|---|---|---|
| whole-ticket policy | 68 / 6 / 10 | 70 / 5 / 9 |
| guarded | 68 / 1 / 15 | 70 / 0 / 14 |

v22 has no support-domain adaptation.

## Limitations and tradeoffs

- **Bosnian and Croatian.** On FLORES devtest full sentences, `bs` and `hr` are almost always withheld (accepted coverage 1.5% and 1.1%). Their raw F1 is 0.67 and 0.74. Serbian is accepted 99.5% of the time.
- **English.** English is withheld more often than most languages (64.7% coverage on FLORES full sentences, v21 45.6%). The model's English confidence is lower because other labels' training data contains English text.
- **Malay and Indonesian.** Accepted coverage is 68.0% for `ms` and 59.5% for `id`.
- **Nynorsk.** Nynorsk is not a label and is mostly answered as `nb` (70.9% of FLORES devtest `nno_Latn` accepted, almost all as `nb`).
- **Hard input in general.** Very short text, names, identifiers, code, transliteration, unsupported languages and mixed languages remain difficult. Evaluate accepted accuracy together with coverage on your own data; public benchmarks do not establish accuracy for your application.

## Privacy and licenses

Inference is offline, and cleanup is not a personal-data anonymizer. Software and authored examples use MIT. Weight contributions use CC BY-SA 4.0, subject to the upstream rights and attribution in [MODEL_LICENSE.md](../MODEL_LICENSE.md) and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Some training rows come from the MT560 compilation, which includes JW300. Its terms are unclear, and the project owner accepted that risk for this release. Source corpora are not distributed.
