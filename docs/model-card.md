# Model card — bundled v21

## Intended use

A local CPU classifier for short application messages and exploratory language routing. It retains all 100 supported codes, including `ja` for Japanese. It predicts one language, not spans or translations. Standard `predict` and `predict_many` always choose a supported code. `predict_details` and `predict_ticket(..., allow_uncertain=True)` can abstain with `und`.

## Architecture and inference

Two mean EmbeddingBag networks each use 131,072 buckets and 64-dimensional vectors, character n-grams 1–5, and whole-word hashes repeated four times. One network has a 128-unit ReLU hidden layer; the other is linear. Each produces 100 logits. Temperature-normalized logits are combined with weights 0.75 and 0.25, then a final validation-fitted temperature produces softmax scores. Cleanup uses ftfy and local regexes, with at most 2,000 cleaned characters for the network.

Acceptance requires agreement between networks, stored cutoffs by predicted language and text length, a per-length top-2 margin floor, and case-sensitive single-word evidence. The margin floor requires (top1 probability - top2 probability) to be at or above a stored value for the input's length group. The lexical rule can reject a standalone ambiguous term or a term whose distinctive lexical language contradicts the prediction. A high softmax score can therefore still result in `und`. These decisions do not inflate raw scores or guarantee correctness.

The checkpoint is about 67.9 MB and performs two network passes. Older linear and MLP checkpoints and their global confidence thresholds remain supported. No v21 latency benchmark is claimed.

## Provenance and model selection

The ensemble derives from the WiLI-2018 and MASSIVE 1.1 public-data base, with subsequent local support-domain adaptation, public-data replay, and project-authored multilingual augmentation. Validation selected the adapted MLP checkpoint and ensemble weight, which v21 retains. v21 refit the language/length confidence cutoffs on the 100,175-row public validation split (disjoint from train and test), capped them at v20's value, and added the per-length top-2 margin floors. No diagnostic test labels were used to fit thresholds or margins. Those are sample observations, not error guarantees. Authored template families and their variants are correlated.

Single-word evidence was estimated from language-balanced training-document counts. The released vocabulary includes only terms also present in public training text. Runtime-required weights, configuration, calibration, and vocabulary are included; working provenance metadata and adaptation records are not distributed. Public training commands reproduce the base workflow, not this complete adapted ensemble.

Repeated historical diagnostics and analysis of their errors informed further development ideas. Their rows and labels did not train weights, populate the vocabulary, or fit thresholds, but the overall development/stopping process was adaptive. A 64-message sample was labeled and locked before final candidate predictions and was not used to select or retune it. Its labels were assigned by an assistant and have not been independently verified by humans. See [evaluation details](benchmark.md).

## Limitations and tradeoffs

Very short text, names, identifiers, code, transliteration, unsupported languages, and closely related or mixed languages remain difficult. Cleanup can remove useful evidence. Abstention does not reliably identify every unsupported input. Always evaluate accepted accuracy together with coverage.

The v21 diagnostics show more accepted correct application responses than v7 on the historical application sets, with the same or fewer accepted errors there; on the authored diagnostics v21 gains 4 correct accepted responses while accepted errors rise from v7's 3 to 8. On 189,557 public test texts, v21 accepts more correct responses and fewer errors than v20 (595 versus 637; v7 had 515), improving selective precision and coverage over v20. A per-length margin floor cost one accepted correct response on the locked 64-message sample, with no accepted errors. This is not universal superiority or a production accuracy guarantee. Test reuse, correlated authored examples, and unverified labels limit the conclusions.

## Privacy and licenses

Inference is offline; cleanup is not a personal-data anonymizer. Software and authored examples use MIT. Weight contributions use CC BY-SA 4.0, subject to upstream rights and attribution in [MODEL_LICENSE.md](../MODEL_LICENSE.md) and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Source corpora are not distributed.
