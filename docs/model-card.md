# Model card

## Identity and intended use

**Wind-Up Language Recognition 0.1.0** is a local, CPU-oriented text language classifier. It outputs one of 100 supported codes and is intended for short messages, exploratory language routing, and learning PyTorch. It does not translate, identify speakers, detect language spans, or establish correctness of downstream decisions.

The supported list in [data/languages.json](../data/languages.json) is an explicit coverage choice. It does not claim to be a population-based top 100. Chinese locales share one `zh` label; Japanese is `ja`.

## Architecture

A PyTorch mean `EmbeddingBag` with 131,072 buckets and 64 dimensions feeds a 100-output linear classifier. Features are stable hashes of character n-grams of lengths 1–5 and whole words (weight four). Cleanup uses ftfy and local regular expressions. Inference inspects at most 2,000 cleaned characters.

## Training provenance

The released checkpoint is separately initialized and trained exclusively from checksum-verified raw **WiLI-2018** and **MASSIVE 1.1** archives. WiLI supplies Wikipedia-derived passages; MASSIVE supplies multilingual assistant utterances. Long inputs are converted into short excerpts, and normalized duplicates/conflicting labels are excluded. Project-authored synthetic examples are reserved for illustrative evaluation and are excluded from training.

WiLI source training data is divided into train and validation by language. MASSIVE's source train/dev/test assignments are retained before exclusions. The release uses Adam, class-frequency weighting, 5% feature dropout, and a maximum of 12 epochs. Validation macro F1 chooses the checkpoint; test performance does not select epochs.

Preparation produces a reproducibility manifest containing source checksums, language codes, seeds, and counts. See [training instructions](training.md), [benchmark notes](benchmark.md), and [third-party notices](../THIRD_PARTY_NOTICES.md) for commands, measured results, and attribution.

## Evaluation interpretation

Report weighted accuracy alongside macro recall and macro F1. Public text and assistant utterances differ from other message domains. Synthetic examples demonstrate behavior but are a small authored sample and cannot certify general performance. Measurements are specific to the tested checkpoint, corpus, hardware, and settings; no universal latency or 99% accuracy claim is made.

Exact normalized split separation is checked under the project's key function. Semantic paraphrases and underlying-source overlaps can still exist. A fair comparison with another detector must state its supported languages, use the same input preprocessing, and report any excluded languages or examples.

## Limitations

Very short text, names, abbreviations, technical identifiers, closely related languages, transliteration, and mixed-language messages can be ambiguous. The detector always chooses a supported language in its standard API, including for unsupported languages. Cleanup can remove useful evidence. One output code cannot describe every language in a multilingual input.

Confidence is a model score. The release applies validation-fitted temperature scaling with a fixed 0.9 operational cutoff; this does not claim 99% correctness. Optional calibrated thresholds can abstain with `und`, but neither calibration nor abstention guarantees correctness or identifies all unsupported input. Assess accepted accuracy and coverage together on new labeled data before adopting a threshold.

## Privacy and licensing

Normal inference is local and makes no network calls. Cleanup is not a personal-data anonymizer. Software and authored examples/logo use MIT; model contributions and upstream data terms are documented separately in [MODEL_LICENSE.md](../MODEL_LICENSE.md).
