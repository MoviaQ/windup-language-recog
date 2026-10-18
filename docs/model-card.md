# Model card

## Identity and intended use

**Wind-Up Language Recognition 0.1.0** is a local, CPU-oriented text language classifier. It outputs one of 100 supported codes and is intended for short messages, exploratory language routing, and learning PyTorch. It does not translate, identify speakers, detect language spans, or establish correctness of downstream decisions.

The supported list in [data/languages.json](../data/languages.json) is an explicit coverage choice. It does not claim to be a population-based top 100. Chinese locales share one `zh` label; Japanese is `ja`.

## Architecture

A PyTorch mean `EmbeddingBag` with 131,072 buckets and 64 dimensions feeds a 100-output linear classifier. Features are stable hashes of character n-grams of lengths 1–5 and whole words (weight four). Cleanup uses ftfy and local regular expressions. Inference inspects at most 2,000 cleaned characters.

## Training provenance

The bundled **v4** checkpoint adapts the original public-data model trained from checksum-verified WiLI-2018 and MASSIVE 1.1 archives. The base received support-domain fine-tuning with public replay and authored synthetic support requests. Assistant-reviewed labels are not independently verified human gold labels. Held-out synthetic diagnostics are separate from training augmentation.

Adaptation uses seed 3187 and Adam with learning rate 0.0003. Epoch 8 of 12 was selected using ticket-validation accuracy plus 0.2 times public-validation macro recall, with a public-validation accuracy preservation constraint. Temperature 1.349526 was fitted on public validation; the confidence threshold 0.929371 was selected on 58 ticket-validation examples, accepting 43 with no observed errors. Test labels did not fit the threshold or choose the epoch.

The complete adaptation records and scripts are not distributed. The public training commands reproduce the base-model workflow, not the bundled v4 weights. See [training instructions](training.md), [benchmark notes](benchmark.md), and [third-party notices](../THIRD_PARTY_NOTICES.md).

## Evaluation interpretation

Report weighted accuracy alongside macro recall and macro F1. Public text and assistant utterances differ from other message domains. Synthetic examples demonstrate behavior but are a small authored sample and cannot certify general performance. Measurements are specific to the tested checkpoint, corpus, hardware, and settings; no universal latency or 99% accuracy claim is made.

Exact normalized split separation is checked under the project's key function. Semantic paraphrases and underlying-source overlaps can still exist. A fair comparison with another detector must state its supported languages, use the same input preprocessing, and report any excluded languages or examples.

## Limitations

Very short text, names, abbreviations, technical identifiers, closely related languages, transliteration, and mixed-language messages can be ambiguous. The detector always chooses a supported language in its standard API, including for unsupported languages. Cleanup can remove useful evidence. One output code cannot describe every language in a multilingual input.

Confidence is a model score. The v4 release applies public-validation temperature scaling with a ticket-validation cutoff of 0.929371; this does not claim 99% correctness. Optional calibrated thresholds can abstain with `und`, but neither calibration nor abstention guarantees correctness or identifies all unsupported input. Assess accepted accuracy and coverage together on new labeled data before adopting a threshold.

## Privacy and licensing

Normal inference is local and makes no network calls. Cleanup is not a personal-data anonymizer. Software and authored examples/logo use MIT; model contributions and upstream data terms are documented separately in [MODEL_LICENSE.md](../MODEL_LICENSE.md).
