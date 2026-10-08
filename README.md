<p align="center"><img src="assets/logo.png" alt="Wind-Up: a winding key and language-inspired mark" width="240"></p>

<h1 align="center">Wind-Up Language Recognition</h1>
<p align="center">A small PyTorch model. One text in, one language code out.</p>
<p align="center">
  <img alt="Software license: MIT" src="https://img.shields.io/badge/software-MIT-blue">
  <img alt="100 languages" src="https://img.shields.io/badge/languages-100-green">
  <img alt="Offline inference" src="https://img.shields.io/badge/inference-offline-orange">
  <img alt="Python 3.10 or newer" src="https://img.shields.io/badge/Python-3.10%2B-blue">
</p>

Wind-Up recognizes the language of short messages and longer text using hashed character fragments, whole words, and a compact neural network. Load the included weights once and run inference locally on your CPU. No API key or remote inference service is required.

The implementation is deliberately approachable: useful as a language detector and as a practical introduction to PyTorch.

## Intended use and fastText comparison

Wind-Up is designed for local language detection in short application messages: support requests, contact forms, and helpdesk titles or descriptions. A typical integration uses the predicted language to suggest a support queue or response language. Its ticket API extracts substantive text from a title and description, and its character and word features can capture language cues in messages containing technical terms or occasional foreign words. Very short or heavily mixed messages still need an uncertainty policy and evaluation on your own examples.

| Consideration | Wind-Up | fastText language identification |
| --- | --- | --- |
| Language coverage | 100 supported codes | 176 supported codes |
| Main reason to choose it | Readable PyTorch implementation, configurable preprocessing and training, and a title/description API | Broad language coverage and fast general-purpose inference |
| Local inference | CPU, offline after installation; PyTorch runtime | CPU, offline after installation; fastText runtime |
| Measured latency | Optional native encoder: 0.106 ms median on 1,000 varied public texts | `.bin`: 0.043 ms; `.ftz`: 0.037 ms on the same sample |
| Uncertainty handling | Calibrated checkpoints support returning `und` below a stored confidence threshold | Prediction scores can be used with an application-defined rejection policy |

Choose Wind-Up when you want to understand or adapt the detector in PyTorch and integrate message cleanup into your application. fastText remains a strong baseline when speed and wider language coverage are priorities. See the [official fastText models](https://fasttext.cc/docs/en/language-identification.html) and our [latency protocol and environment](docs/native-performance.md); these timings exclude model loading and structured-description extraction.

The bundled checkpoint is trained on public WiLI-2018 and MASSIVE 1.1 data. The support-message workflow is an intended use, not a claim that these released weights outperform fastText on real support traffic. The release's 98.29% public-test accuracy is not a head-to-head fastText result or a guarantee for one-sentence tickets. Compare both models on the same independently labeled application test before choosing an automatic routing policy. Wind-Up returns one language per input, not language spans or a translation.

## Quick start

```bash
git clone https://github.com/MoviaQ/windup-language-recog.git
cd windup-language-recog
./setup.sh
.venv/bin/python language_detector.py predict "Nie mogę zalogować się do konta."
```

`setup.sh` creates a virtual environment and installs PyTorch and ftfy. Python 3.10+ and an internet connection are needed for installation; inference then runs offline. The bundled `model.pt` needs no training before use.

For an optional faster feature encoder, see [native build instructions](native/README.md). The network remains PyTorch and the model weights are unchanged.

## Python API

Run from the repository directory, or add it to your Python import path:

```python
from language_detector import LanguageDetector

detector = LanguageDetector("model.pt")
print(detector.predict("Nie mogę zalogować się do konta."))
print(detector.predict_many([
    "I cannot sign in to my account.",
    "Ich kann mich nicht anmelden.",
]))
print(detector.predict_ticket(
    title="Login problem",
    description="Nie mogę się zalogować po zmianie hasła.",
))
print(detector.predict_details("Non riesco ad accedere al mio account."))
```

The model returns short language codes, such as `pl`, `en`, `de`, `it`, and `ja` for Japanese. All 100 supported codes are listed in [data/languages.json](data/languages.json); the list is a coverage choice, not a ranking of the world's most popular languages.

`predict` and `predict_many` always choose one supported language. `predict_ticket` cleans a title and description, preferring substantive description text. `predict_details` adds a model confidence score and calibration status. The release uses validation temperature scaling and a fixed 0.9 confidence cutoff. An explicitly calibrated checkpoint can return `und` when confidence falls below its stored threshold; `predict_ticket(..., allow_uncertain=True)` requires such a checkpoint. Confidence is not a guarantee of correctness, and uncertainty mode is not an out-of-distribution detector.

## How it works

```text
Text → local cleanup → hashed character n-grams + words
     → mean embedding → linear classifier → language code
```

The release uses 131,072 hash buckets, 64-dimensional embeddings, character n-grams of lengths 1–5, and weighted whole-word features. There is no vocabulary download or transformer runtime. See [architecture](docs/architecture.md) for the design and [the PyTorch tutorial](docs/pytorch-tutorial.md) for tensors, gradients, and a small training exercise.

## Quality and limitations

Short text can be ambiguous: shared words, names, code snippets, mixed languages, and closely related languages may not contain enough evidence for a reliable label. The model predicts one language per input and does not identify language spans. Performance on public corpora does not establish accuracy for your application.

The release is trained from public WiLI-2018 and MASSIVE 1.1 data. Its [model card](docs/model-card.md) explains provenance and evaluation limits. See [benchmark notes](docs/benchmark.md) for measured results and methodology; latency depends on hardware, text length, and thread settings.

## Documentation

- [Getting started and API behavior](docs/getting-started.md)
- [Architecture and preprocessing](docs/architecture.md)
- [Learn PyTorch with this model](docs/pytorch-tutorial.md)
- [Reproduce public-data training](docs/training.md)
- [Model card](docs/model-card.md)
- [Contributing](CONTRIBUTING.md) and [security reporting](SECURITY.md)

## Using with Codex

Open the repository in Codex. [AGENTS.md](AGENTS.md) provides verified commands, file ownership context, and architecture notes so changes start from the actual implementation.

## Licenses

Software, project-authored demo and synthetic examples, and the logo are licensed under [MIT](LICENSE). The released model weight contributions are offered under [CC BY-SA 4.0](MODEL_LICENSE.md), with upstream rights and attribution described in [third-party notices](THIRD_PARTY_NOTICES.md). Source datasets are not distributed or relicensed by this repository.
