<p align="center"><img src="assets/logo.png" alt="Wind-Up: a winding key and language-inspired mark" width="240"></p>

<h1 align="center">Wind-Up Language Recognition</h1>
<p align="center">A small numpy model. One text in, one language code out.</p>
<p align="center">
  <img alt="Software license: MIT" src="https://img.shields.io/badge/software-MIT-blue">
  <img alt="100 languages" src="https://img.shields.io/badge/languages-100-green">
  <img alt="Offline inference" src="https://img.shields.io/badge/inference-offline-orange">
  <img alt="Python 3.10 or newer" src="https://img.shields.io/badge/Python-3.10%2B-blue">
</p>

Wind-Up recognizes the language of short messages and longer text using hashed character fragments, whole words, and a compact neural network. Load the included weights once and run inference locally on your CPU. No API key or remote inference service is required.

The bundled v22 model (`model.npz`, 8.7 MB) is one network trained on a 15-million-row multidomain corpus, with a validation-fitted acceptance policy. Inference needs only numpy and ftfy. See the [model card](docs/model-card.md) for provenance, results and limits.

On FLORES-200 devtest (100 codes), raw accuracy is 99.27% on full sentences and 91.24% on texts of 20 characters or fewer. Against Meta's NLLB lid218e, restricted to the same codes, the paired differences are +0.14 pp and +1.86 pp, both significant. These are public-benchmark results, not a guarantee for your data.

Training uses PyTorch, so the code also works as a practical introduction to it.

## Quick start

```bash
git clone https://github.com/MoviaQ/windup-language-recog.git
cd windup-language-recog
./setup.sh
.venv/bin/python language_detector.py predict "Nie mogę zalogować się do konta."
```

`setup.sh` creates a virtual environment and installs numpy and ftfy (about 90 MB). Python 3.10+ and an internet connection are needed for installation; inference then runs offline. The bundled `model.npz` needs no training before use. PyTorch is only needed to train (`requirements-training.txt`) or to read old `.pt` checkpoints.

## Python API

Run from the repository directory, or add it to your Python import path:

```python
from language_detector import LanguageDetector

detector = LanguageDetector("model.npz")
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

`predict` and `predict_many` always choose one supported language. `predict_ticket` cleans a title and description, preferring substantive description text. `predict_details` adds a model confidence score and calibration status. `predict_guarded_ticket(title, description)` is an opt-in variant for support tickets: it can return `und` when one sentence confidently disagrees with the whole-ticket vote. See [ticket guard](docs/ticket-guard.md) for behavior, limits, and the small directional evaluation. The v22 policy applies a fitted temperature and confidence cutoffs per predicted language and length group. An explicitly calibrated model can return `und` when its stored acceptance policy rejects an input; `predict_ticket(..., allow_uncertain=True)` requires such a checkpoint. Confidence is not a guarantee of correctness, and uncertainty mode is not an out-of-distribution detector.

## How it works

```text
Text → local cleanup → hashed character n-grams + words
     → mean embedding → MLP → logits → language code
     → optional language/length confidence cutoffs → language code or und
```

The release uses 131,072 hash buckets, 64-dimensional embeddings stored as int8, character n-grams of lengths 1–5, and weighted whole-word features. No vocabulary download or transformer runtime is needed. In one measured run, a process using the model needed about 56 MB of RAM, and median latency was about 0.4 ms per text on one CPU thread. See [architecture](docs/architecture.md) for the design and [the PyTorch tutorial](docs/pytorch-tutorial.md) for tensors, gradients, and a small training exercise.

## Quality and limitations

Short text can be ambiguous: shared words, names, code snippets, mixed languages, and closely related languages may not contain enough evidence for a reliable label. The model predicts one language per input and does not identify language spans. Performance on public corpora does not establish accuracy for your application.

Known weak spots: Bosnian and Croatian are usually withheld (`und`), English and Malay/Indonesian are withheld more often than other languages, and Norwegian Nynorsk (not a label) is answered as Bokmål (`nb`). The [model card](docs/model-card.md) explains provenance and evaluation limits, and [experiments](docs/experiments.md) records how the model was built and compared; latency depends on hardware and text length.

## Documentation

- [Getting started and API behavior](docs/getting-started.md)
- [Ticket guard (optional support-ticket veto)](docs/ticket-guard.md)
- [Architecture and preprocessing](docs/architecture.md)
- [Learn PyTorch with this model](docs/pytorch-tutorial.md)
- [Reproduce public-data training](docs/training.md)
- [Model card](docs/model-card.md)
- [Contributing](CONTRIBUTING.md) and [security reporting](SECURITY.md)

## Using with Codex

Open the repository in Codex. [AGENTS.md](AGENTS.md) provides verified commands, file ownership context, and architecture notes so changes start from the actual implementation.

## Licenses

Software, project-authored demo and synthetic examples, and the logo are licensed under [MIT](LICENSE). The released model weight contributions are offered under [CC BY-SA 4.0](MODEL_LICENSE.md), with upstream rights and attribution described in [third-party notices](THIRD_PARTY_NOTICES.md). Source datasets are not distributed or relicensed by this repository.
