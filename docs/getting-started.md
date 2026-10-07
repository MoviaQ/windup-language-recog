# Getting started

## Install

Python 3.10+ is required. From the repository root:

```bash
./setup.sh
.venv/bin/python language_detector.py predict "Please help me reset my password."
```

The initial installation downloads dependencies; the model checkpoint is bundled. Thereafter normal inference needs no internet connection. `PYTHON=/path/to/python3 ./setup.sh` selects an interpreter, and `VENV_DIR=.venv-custom ./setup.sh` selects a virtual environment directory. Use that directory's Python for subsequent commands.

For Windows or manual installation:

```powershell
py -3 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python language_detector.py predict "Please help me reset my password."
```

## Reuse a detector

```python
from language_detector import LanguageDetector

model = LanguageDetector("model.pt")
code = model.predict("Potrzebuję pomocy ze zmianą hasła.")
codes = model.predict_many([
    "Please reset my password.",
    "Bitte setzen Sie mein Passwort zurück.",
], batch_size=128)
```

Loading weights for every request wastes time. Construct one detector and reuse it. Check `model.languages` for supported codes. Japanese is `ja`; `jp` is a country code.

## Titles and descriptions

```python
code = model.predict_ticket(
    "Account access",
    "Nie mogę wejść na konto po zmianie hasła.",
)
```

The extractor removes recognizable system/mail headers, quoted replies, and some signatures. It prefers a description containing enough letters or words; otherwise it uses the title. Cleanup is heuristic, so inspect extraction behavior for your message formats. The classifier sees at most 2,000 characters after cleanup.

## Confidence and optional uncertainty

```python
result = model.predict_details("I need help opening this file.")
print(result["language"], result["confidence"], result["calibrated"])
code_or_und = model.predict_ticket("Access problem", "Please help me sign in.", allow_uncertain=True)
```

Results contain `language`, `best_language`, `confidence`, `uncertain`, and `calibrated`. Confidence is the largest softmax score, optionally temperature-scaled. With no checkpoint calibration, `calibrated` is false and the method always returns its best supported code.

The release checkpoint uses validation-fitted temperature scaling and a fixed 0.9 operational cutoff. This threshold is not a correctness guarantee.

A separately calibrated checkpoint can store `temperature` and `min_confidence`; then `predict_details` returns `und` below that threshold. `predict_ticket(..., allow_uncertain=True)` requires calibration and can return `und` for empty cleaned text. This reduces accepted predictions; it does not guarantee an accuracy level or reliably identify unsupported languages. Fit calibration on validation data, and evaluate accuracy together with acceptance coverage on held-out data.

## Errors and performance

Plain `predict` raises `ValueError` when cleaned input has no letters. Batch prediction aborts if an input is invalid; filter invalid inputs first when handling heterogeneous batches. An empty batch returns an empty list, and batch size must be positive.

For short CPU requests you can benchmark `torch.set_num_threads(1)` before constructing the detector. PyTorch's default thread count can add overhead. This setting affects the process globally; choose it for your workload rather than treating a measured latency as a universal guarantee.
