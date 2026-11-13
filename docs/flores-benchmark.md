# FLORES-200 evaluation

`tools/evaluate_flores.py` scores the bundled checkpoint on the public
FLORES-200 sentences for all 100 supported codes, and optionally scores
fastText `lid.176` and NLLB `lid218e` predictions on the same items. It is a
general-text diagnostic, not evidence about support tickets.

## Split policy

- `--split dev` (997 sentences per language) may be used for checkpoint
  selection, thresholds and other decisions.
- `--split devtest` (1012 sentences per language, the default) is for
  reporting only. Never select checkpoints, hyperparameters or thresholds on
  devtest numbers.

## Data

Download and extract FLORES-200 outside the repository (or under the
ignored `reports/`). It is licensed **CC BY-SA 4.0** and must not be
committed:

```bash
mkdir -p reports/flores
curl --fail --location --output reports/flores/flores200_dataset.tar.gz \
  https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
tar -xzf reports/flores/flores200_dataset.tar.gz -C reports/flores
```

## Items

Each sentence is evaluated in full and as prefixes of at most 20, 50 and 100
characters (word-prefix truncation; scripts without spaces fall back to a
character prefix). A prefix equal to a text already produced for that
sentence is skipped. Slices: `full` (full sentences) and `len<=20`,
`len21-50`, `len51-100`, `len>100`, bucketed by the actual length of all
items. With `--extra-variants`, `zho_Hant -> zh`, `prs_Arab -> fa` and
`ckb_Arab -> ku` are reported separately; they never enter the main slices.

The code-to-file mapping is the `FLORES_FOR_CODE` constant in the tool,
with comments on ambiguous choices (for example `fa=pes_Arab`,
`ku=kmr_Latn`, `zh=zho_Hans`, `sr=srp_Cyrl`). The tool refuses to run when
a supported code lacks a mapping or file, unless `--allow-missing` is
passed, which lists the skipped codes in the output.

## Metrics

Accuracy, macro precision/recall/F1 and macro false-positive rate over the
gold languages of each slice, per-language rows, and the top confusion
pairs. For our checkpoint, `abstention` gives coverage and accepted
precision of the stored acceptance policy. An input that raises in our
predictor is recorded with its error under `ours_errors`, counted wrong and
not accepted; the count is printed.

For external models, `restricted` takes the highest-ranked label that maps
to one of our 100 codes; `unrestricted` takes the top-1 label and counts an
unmapped label (for example `sh`, `nn`, `azb`) as wrong. The mapping of
fastText and NLLB labels lives in the same tool (`FASTTEXT_EXTRA_LABELS`,
`NLLB_LABELS`). `lid.176` has no label for `ha`, `ig`, `om`, `rw`, `sn` or
`tn`, so those languages score zero for it.

## Commands

```bash
# Our model (read-only; default model.npz). One process per worker.
.venv/bin/python tools/evaluate_flores.py --flores-dir reports/flores/flores200_dataset \
  --split dev --extra-variants --output reports/flores_dev.json

# Optional external models, in any environment with fastText bindings
# (requirements-benchmark.txt). The predictions carry per-item text digests;
# the evaluator refuses files for another split or with missing items.
python tools/predict_fasttext.py --model reports/lid.176.bin --name ft176 \
  --flores-dir reports/flores/flores200_dataset --split devtest --output reports/ft176_devtest.jsonl
python tools/predict_fasttext.py --model reports/lid218e.bin --name nllb218e \
  --flores-dir reports/flores/flores200_dataset --split devtest --output reports/nllb218e_devtest.jsonl
.venv/bin/python tools/evaluate_flores.py --flores-dir reports/flores/flores200_dataset \
  --split devtest --extra-variants \
  --fasttext-predictions reports/ft176_devtest.jsonl \
  --fasttext-predictions reports/nllb218e_devtest.jsonl
```

`--limit-per-language N` makes quick smoke runs; its numbers are not
comparable with full runs.

## Caveats

- NLLB `lid218e` comes from the same project as FLORES-200 and was tuned
  on FLORES dev, so its FLORES numbers (dev especially) are optimistic and
  not a fair held-out comparison.
- FLORES is translated Wikipedia-style prose. It says nothing about short
  support messages, code-mixed text or metadata noise.
- The `lid.176.bin` used below has SHA-256
  `7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e`, the
  same file as in [ticket-guard.md](ticket-guard.md). Check the hash of any
  copy obtained from a mirror before comparing numbers. `lid218e.bin` had
  SHA-256 `8ded5749a2ad79ae9ab7c9190c7c8b97ff20d54ad8b9527ffa50107238fc7f6a`.
  Both weights have their own licenses and are not bundled.

## Recorded devtest run (v21)

This run used the v21 release; v22 results are in the [model card](model-card.md) and [experiments](experiments.md) ("v22 release").

Commit `e4688fe` (working tree: only the then-uncommitted tool and its
test), `model.pt` SHA-256 `358586ac…579b`, split `devtest`, fasttext-wheel
0.9.2 on Python 3.12 for the external predictions, commands as above with
`--workers 4`. Cells are accuracy / macro F1 in percent; the last column is
our acceptance coverage / accepted precision.

| Slice | Texts | Ours raw | lid.176 restricted | lid.176 unrestricted | lid218e restricted | lid218e unrestricted | Ours accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 101200 | 98.51 / 98.52 | 88.19 / 86.01 | 87.93 / 86.01 | 99.13 / 99.10 | 98.97 / 99.03 | 93.63 / 99.96 |
| len<=20 | 101200 | 84.08 / 84.30 | 71.31 / 69.69 | 70.33 / 69.50 | 89.38 / 89.38 | 88.36 / 89.08 | 71.09 / 96.87 |
| len21-50 | 101135 | 96.22 / 96.24 | 83.34 / 81.28 | 82.80 / 81.20 | 98.03 / 98.00 | 97.69 / 97.85 | 88.17 / 99.55 |
| len51-100 | 98885 | 98.24 / 98.27 | 87.05 / 85.14 | 86.77 / 85.16 | 99.06 / 99.00 | 98.93 / 98.87 | 92.75 / 99.92 |

Three devtest inputs (20-character prefixes) raised
`ValueError: Text must contain at least one letter.` in our predictor and
are counted wrong. With `--extra-variants`, our checkpoint scores 0% raw
accuracy on `ckb_Arab` and 100% on `zho_Hant` and `prs_Arab`.
