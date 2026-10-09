# hbs cluster: evidence for merging bs/hr/sr (98-class option)

Bosnian, Croatian and Serbian are mutually intelligible varieties sharing
ijekavian forms. A hashed character-bag cannot separate identical strings
carrying two labels: the train split itself contains 67 shared normalized
keys labeled both ways, and the confusion is a zero-sum game between
siblings (a full retrain with extra bs/hr/sr data moved bs 0.664 -> 0.761
while hr fell 0.705 -> 0.600, overall unchanged at 0.9833).

Measured on the public held-out `test_short.csv` (189,557 texts, real
labels), mapping both sides' bs/hr/sr to `hbs` for a fair comparison:

| Model | Overall | hbs cluster (n=1486) |
| --- | ---: | ---: |
| v20-style 100-class baseline | 0.9828 | 0.9684 |
| `--merge-hbs` retrain (98 classes) | 0.9850 | 0.9731 |

With ticket-domain adaptation on top (same recipe as v4, guardrail held):
ticket-validation 0.931 -> 0.966 (56/58), fresh 120-ticket test 119/120
plain and 102/120 accepted at 100% precision in cautious mode.

This branch adds only the mechanism (`--merge-hbs` in `train_model.py`,
`merge_code`/`merge_languages` in `language_detector.py`); default
training is unchanged (100 classes). Publishing 98-class weights would be
a release decision: it changes the `languages.json` contract and the
`model.pt` artifact.
