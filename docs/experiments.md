# Experiments

Experimental checkpoints are not the bundled `model.pt`. Each entry names the commit, data, command, selection rule and the single devtest evaluation made after selection.

## Exp A: v21 MLP member architecture on the multidomain corpus

- **Question:** how much do data alone change FLORES results, with the architecture unchanged?
- **Data:** `prepare_multidomain_data.py --cap 50000 --workers 4` with the filters of commit `269d6e5` gives 5,000,000 train rows, 50,000 for each of the 100 codes. FLORES dev and devtest are filtered out of the training data (see [training](training.md)).
- **Model:** single MLP. Configuration is the same as the v21 MLP member: 131,072 buckets, dim 64, hidden 128, 1–5-grams plus words, `ticket_v2`, dropout 0.05, Adam lr 0.003, batch 512.
- **Selection:** macro F1 on FLORES-200 **dev**. Dev contains full sentences and their 20/50/100-character prefixes, 367,729 items (`tools/export_flores_csv.py`).

```bash
.venv/bin/python train_model.py --data-dir data/prepared_v2 --valid-csv reports/flores_dev.csv \
  --model reports/exp_a/model-exp-a.pt --report-dir reports/exp_a --arch mlp --buckets 131072 \
  --embedding-dim 64 --hidden-dim 128 --ngram-max 5 --word-features --preprocessing ticket_v2 \
  --dropout 0.05 --lr 0.003 --batch-size 512 --epochs 12 --patience 4 --threads 4 --encode-workers 4
```

**Dev accuracy per epoch** (4 threads, about 8 minutes per epoch):

| Epoch | Dev accuracy |
|---:|---:|
| 1 | 96.08 |
| 2 | **96.27 (selected)** |
| 3 | 96.20 |
| 4 | 95.96 |
| 5 | 95.84 (learning rate halved) |
| 6 | 95.83 |

Training loss kept falling while dev accuracy fell after epoch 2, so the model overfits the training domains. Early stopping ended the run after epoch 6.

For reference, pooled dev accuracy over the same items is 93.93 for v21 and 96.21 for lid218e (restricted).

### Devtest

The selected checkpoint was evaluated on devtest once. External predictions were the same as in [FLORES benchmark](flores-benchmark.md).

Each cell is accuracy / macro F1 in percent:

| Slice | v21 `model.pt` | Exp A | lid218e restricted |
|---|---|---|---|
| full | 98.51 / 98.52 | 99.14 / 99.12 | 99.13 / 99.10 |
| ≤20 characters | 84.08 / 84.30 | 89.63 / 89.64 | 89.38 / 89.38 |
| 21–50 | 96.22 / 96.24 | 98.04 / 98.03 | 98.03 / 98.00 |
| 51–100 | 98.24 / 98.27 | 99.01 / 99.01 | 99.06 / 99.00 |
| >100 | 98.69 / 98.69 | 99.25 / 99.23 | 99.31 / 99.09 |
| Sorani `ckb_Arab` → ku | 0.00 | 100.00 | 100.00 |

**Paired comparison, Exp A minus lid218e restricted.** The 95% interval comes from a bootstrap over FLORES sentences; the p-value is from McNemar's test.

| Slice | Accuracy difference | 95% interval | McNemar p |
|---|---:|---|---:|
| full | +0.01 pp | −0.04 .. +0.07 | 0.66 |
| ≤20 characters | **+0.26 pp** | +0.06 .. +0.45 | 0.01 |
| 51–100 | −0.05 pp | −0.11 .. +0.02 | 0.13 |
| >100 | −0.06 pp | −0.12 .. +0.00 | 0.051 |

**Reading.**
- On FLORES devtest, Exp A **ties lid218e on full sentences** (no significant difference) and **is significantly better on texts of 20 characters or fewer**.
- It is 34.0 MB, compared with about 1.18 GB for lid218e.
- The largest remaining gaps are hr (F1 0.715 vs 0.734) and nb/da. bs→hr is still the top confusion, with 452 errors.

**Caveats.**
- This is one seed with no seed variance.
- lid218e was tuned on FLORES dev, and Exp A was selected on FLORES dev. Neither number is an out-of-domain measurement.
- Exp A has no acceptance calibration, so its `und` behaviour is not comparable with v21.
- Support-ticket diagnostics in [benchmark](benchmark.md) were not rerun.

### Size and latency

Measured on 2,000 FLORES texts, single thread, with the cache cleared before every call:

| Model | File size | Parameters | Median `predict` |
|---|---|---|---|
| v21 `model.pt` | 67.9 MB | 8.41 M per member (two-member ensemble) | 0.52 ms |
| Exp A | 34.0 MB | 8.41 M | 0.46 ms |

In both models, 99.7% of the parameters are the 131,072 × 64 embedding table.

## Exp B: post-training compression of Exp A

- **Question:** how much smaller can the Exp A checkpoint get without retraining, and what does it cost on FLORES?
- **Method:** `tools/compress_checkpoint.py` rewrites the weights and records `config["weight_storage"]`. `model_from_checkpoint` restores float32 weights at load, so the inference code is unchanged. A checkpoint without the key loads exactly as before, and an unknown value raises.
  - `fp16`: every tensor stored as float16.
  - `int8_rowwise`: the 131,072 × 64 embedding stored as int8 with a symmetric float32 scale per row (`max |w| / 127`). The small hidden and classifier layers stay float32.
- **Scope:** single-model checkpoints only. The tool refuses ensemble checkpoints such as the bundled `model.pt`, refuses already compressed input and never overwrites a file.

```bash
.venv/bin/python tools/compress_checkpoint.py --input reports/exp_a/model-exp-a.pt \
  --output reports/exp_b/model-exp-a-fp16.pt --storage fp16
.venv/bin/python tools/compress_checkpoint.py --input reports/exp_a/model-exp-a.pt \
  --output reports/exp_b/model-exp-a-int8_rowwise.pt --storage int8_rowwise
# for each checkpoint and for --split dev and devtest
.venv/bin/python tools/evaluate_flores.py --flores-dir <flores200_dataset> --split devtest \
  --model reports/exp_b/model-exp-a-int8_rowwise.pt --extra-variants --workers 2 \
  --output reports/exp_b/flores_devtest_int8_rowwise.json \
  --predictions-out reports/exp_b/flores_devtest_int8_rowwise.predictions.jsonl
```

The float32 devtest rerun gave the same prediction as the Exp A record for all 378,502 items.

**FLORES, accuracy / macro F1 in percent.** Dev was used to check the variants. Devtest is reported once.

| Split | Slice | fp32 (Exp A) | fp16 | int8_rowwise |
|---|---|---|---|---|
| dev | full | 99.14 / 99.13 | 99.14 / 99.13 | 99.14 / 99.13 |
| dev | ≤20 | 89.81 / 89.81 | 89.81 / 89.81 | 89.81 / 89.81 |
| dev | 21–50 | 97.96 / 97.94 | 97.96 / 97.94 | 97.96 / 97.95 |
| dev | 51–100 | 99.00 / 99.01 | 99.00 / 99.01 | 99.00 / 99.00 |
| dev | >100 | 99.24 / 99.21 | 99.24 / 99.21 | 99.24 / 99.21 |
| devtest | full | 99.14 / 99.12 | 99.14 / 99.12 | 99.14 / 99.12 |
| devtest | ≤20 | 89.63 / 89.64 | 89.63 / 89.64 | 89.66 / 89.66 |
| devtest | 21–50 | 98.04 / 98.03 | 98.04 / 98.03 | 98.04 / 98.02 |
| devtest | 51–100 | 99.01 / 99.01 | 99.01 / 99.01 | 99.01 / 99.01 |
| devtest | >100 | 99.25 / 99.23 | 99.25 / 99.23 | 99.25 / 99.23 |

**Agreement with fp32 per item.** Counts cover all items, including the `--extra-variants` items.

| Split | Variant | Items | Different predictions | Agreement |
|---|---|---:|---:|---:|
| dev | fp16 | 370,720 | 6 | 99.998% |
| dev | int8_rowwise | 370,720 | 90 | 99.976% |
| devtest | fp16 | 378,502 | 6 | 99.998% |
| devtest | int8_rowwise | 378,502 | 110 | 99.971% |

**Paired accuracy difference, int8_rowwise minus fp32** (`paired_comparison`: bootstrap over sentences, McNemar):

| Split | Slice | Difference | 95% interval | int8-only / fp32-only correct | McNemar p |
|---|---|---:|---|---:|---:|
| dev | full | +0.000 pp | −0.005 .. +0.005 | 3 / 3 | 1 |
| dev | ≤20 | +0.000 pp | −0.012 .. +0.012 | 20 / 20 | 1 |
| devtest | full | +0.000 pp | −0.005 .. +0.006 | 4 / 4 | 1 |
| devtest | ≤20 | +0.021 pp | +0.007 .. +0.036 | 38 / 17 | 0.007 |

Every other slice is within ±0.005 pp with p ≥ 0.27. For fp16 every slice is within ±0.002 pp and p ≥ 0.48.

**Size, load and latency.** Measured with one torch thread while another training job used the same 4-core machine, so absolute times are noisy.

| Variant | File size | Load time (median of 5 reloads, 3 processes) | RSS growth at load | Peak RSS after the first load | Median `predict`, interleaved |
|---|---:|---|---:|---:|---:|
| fp32 | 34.0 MB | 0.12–0.13 s | 37.5 MB | 581 MB | 0.452 ms |
| fp16 | 17.2 MB (−49%) | 0.12–0.18 s | 37.9 MB | 598 MB | 0.432 ms |
| int8_rowwise | 9.4 MB (−72%) | 0.17–0.20 s | 39.0 MB | 623 MB | 0.422 ms |

- Latency comes from 2,000 FLORES texts, three rounds, with all three detectors in one process and each call interleaved across them, cache cleared before every call. The per-process runs gave medians from 0.39 to 0.69 ms with no consistent order.
- Interpreter and torch account for about 512 MB of the RSS.

**Reading.**
- fp16 halves the file and int8_rowwise cuts it by 72%. Neither changes accuracy or macro F1 by more than 0.03 pp on any slice.
- The +0.021 pp for int8 on devtest ≤20 is 55 discordant items out of 101,200. Dev shows 0.000 pp, so it is not evidence that quantization helps.
- **RAM does not shrink.** The weights are dequantized to float32 at load, so resident memory and latency stay as they are for fp32. The load peak is 17–42 MB higher because of the temporary copy.

**Caveats.**
- Only the Exp A single MLP was compressed. `model.pt` (ensemble with a calibrated acceptance policy) is refused, and its `und` behaviour under compression was not measured.
- Smaller RAM would need int8 or float16 kept in memory (an embedding-bag kernel on quantized weights). That was not tried.
- Load times and RSS come from one machine with a competing job. Support-ticket diagnostics were not rerun.

## Exp C: embedding size sweep

**Question.** Can a smaller hash table or a narrower embedding keep Exp A's accuracy?

**Setup.**
- Same data (`data/prepared_v2`), architecture (single MLP, hidden 128, n-grams 1–5 plus words, `ticket_v2`) and optimiser as Exp A.
- Only `--buckets` and `--embedding-dim` change. Each run trains for `--epochs 4 --patience 2`.
- The epoch is chosen on FLORES dev only; all three runs picked epoch 2. Devtest was then evaluated once per checkpoint with `tools/evaluate_flores.py --split devtest --extra-variants`.
- The paired difference against Exp A (bootstrap over sentences, McNemar) uses the per-item predictions of both models.

| Run | Buckets × dim | Parameters | File (fp32) | Dev acc. | Devtest full | Devtest ≤20 | Devtest >100 |
|---|---|---:|---:|---:|---:|---:|---:|
| Exp A | 131,072 × 64 | 8.41 M | 34.0 MB | 96.27 | 99.14 | 89.63 | 99.25 |
| C1 | 65,536 × 64 | 4.22 M | 17.3 MB | 96.02 | 99.05 | 89.05 | 99.16 |
| C2 | 131,072 × 32 | 4.21 M | 17.2 MB | 96.17 | 99.09 | 89.24 | 99.21 |
| C3 | 262,144 × 32 | 8.41 M | 34.0 MB | 96.33 | 99.05 | 89.86 | 99.13 |

**Paired difference against Exp A (devtest, pp):**

| Run | full | ≤20 | >100 |
|---|---|---|---|
| C1 | −0.09 (−0.14..−0.04), p=0.001 | −0.59 (−0.74..−0.42), p=3e-12 | −0.08 (−0.14..−0.02), p=0.006 |
| C2 | −0.04 (−0.10..+0.00), p=0.08 | −0.39 (−0.55..−0.23), p=3e-6 | −0.04 (−0.09..+0.02), p=0.2 |
| C3 | −0.09 (−0.14..−0.04), p=0.001 | +0.23 (+0.07..+0.40), p=0.006 | −0.11 (−0.17..−0.06), p=2e-4 |

**Paired difference against NLLB lid218e, restricted mapping (devtest, pp).** Exp A was +0.01 full and +0.26 on ≤20.

| Run | full | ≤20 |
|---|---|---|
| C1 | −0.07 (−0.13..−0.01), p=0.03 | −0.33 (−0.53..−0.13), p=0.001 |
| C2 | −0.03 (−0.09..+0.03), p=0.33 | −0.13 (−0.34..+0.06), p=0.18 |
| C3 | −0.07 (−0.14..−0.01), p=0.02 | +0.49 (+0.30..+0.68), p=8.5e-7 |

**Latency.** Single-thread median `predict` on the 2,000-text sample, cache cleared before each call. The data build was running on the same machine, so the numbers are noisy.

| Run | Median `predict` |
|---|---:|
| Exp A | 0.41 ms |
| C1 | 0.40 ms |
| C2 | 0.52 ms |
| C3 | 0.35 ms |

Feature hashing dominates the time, so model size does not measurably change latency.

**Reading.**
- Halving the model costs short-text accuracy either way. Halving the buckets (C1) costs more than halving the width (C2); C1 falls significantly below NLLB.
- To make the model smaller, int8 storage of Exp A (Exp B: 9.4 MB, ≤0.03 pp) is clearly better than a smaller architecture.
- Trading width for buckets (C3) improves short text significantly (+0.23 pp vs Exp A, +0.49 pp vs NLLB) but loses 0.1 pp on long text. It is a length trade-off, not a free gain.
- The trade-off points toward more buckets for short inputs. It also motivates separate word and char bags, so words do not share the char-n-gram table. Both are untested.

**Caveats.**
- One seed per configuration; seed variance was not measured. Differences of about 0.05 pp on the full slice may be within it.
- 4 epochs, selected at epoch 2 in every run. All runs overfit after epoch 2, as Exp A did.

## Exp D: three times more data (150k rows per language)

**Question.** Does more data help when the architecture stays the same as Exp A?

**Data.** `prepare_multidomain_data.py --cap 150000 --workers 3 --out data/prepared_v3`, with the same filters and domain mix as Exp A's data.

| | Rows |
|---|---:|
| Train | 14,922,860 |
| Validation | 99,205 |
| `nn` negatives | 20,000 |

- Three labels have less than 150k rows of usable text: bo 107,756, gn 123,237, tn 141,867.
- The manifest records no source errors.
- Rows dropped by filters: 272 FLORES shingle matches, 64,388 held-out matches (exact or shingle), 145,895 markup rows.
- A scan of all rows after `ticket_v2` cleanup found 0 rows with no letters.

**Training.** The configuration and selection are the same as Exp A. Only `--data-dir` changes, and features are streamed (`--stream-workers 3`) because a cache would need about 27 GB.

```bash
.venv/bin/python train_model.py --data-dir data/prepared_v3 --valid-csv reports/flores_dev.csv \
  --model reports/exp_d/model-exp-d.pt --report-dir reports/exp_d --arch mlp --buckets 131072 \
  --embedding-dim 64 --hidden-dim 128 --ngram-max 5 --word-features --preprocessing ticket_v2 \
  --dropout 0.05 --lr 0.003 --batch-size 512 --epochs 6 --patience 2 --threads 2 \
  --encode-workers 3 --stream-workers 3
```

Each epoch took about 41 minutes on 4 cores, and the trainer's peak RSS was 5.2 GB.

| Epoch | Dev accuracy | Dev macro F1 |
|---:|---:|---:|
| 1 | 96.78 | 96.74 |
| 2 | **96.84** | **96.85** (selected) |
| 3 | 96.79 | 96.79 |
| 4 | 96.71 | 96.69 (early stop) |

Exp A's best epoch reached 96.27 dev accuracy.

**FLORES devtest.** One evaluation of the selected checkpoint. Values are accuracy / macro F1.

| Slice | Exp A | Exp D | lid218e restricted |
|---|---|---|---|
| full | 99.14 / 99.12 | **99.27 / 99.26** | 99.13 / 99.10 |
| ≤20 | 89.63 | **91.24 / 91.21** | 89.38 / 89.38 |
| 21–50 | – | **98.32 / 98.30** | 98.03 / 98.00 |
| 51–100 | – | **99.14 / 99.14** | 99.06 / 99.00 |
| >100 | 99.25 / 99.23 | **99.37 / 99.36** | 99.31 / 99.09 |
| extra variants (zho_Hant, prs, ckb) | – | **100.00** | 97.66 |

**Paired difference** (bootstrap over sentences, McNemar):

| Slice | Exp D − lid218e restricted | Exp D − Exp A |
|---|---|---|
| full | +0.15 pp (+0.10..+0.20), p=1.5e-7 | +0.13 pp (+0.09..+0.18), p=5e-8 |
| ≤20 | +1.86 pp (+1.66..+2.04), p=3e-85 | +1.60 pp (+1.45..+1.76), p=2e-92 |
| 21–50 | +0.29 pp (+0.21..+0.37), p=7e-11 | – |
| 51–100 | +0.09 pp (+0.03..+0.15), p=0.003 | – |
| >100 | +0.07 pp (+0.00..+0.12), p=0.03 | +0.13 pp (+0.07..+0.18), p=2e-6 |

Against fastText lid.176 (restricted), the difference is +11.09 pp on full sentences and +19.93 pp on ≤20.

**Hard pairs** (devtest full, F1):

| Language | Exp A | Exp D | lid218e |
|---|---:|---:|---:|
| bs | 0.621 | 0.673 | 0.596 |
| hr | 0.715 | 0.737 | 0.734 |
| ms | 0.946 | 0.951 | 0.953 |
| id | 0.943 | 0.954 | 0.950 |
| nb | 0.984 | 0.991 | 0.996 |
| da | 0.986 | 0.991 | 0.995 |

- The largest remaining confusions are bs→hr (385), hr→bs (193) and ms→id (85).
- Per-language F1 trails lid218e most on sr (−0.009), nb and da (−0.004 each).

**Reading.**
- With the architecture unchanged, three times more data makes the model significantly better than lid218e on every length slice of FLORES-200 devtest for these 100 codes.
- The largest gain is on short texts: +1.86 pp on ≤20 characters.
- The model is also significantly better than Exp A everywhere, so data volume was the main limit.

**Caveats.**
- One seed.
- The lid218e numbers use the restricted mapping onto our 100 codes; see [FLORES benchmark](flores-benchmark.md).
- bs/hr remain at about 0.7 F1. FLORES treats them as separate languages, and their sentences are often hard to tell apart.
- Not done for this checkpoint:
  - acceptance calibration (`und`) and the support-ticket diagnostics (done in Exp E);
  - compression; Exp B suggests int8 would give about 9.4 MB.

## Exp E: acceptance policy for Exp D

**Question.** Which `und` (abstain) policy should the single Exp D network carry, and how does the calibrated checkpoint compare with v21 on data that never took part in fitting?

**Tool.** `tools/fit_acceptance.py` (commit `a91c6be`). It writes the policy into `checkpoint["calibration"]` in the v21 schema, using only keys that `predict_details` reads (`temperature`, `min_confidence`, `min_confidence_by_language`, `min_confidence_by_language_and_length`, `min_margin_by_length`) plus the `rule` string. Provenance goes into `checkpoint["acceptance_fit"]`: data paths, SHA-256 digests, row counts, overlap removals, the cross-fit table and the git commit.

**Method.**
- **Temperature.** Fitted by NLL on the labelled fitting texts.
- **Thresholds.** Each threshold maximises `correct − 25 × wrong` over the texts it accepts. λ = 25 is the value v21 used, so the last text accepted in any cell has at least 96% precision.
  - Cells with fewer than 200 texts fall back to the next coarser threshold: (language, length) → language → global.
  - Length groups come from `evidence_profile`, so they match inference.
- **Negatives.** Nynorsk texts always count as wrong. Each one weighs 0.41, so the whole Nynorsk set counts as much as one average label (`--negative-scale 1`).
- **Choosing components.** Two-fold cross-fitting decides which components stay. Each candidate is fitted on one half and scored on the other, then the halves swap. FLORES lines are kept together, so all prefixes and all translations of a sentence fall in the same half. The candidate with the highest held-out objective is refitted on all fitting data. On a tie, the simpler candidate wins.
- **Check after writing.** The tool reloads the written checkpoint and compares `predict_details` with its own vectorised simulation on 3,000 fitting texts. 2,988 agreed. The other 12 were float ties at a threshold and were skipped. Any other disagreement would have aborted the run.

**Fitting data.** All of it is allowed by the split policy.

| Source | Rows used |
|---|---:|
| FLORES-200 dev, 100 codes (full sentences + 20/50/100-char prefixes) | 365,313 |
| `data/prepared_v3/valid.csv` | 99,200 |
| Public validation `data/prepared/valid_short.csv` (rebuilt with `prepare_public_data.py` from the local archives) | 100,169 |
| Negatives: FLORES dev `nno_Latn` (full + prefixes) | 3,699 |
| Negatives: fitting half of `prepared_v3/negatives/nn.csv` (stable hash < 0.5) | 9,985 |

- The tool refuses FLORES devtest and any file named `*test*`.
- It also refuses any fitting text whose `text_key` occurs in reporting data: FLORES devtest (all codes and `nno_Latn`) and the public `test_short.csv`.
- 2,427 such overlaps were removed with `--drop-reporting-overlap` and are recorded in the checkpoint. Almost all of them (2,416) are short FLORES dev prefixes that equal devtest prefixes.

**Command.**

```bash
.venv/bin/python tools/fit_acceptance.py --model reports/exp_d/model-exp-d.pt \
  --out reports/exp_d/model-exp-d-calibrated.pt --report reports/exp_e/fit_report.json \
  --flores-dir data/raw_multidomain/flores200_dataset --flores-negative nno_Latn \
  --csv prepared_v3_valid=data/prepared_v3/valid.csv --csv public_valid=data/prepared/valid_short.csv \
  --negatives prepared_v3_nn=data/prepared_v3/negatives/nn.csv \
  --reporting-csv data/prepared/test_short.csv --drop-reporting-overlap --workers 2
```

**Component choice** (held-out halves of the fitting data, both folds summed; coverage and precision over labelled texts):

| Candidate | Held-out objective | Coverage | Accepted precision | Accepted errors | Nynorsk accepted |
|---|---:|---:|---:|---:|---:|
| none | 2,286 | 100.00 | 97.131 | 16,201 | 13,684 / 13,684 |
| global | 379,727 | 87.56 | 99.893 | 527 | 9,788 |
| global + margin | 382,065 | 88.03 | 99.905 | 473 | 9,957 |
| length | 381,101 | 87.79 | 99.906 | 465 | 9,938 |
| length + margin | 382,250 | 87.54 | 99.921 | 392 | 9,878 |
| language | 483,842 | 89.94 | 99.825 | 889 | 90 |
| **language + length** (selected) | **486,716** | **90.43** | 99.827 | 882 | 97 |
| language + length + margin | 486,674 | 90.43 | 99.827 | 884 | 97 |

- **Kept:** per-language and per-(language, length) confidence thresholds. Most of the objective comes from rejecting Nynorsk inside the `nb` cell.
- **Dropped:**
  - Margin floors lower the held-out objective by 42 once the language thresholds are in place. With one softmax, the margin adds nothing beyond confidence.
  - Ensemble agreement does not exist for one network.
  - Lexical single-word tables (`ambiguous_single_words`, `single_word_language`, `word_evidence_case_sensitive`) are not fitted. Only 1,671 fitting texts (0.3%) are a single Latin word. Under the confidence policy their accepted precision is already 99.07% (8 errors), so this data cannot show a benefit. v21's tables came from v21's training vocabulary and were not copied.

**Result of the fit** (in-sample, fitting data):
- Temperature 1.1685.
- Global threshold 0.974. There are 100 per-language thresholds and 180 (language, length) cells.
- Coverage 90.20%, accepted precision 99.902%, 72 / 13,684 Nynorsk texts accepted.

**Reporting-only comparison with v21.** The calibrated Exp D was evaluated once, after the policy was fixed. Commands:

```bash
.venv/bin/python tools/evaluate_flores.py --flores-dir data/raw_multidomain/flores200_dataset --split devtest \
  --extra-variants --model <model> --workers 2 --output <json> --predictions-out <jsonl>
.venv/bin/python tools/evaluate_acceptance.py --model v21=model.pt \
  --model expd_calibrated=reports/exp_d/model-exp-d-calibrated.pt --csv public_test=data/prepared/test_short.csv \
  --negatives prepared_v3_nn=data/prepared_v3/negatives/nn.csv --flores-dir data/raw_multidomain/flores200_dataset \
  --flores-negative nno_Latn --flores-predictions v21=<v21 jsonl> --flores-predictions expd_calibrated=<exp d jsonl> \
  --tickets tests/fixtures/authored_tickets.jsonl --workers 2 --output reports/exp_e/reporting_comparison.json
```

FLORES-200 devtest. Each cell gives coverage / accepted precision (%), accepted errors, and macro FPR of accepted answers (%):

| Slice | Texts | v21 | Exp D calibrated |
|---|---:|---|---|
| full | 101,200 | 93.63 / 99.960, 38, 0.0004 | **95.60 / 99.996, 4, 0.0000** |
| ≤20 | 101,200 | 71.09 / 96.87, 2,252, 0.0225 | **76.38 / 99.52, 369, 0.0037** |
| 21–50 | 101,135 | 88.17 / 99.554, 398, 0.0040 | **93.12 / 99.941, 56, 0.0006** |
| 51–100 | 98,885 | 92.75 / 99.918, 75, 0.0008 | **95.26 / 99.993, 7, 0.0001** |
| >100 | 74,246 | 94.08 / 99.974, 18, 0.0002 | **95.84 / 99.996, 3, 0.0000** |
| extra variants (zho_Hant, prs, ckb) | 3,036 | 69.07 / 95.61, 92 | **99.87 / 100.00, 0** |

Public test (`test_short.csv`, 189,599 rows, rebuilt locally; see caveats). Same cell format:

| Group | Texts | v21 | Exp D calibrated |
|---|---:|---|---|
| all | 189,599 | **95.39** / 99.676, 586, 0.0031 | 91.63 / **99.770, 399, 0.0021** |
| short (≤3 words) | 14,120 | **88.63** / 99.409, 74 | 73.28 / **99.749, 26** |
| long | 175,479 | **95.94** / 99.696, 512 | 93.11 / **99.772, 373** |

- Raw accuracy is 98.10% for v21 and 98.00% for Exp D.
- The public data are v21's own domain: v21 was trained on WiLI/MASSIVE train, while Exp D held these sources' validation and test parts out.

Nynorsk (not a label). Accepted means answered as some label:

| Set | v21 accepted | Exp D calibrated accepted |
|---|---|---|
| Reporting half of `nn.csv` (10,015) | 6,612 (66.0%): nb 6,290, da 128, de 90, sv 50 | **85 (0.85%)**: sv 27, da 17, en 16, is 10, nb 7 |
| FLORES devtest `nno_Latn` (3,777) | 2,506 (66.3%): nb 2,395, da 43 | **19 (0.50%)**: sv 8, is 5, nb 2, da 1 |

Authored tickets (84 rows, AI-authored, not human-verified). C = correct, E = accepted wrong, U = withheld:

| API | v21 | Exp D calibrated |
|---|---|---|
| whole-ticket `predict_details` (= `predict_ticket(allow_uncertain=True)`) | 70 C / 5 E / 9 U | 68 C / 6 E / 10 U |
| `predict_guarded_ticket` | 70 C / 0 E / 14 U | 68 C / 1 E / 15 U |

- Exp D loses two single-language tickets:
  - an English one: raw top-1 `tn` at 0.18;
  - a Czech one: confidence 0.53, below the `cs` threshold.
- The guarded API accepts one mixed error-quote ticket as `de`; v21 withheld it.
- All E come from the 13 mixed or error-quote tickets (gold `und`).

**Reading.**
- On FLORES devtest the calibrated Exp D answers more and is more precise than v21 on every length slice. On ≤20 characters it accepts 6.1× fewer errors at 5.3 pp more coverage.
- Nynorsk is now rejected: 0.5–0.9% accepted, against 66% for v21.
- On the public test, Exp D is more precise but answers 3.8 pp less, mostly on short texts.
- On the tickets, the results are about the same, slightly worse: −2 C, +1 E.

**Caveats.**
- **Bokmål is effectively not answered.** Rejecting Nynorsk costs `nb`: the `nb` short cell rejects everything and the long cell needs confidence 1.0. Bokmål coverage on FLORES devtest full sentences is 0.2%, against 89.1% for v21. The cause is the negative weight, which is a product decision, not something the data can settle. Two alternative checkpoints were fitted with the same tool; their numbers below are cross-fitted on fitting data only, and they were not evaluated on reporting data:

  | `--negative-scale` | Held-out coverage / precision | `nb` held-out coverage | Nynorsk accepted (held out) | File |
  |---:|---|---:|---:|---|
  | 1 (shipped) | 90.43 / 99.827 | 0.7% | 97 / 13,684 | `reports/exp_d/model-exp-d-calibrated.pt` |
  | 0.1 | 90.70 / 99.827 | 18.6% | 831 / 13,684 | `reports/exp_e/model-exp-d-calibrated-nnscale0.1.pt` |
  | 0 (measure only) | 91.42 / 99.823 | 78.5% | 9,604 / 13,684 | `reports/exp_e/model-exp-d-calibrated-nnscale0.pt` |

- **English confidence is low.** On FLORES dev long English sentences, Exp D's median top-1 probability is 0.83; for de it is 0.996. The runner-up is usually a lower-resource label (su, tl, sn, yo, ga), which suggests English text inside those labels' training data. The `en` cells therefore need 0.80 (long) and 0.86 (short), and English coverage is 64.8% on FLORES devtest full sentences (v21: 45.6%). The fix belongs in the training data, not in the policy.
- bs/hr stay almost entirely withheld (1.1–1.4% coverage), as in v21.
- The public splits were rebuilt locally with `prepare_public_data.py` from the same archives (valid 100,173 / test 189,599 rows). v21's documented counts are 100,175 / 189,557, so the files are not byte-identical to v21's; probably different library versions in the cleanup, which was not investigated. Both models were scored on the same rebuilt file.
- FLORES dev was already used to choose Exp D's checkpoint (epoch selection), and now its thresholds as well. Devtest remains untouched by both.
- One fit, one seed, λ fixed in advance at v21's value; no λ sweep was run.
- `tools/fit_acceptance.py` computes logits through the current torch `LanguageDetector` internals (`encode`, `model`). If inference moves to numpy, this one function has to follow. The policy itself depends only on `predict_details` semantics.
