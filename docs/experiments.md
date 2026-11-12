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
