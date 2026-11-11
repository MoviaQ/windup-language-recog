# Experiments

Experimental checkpoints are not the bundled `model.pt`. Each entry names the commit, data, command, selection rule and the single devtest evaluation made after selection.

## Exp A: v21 MLP member architecture on the multidomain corpus

- **Question:** how much do data alone change FLORES results, with the architecture unchanged?
- **Data:** `prepare_multidomain_data.py --cap 50000 --workers 4` at commit `30df1d6` gives 5,000,000 train rows, 50,000 for each of the 100 codes. FLORES dev and devtest are filtered out of the training data (see [training](training.md)).
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
