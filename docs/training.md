# Reproduce public-data training

These commands reproduce the public **WiLI-2018** and **MASSIVE 1.1** base-model workflow. The bundled v22 model is trained instead on the multidomain corpus (see "Larger corpora" below and [experiments](experiments.md), Exp D, Exp E and the v22 release section, for the exact commands). Training is optional: normal inference uses the bundled `model.npz` and does not need PyTorch.

Training writes PyTorch `.pt` checkpoints. To use one with the torch-free runtime, export it with `tools/export_numpy.py --input model-retrained.pt --output model-retrained.npz` (add `--storage int8_rowwise` for a 4× smaller embedding table). `LanguageDetector` also reads `.pt` files directly when PyTorch is installed.

## Prepare and train

From the repository root:

```bash
./setup.sh
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python prepare_public_data.py
.venv/bin/python train_model.py \
  --short-only --epochs 12 --patience 4 --batch-size 512 \
  --buckets 131072 --embedding-dim 64 --ngram-max 5 \
  --word-features --preprocessing ticket_v2 --dropout 0.05 --lr 0.003 \
  --model model-retrained.pt
.venv/bin/python evaluate_public.py --model model-retrained.pt
```

Use `model-retrained.pt` to preserve the released checkpoint. Defaults select validation macro F1, seed 42, two CPU threads, a 384-character training cap, and generated data in `data/prepared`. Reports are written to `reports/training.json`; features are cached under `data/cache`. Downloaded archives, prepared text, caches, and generated reports are excluded from Git.

The preparation script verifies source archive SHA-256 checksums, and records sources, raw checksums, seed, language list, sample counts, and synthetic evaluation checksum in `data/prepared/manifest.json`. Substantial RAM, disk space, download time, and CPU time may be needed. Exact elapsed time depends on the machine; dependency versions and numerical backends can affect reproducibility even with fixed seeds.

## Source handling and separation

- WiLI contributes Wikipedia-derived text for the selected supported languages. Within its source training portion, 50 examples per language are assigned to validation before normalization/exclusion; its source test portion remains test.
- MASSIVE contributes multilingual localized assistant utterances, using official train/dev/test partitions. Locale prefixes map to output codes, so two Chinese locales share `zh`.
- Cleanup runs before normalized-key duplicate checks. Keys case-fold text, remove accents and nonletters, and inspect up to 384 characters. Examples with conflicting language labels under this rule are excluded.
- Held-out synthetic examples are reserved. Test takes priority over validation, and validation over training when normalized examples overlap.
- Longer examples produce sentence or word excerpts; training may use two excerpts. Short examples remain intact. No authored demo or synthetic evaluation CSV is added to the public base training set.

These checks protect against exact normalized overlap, not semantic duplicates or every possible overlap in the underlying sources. Keep that distinction in evaluation claims.

## Learning and selection

Adam minimizes class-weighted cross entropy. Inverse-square-root class-frequency weights reduce the influence of large languages. Feature dropout removes some features while retaining one per input. Validation controls checkpoint selection and learning-rate reduction; early stopping may finish before epoch 12.

The test split must be evaluated only after selecting the checkpoint. Do not change settings because a test score looks disappointing and then describe the same test as independent. For application evaluation, use a newly collected and independently labeled sample that reflects actual message lengths and language frequencies.

## Calibration and evaluation

`evaluate_public.py --model model-retrained.pt` fits temperature on public validation data, stores it in that checkpoint, and evaluates public held-out and authored synthetic examples. This base-model workflow uses a fixed confidence cutoff of 0.9 after scaling; the cutoff is an operational choice, not a claim of 99% accuracy. Evaluation writes `reports/public_evaluation.json` and measures warmed CPU predictions at selected text lengths. The checkpoint file is updated with calibration metadata. `--model` is required, and the script refuses checkpoints that carry a release acceptance policy (such as the bundled v21 `model.pt`), because it would replace that policy with a single temperature. Latency is measured with the prediction cache cleared before every call.

## Inspect and experiment

```bash
.venv/bin/python train_model.py --help
.venv/bin/python language_detector.py predict "Please help me open this file." --model model-retrained.pt
```

Try fewer hash buckets or smaller embeddings to study the memory/accuracy tradeoff. Compare weighted accuracy, macro recall, and macro F1. Temperature calibration and optional acceptance thresholds must use validation examples; accepted accuracy must always be reported together with coverage.

See [third-party notices](../THIRD_PARTY_NOTICES.md) before redistributing source text, prepared databases, or a derivative model.

## Larger corpora (experimental)

For multi-million-row training sets:

- `--encode-workers N` builds the feature cache in N processes; the cache is byte-identical to a single-process build.
- `--sampling-alpha 0.3 --uniform-loss` draws each epoch's rows so that a language with n rows gets a share proportional to n^0.3 (temperature sampling as in NLLB, OpenLID and GlotLID). The trainer refuses `--sampling-alpha` without `--uniform-loss`, so languages are not rebalanced twice.
- `--valid-csv PATH` replaces the data-dir validation files for checkpoint selection. `tools/export_flores_csv.py --flores-dir … --output reports/flores_dev.csv` writes FLORES-200 **dev** (full sentences and 20/50/100-character prefixes); devtest stays reserved for reporting ([FLORES benchmark](flores-benchmark.md)).

### Multidomain corpus (experimental)

`prepare_multidomain_data.py` builds `data/prepared_v2/` from web (MADLAD-400, FineWeb-2 slices, MaCoCu, *WaC), Wikipedia, short-sentence (Tatoeba, MASSIVE train) and other (OpenLID-v3 permissive tags and MT560, SETimes) text. It needs `requirements-training.txt` (adds pyarrow) and network access; `--help` lists the flags.

```bash
.venv/bin/python prepare_multidomain_data.py --cap 50000 --workers 4
.venv/bin/python train_model.py --data-dir data/prepared_v2 ...   # not --short-only
```

- Every segment passes ftfy + NFC, a length filter, a per-label script filter (at least half of the letters in the label's scripts), exact and prefix/suffix near-deduplication, and a FLORES-200 filter over all 204 dev and devtest files (exact normalized match, or at least half of the segment's 8-word / 30-character shingles in FLORES). WiLI test, MASSIVE dev/test, the WiLI validation rows of `prepare_public_data.py`, `data/synthetic.csv` and the ticket fixtures are filtered the same way. Keys seen under more than one label are dropped everywhere.
- Each language gets `--cap` training rows: 70% full segments in a 40/30/15/15 web/wiki/short/other mix (a domain short of rows passes its share to the others) and 30% crops of 4–16 words (16–60 characters for unspaced scripts). `valid.csv` is an in-corpus split (`--valid` rows per language) for early stopping only.
- Label decisions: Sorani (ckb) and Kurmanji train `ku`; `azb`, `sh` and `yue` are dropped; `zh` includes Traditional script; `sr` and `uz` keep both scripts; `bs`/`hr`/`sr` stay separate. Nynorsk is not a label: its rows go to `negatives/nn.csv` for abstention calibration.
- `manifest.json` records per-language and per-source counts, every drop counter (see `counter_legend`), licences, pinned dataset revisions and a `sources_lock` entry (URL, upstream sha256 when published, bytes read and their sha256) for every remote file read. `train_sources.tsv` gives the source of every row.
- `--stream-workers N` encodes each training batch on the fly in N processes instead of building `data/cache` (about 9.2 GB per 5 million rows with the v21 encoding). Use it when the cache would not fit on disk. Batches are identical to the cached path; a one-epoch comparison on 20,000 rows gave bit-identical weights. Feature caches are keyed only on encoding settings (buckets, n-grams, word features, preprocessing), so changing `--embedding-dim`, `--hidden-dim` or `--arch` reuses a cache.
