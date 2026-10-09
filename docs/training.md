# Reproduce public-data training

These commands reproduce the public **WiLI-2018** and **MASSIVE 1.1** base-model workflow. The bundled v20 weights additionally use support-domain adaptation and synthetic training augmentation. The complete adaptation workflow is not distributed, so these commands do not reproduce v20. Training is optional: normal inference uses the bundled `model.pt`.

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

`evaluate_public.py --model model-retrained.pt` fits temperature on public validation data, stores it in that checkpoint, and evaluates public held-out and authored synthetic examples. This base-model workflow uses a fixed confidence cutoff of 0.9 after scaling; the cutoff is an operational choice, not a claim of 99% accuracy. Evaluation writes `reports/public_evaluation.json` and measures warmed CPU predictions at selected text lengths. The checkpoint file is updated with calibration metadata. Run this workflow on `model-retrained.pt`, not on the bundled v20 checkpoint: it replaces calibration and does not reproduce the v20 ensemble acceptance policy.

## Inspect and experiment

```bash
.venv/bin/python train_model.py --help
.venv/bin/python language_detector.py predict "Please help me open this file." --model model-retrained.pt
```

Try fewer hash buckets or smaller embeddings to study the memory/accuracy tradeoff. Compare weighted accuracy, macro recall, and macro F1. Temperature calibration and optional acceptance thresholds must use validation examples; accepted accuracy must always be reported together with coverage.

See [third-party notices](../THIRD_PARTY_NOTICES.md) before redistributing source text, prepared databases, or a derivative model.
