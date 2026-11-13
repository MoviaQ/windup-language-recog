# Architecture

## Feature pipeline

1. `ticket_text.py` fixes common encoding artifacts with ftfy and unescapes HTML. It removes selected URLs, email addresses, fenced code, markup, numbers, and technical identifiers.
2. `encode` normalizes Unicode to NFC, lowercases, and limits text length. It rejects cleaned input without letters.
3. Every character n-gram of lengths 1–5 is hashed with stable BLAKE2b into one of 131,072 buckets. Whole words are separately hashed with a namespace prefix, then repeated four times to increase their weight.
4. `encode_batch` flattens feature IDs and records the start of each input's feature bag.
5. The network averages the 64-dimensional vectors of each bag (an EmbeddingBag with mean pooling). The v22 MLP adds a 128-unit ReLU layer and outputs 100 logits. The largest logit selects the raw label.
6. Detail mode applies the stored temperature and the acceptance policy. For v22, that is confidence cutoffs per predicted language and length group.

```text
Hashed features → mean embedding (int8 table) → ReLU MLP → logits → raw language code
                                          temperature + cutoffs → code or und
```

**Inference runs on numpy** (`numpy_runtime.py`). The pooling is `np.add.reduceat` over the gathered rows. An int8 table stays int8 in memory: only the gathered rows are converted (value × row scale), which gives exactly the same float32 rows as dequantizing the whole table first.

The runtime still supports the older formats:

- linear networks;
- weighted ensembles with an agreement gate (v20/v21);
- fp16 storage;
- per-length top-2 margin floors and standalone-word evidence tables stored in the calibration.

Hashing bounds neural-network size. Collisions share parameters; they are a deliberate memory/accuracy tradeoff. Mean pooling is cheap but discards feature order beyond that encoded inside each n-gram.

## Optional ticket guard

`ticket_guard.py` reuses the same request source the whole-ticket text sees (`ticket_text.select_request_source`, then the same description-else-title choice), so guard and whole vote can never disagree about which field they read. The source is split into lines with newlines preserved; each line is cleaned exactly like the whole input, and only the first 2,000 normalized characters may supply sentences (a straddling line is clipped, later lines excluded; if per-line mapping ever drifts from the whole text, segmentation falls back to the cleaned whole text). Sentences need at least 3 Unicode letter words; digit/underscore tokens never count. `predict_guarded_ticket` scores the whole text through `predict_details` acceptance, then each sentence raw; one foreign sentence at or above the cutoff (default 0.65) withholds the answer as `und`. No fastText, no extra weights, no training.

## Training

`prepare_public_data.py` downloads checksum-verified public archives, maps source language IDs, retains source partitions, excludes normalized duplicates/conflicts, and builds short examples. Synthetic evaluation examples are reserved from the corpus. The normalization key considers up to 384 characters; this prevents exact normalized overlap under that rule, not every semantic paraphrase or source-level overlap.

`EncodedDataset` caches feature IDs as disk-backed int32 arrays; `StreamingDataset` (`--stream-workers`) encodes batches on the fly instead, with a bounded number of batches in flight. Batches convert IDs to PyTorch int64 tensors. Training uses Adam, inverse-square-root frequency class weights, and feature dropout. A plateau scheduler reduces the learning rate, and validation macro F1 selects the checkpoint. The held-out test partition is reserved for evaluation.

## Checkpoints and release files

A training checkpoint (`.pt`, PyTorch) contains `state_dict`, supported `languages`, feature/model `config`, and training provenance. Optional `calibration` is separate from architecture configuration. v20 added `ensemble` with companion checkpoint weights, member temperatures, mixing weights, and an agreement requirement. `config["weight_storage"]` records fp32, fp16 or int8_rowwise storage.

A release file (`.npz`, written by `tools/export_numpy.py`) holds the same weights as arrays (`member<i>/<name>`) plus one JSON metadata array:

- the language list;
- each member's config;
- ensemble weights;
- `calibration`;
- `source_metadata`, the remaining checkpoint entries, such as training and acceptance-fit provenance;
- `export`, the source file name and sha256.

The file is read with `allow_pickle=False` and validated: shapes against the config, every array accounted for, and the format version.

`LanguageDetector` loads `.npz` files with numpy only. It still loads `.pt` files, using `torch.load(..., map_location="cpu", weights_only=True)` and then the same numpy forward pass. Missing configuration fields default for older small demonstration checkpoints; this is compatibility behavior, not the release architecture.

No server, port, database, API key, GPU, or remote tokenizer is required for inference. Input text remains within the calling process. Dependencies must first be installed locally.
