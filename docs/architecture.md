# Architecture

## Feature pipeline

1. `ticket_text.py` fixes common encoding artifacts with ftfy and unescapes HTML. It removes selected URLs, email addresses, fenced code, markup, numbers, and technical identifiers.
2. `encode` normalizes Unicode to NFC, lowercases, and limits text length. It rejects cleaned input without letters.
3. Every character n-gram of lengths 1–5 is hashed with stable BLAKE2b into one of 131,072 buckets. Whole words are separately hashed with a namespace prefix, then repeated four times to increase their weight.
4. `batch` flattens feature IDs and records the start of each input's feature bag.
5. `nn.EmbeddingBag(..., mode="mean")` averages a bag's 64-dimensional learned vectors. `nn.Linear(64, 100)` produces one logit per language.
6. The largest logit selects the predicted language. Detail mode applies optional temperature calibration before computing confidence.

```text
IDs:      [features for text A | features for text B | ...]
Offsets:  [0, start of B, ...]
                     ↓
EmbeddingBag: [number of inputs, 64]
                     ↓
Linear:       [number of inputs, 100]
                     ↓
argmax:       one language code per input
```

Hashing bounds model size and avoids a separate vocabulary. Collisions share parameters; they are a deliberate memory/accuracy tradeoff. Mean pooling is cheap but discards feature order beyond that encoded inside each n-gram.

## Training

`prepare_public_data.py` downloads checksum-verified public archives, maps source language IDs, retains source partitions, excludes normalized duplicates/conflicts, and builds short examples. Synthetic evaluation examples are reserved from the corpus. The normalization key considers up to 384 characters; this prevents exact normalized overlap under that rule, not every semantic paraphrase or source-level overlap.

`EncodedDataset` caches feature IDs as disk-backed int32 arrays. Batches convert IDs to PyTorch int64 tensors. Training uses Adam, inverse-square-root frequency class weights, and feature dropout. A plateau scheduler reduces the learning rate, and validation macro F1 selects the checkpoint. The held-out test partition is reserved for evaluation.

## Checkpoints

A checkpoint contains `state_dict`, supported `languages`, feature/model `config`, and training provenance. Optional `calibration` is separate from the architecture configuration. The loader uses `torch.load(..., map_location="cpu", weights_only=True)` and defaults missing configuration fields for older small demonstration checkpoints. These defaults are compatibility behavior, not the release architecture.

No server, port, database, API key, GPU, or remote tokenizer is required for inference. Input text remains within the calling process. Dependencies must first be installed locally.
