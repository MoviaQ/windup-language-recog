# Architecture

## Feature pipeline

1. `ticket_text.py` fixes common encoding artifacts with ftfy and unescapes HTML. It removes selected URLs, email addresses, fenced code, markup, numbers, and technical identifiers.
2. `encode` normalizes Unicode to NFC, lowercases, and limits text length. It rejects cleaned input without letters.
3. Every character n-gram of lengths 1–5 is hashed with stable BLAKE2b into one of 131,072 buckets. Whole words are separately hashed with a namespace prefix, then repeated four times to increase their weight.
4. `batch` flattens feature IDs and records the start of each input's feature bag.
5. Each network averages 64-dimensional vectors with `nn.EmbeddingBag(..., mode="mean")`. The MLP adds a 128-unit ReLU layer; the earlier linear network directly classifies the pooled vectors. Both output 100 logits.
6. v21 combines temperature-normalized logits with weights 0.75 (MLP) and 0.25 (linear), unchanged from v20. The largest combined logit selects the raw label. Detail mode applies final temperature scaling, consensus, language/length and per-length top-2 margin acceptance rules, and standalone-word checks.

```text
Hashed features → EmbeddingBag → ReLU MLP → calibrated logits ─┐
                → EmbeddingBag → Linear   → calibrated logits ─┤
                                  weighted sum → raw language code
                                  acceptance policy → code or und
```

Hashing bounds neural-network size. The optional standalone-word policy also uses evidence tables embedded in the checkpoint. Collisions share parameters; they are a deliberate memory/accuracy tradeoff. Mean pooling is cheap but discards feature order beyond that encoded inside each n-gram.

## Optional ticket guard

`ticket_guard.py` reuses the same request source the whole-ticket text sees (`ticket_text.select_request_source`, then the same description-else-title choice), so guard and whole vote can never disagree about which field they read. The source is split into lines with newlines preserved; each line is cleaned exactly like the whole input, and only the first 2,000 normalized characters may supply sentences (a straddling line is clipped, later lines excluded; if per-line mapping ever drifts from the whole text, segmentation falls back to the cleaned whole text). Sentences need at least 3 Unicode letter words; digit/underscore tokens never count. `predict_guarded_ticket` scores the whole text through `predict_details` acceptance, then each sentence raw; one foreign sentence at or above the cutoff (default 0.65) withholds the answer as `und`. No fastText, no extra weights, no training.

## Training

`prepare_public_data.py` downloads checksum-verified public archives, maps source language IDs, retains source partitions, excludes normalized duplicates/conflicts, and builds short examples. Synthetic evaluation examples are reserved from the corpus. The normalization key considers up to 384 characters; this prevents exact normalized overlap under that rule, not every semantic paraphrase or source-level overlap.

`EncodedDataset` caches feature IDs as disk-backed int32 arrays. Batches convert IDs to PyTorch int64 tensors. Training uses Adam, inverse-square-root frequency class weights, and feature dropout. A plateau scheduler reduces the learning rate, and validation macro F1 selects the checkpoint. The held-out test partition is reserved for evaluation.

## Checkpoints

A checkpoint contains `state_dict`, supported `languages`, feature/model `config`, and training provenance. Optional `calibration` is separate from architecture configuration. v20 added `ensemble` with companion checkpoint weights, member temperatures, mixing weights, and an agreement requirement; v21 keeps that ensemble and changes only the acceptance calibration. Loader validation checks compatible language ordering and feature encoding. Calibration includes language/length cutoffs, per-length top-2 margin floors, and case-sensitive single-word tables. The loader uses `torch.load(..., map_location="cpu", weights_only=True)` and defaults missing configuration fields for older small demonstration checkpoints. These defaults are compatibility behavior, not the release architecture.

No server, port, database, API key, GPU, or remote tokenizer is required for inference. Input text remains within the calling process. Dependencies must first be installed locally.
