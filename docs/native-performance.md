# Optional native encoder measurements

Measured on 2026-10-08 using the unchanged bundled checkpoint. The default
Python encoder remains available; the optional C module preserves exact feature
indices and speeds up character extraction. The network remains PyTorch.

| Input/protocol | Python encoder | Native encoder | fastText .bin | fastText .ftz |
| --- | ---: | ---: | ---: | ---: |
| 64 characters, repeated/warm | 0.226 ms | 0.096 ms | 0.047 ms | 0.049 ms |
| 256 characters, repeated/warm | 0.779 ms | 0.286 ms | 0.154 ms | 0.156 ms |
| 2,000 characters, repeated/warm | 6.044 ms | 1.999 ms | 1.127 ms | 1.135 ms |
| 1,000 varied public texts, median | 0.173 ms | 0.106 ms | 0.043 ms | 0.037 ms |
| 1,000 varied public texts, p95 | 0.592 ms | 0.315 ms | 0.180 ms | 0.125 ms |

These are local single-input CPU timings, not universal performance guarantees.
The native encoder is about 2.35x faster for repeated 64-character input and
1.64x faster for the varied sample. fastText remains faster in both protocols.
No change in classification quality is claimed from this optimization.

## Protocol and cache

Environment: macOS ARM64, Python 3.13.0, PyTorch 2.14.1, two CPU threads.
Repeated-input measurements use 30warmup calls followed by 1,000timed calls.
The native fragment cache is warm: repeated text gives unusually high hit rates.

The varied sample is selected with seed 913 from the prepared public held-out
`test_short.csv`, using 1,000 different records. The native cache is cleared once
before that pass; it can populate across successive requests, as in a running
process. This is not a test where the cache is cleared before every request.
The cache stores exact hashes of short fragments (up to 20 UTF-8 bytes, 32,768 slots,
about 1 MiB) and does not cache whole-text predictions. See [native details](../native/README.md).

Timings include shared cleanup, NFC/lowercase/2,000-character cap, feature encoding
and classification; imports, model loading and structured-description extraction
are excluded. Identical model input/preprocessing is used for fastText. The
benchmark checks exact Python/native feature equality on all 1,000 sample inputs.
It measures speed, not domain accuracy; assess accuracy on independent labeled
examples from the intended application before deployment.

## Reproduce

Build the [native encoder](../native/README.md), prepare the public corpus following
[training instructions](training.md), and optionally install `fasttext==0.9.3`.
Obtain the official [fastText language identification models](https://fasttext.cc/docs/en/language-identification.html).

```bash
python benchmark_native.py --csv data/prepared/test_short.csv \
  --fasttext-bin /path/to/lid.176.bin --fasttext-ftz /path/to/lid.176.ftz
```

Without fastText paths, the script compares only the two Wind-Up encoders.
The default CSV is the small bundled synthetic diagnostic; its timings should
be distinguished from the 1,000-record public sample above. Raw inputs and
compiled extensions are excluded from Git.

[Machine-readable baseline](../.ecc/benchmarks/native.json) includes p95 timings,
environment details, checksums and protocol. Performance varies with hardware,
compiler, languages, content length, cache state and system load.
