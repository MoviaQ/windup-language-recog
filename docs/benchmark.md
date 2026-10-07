# Release benchmark

Measured for the bundled public-data checkpoint on 2026-10-07. All predictions below choose one of the 100 supported codes; results do not filter low-confidence predictions.

| Evaluation | Examples | Result |
| --- | ---: | ---: |
| Public held-out test accuracy | 189,557 | 98.29% |
| Public macro recall (equal weight per language) | 100 languages | 96.91% |
| Public macro F1 | 100 languages | 97.10% |
| Authored synthetic diagnostic accuracy | 63 examples, 20 languages | 98.41% (62/63) |

The prepared training, validation and test splits contain 595,196, 100,175 and 189,557 examples. They combine WiLI passages/excerpts and MASSIVE assistant utterances; the test is not exclusively one-sentence support messages. Macro metrics make weak performance on smaller language groups more visible than the overall accuracy. Synthetic examples are a small authored diagnostic, not evidence of production accuracy. No 99% guarantee follows from these results.

## CPU latency

| Input length (characters) | Median | 95th percentile |
| ---: | ---: | ---: |
| 64 | 0.224 ms | 0.261 ms |
| 256 | 0.800 ms | 0.907 ms |
| 2,000 | 6.126 ms | 6.383 ms |

Measured on macOS ARM64, Python 3.13.0, PyTorch 2.14.1, two CPU threads. Each length uses 30 warmup predictions followed by 1,000 sequential timed calls on a repeated Polish sentence with an English insertion. Timings include cleanup, feature encoding and the neural network; they exclude imports, checkpoint loading and structured-description extraction. This is single-input latency, not a batch-throughput measurement. Hardware, concurrent load and text content affect results.

The checkpoint is 33,584,911 bytes (32.03 MiB). PyTorch and process memory add to this size.

## Reproducibility

See [training instructions](training.md) for the complete command. Validation macro F1 selected epoch 8 of the 12-epoch run. A temperature of 1.225561 was fitted only on validation; the 0.9 uncertainty cutoff is an operational choice. Calibration changes confidence, not the highest-scoring language. Hyperparameters and checkpoint selection did not use this test's scores.

[Machine-readable measurements and per-language scores](evaluation.json) include the environment, test checksum and model checksum. Small floating-point/runtime differences can affect retraining results. Raw corpora, feature caches and local training reports are excluded from Git.

Model SHA-256: `ec2bb56baba69d1b71d7cf288d0723199f169dfff45a3425b5544fc41bb2fe30`.

These measurements do not include a fastText comparison. Comparisons require running this released checkpoint and the baseline against the same test, preprocessing, hardware and supported-language intersection.
