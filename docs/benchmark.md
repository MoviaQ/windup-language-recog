# Bundled v21 checkpoint evaluation

v21 keeps the v20 raw neural weights, ensemble members, weights, and temperatures; only the acceptance calibration changed. v20 combined a support-adapted MLP with the earlier v4 linear network. All predictions choose among the same 100 languages; the optional acceptance policy can abstain. These are local diagnostics, not production guarantees.

## Accepted responses

Cells report **correct accepted / incorrect accepted**. Other inputs receive `und`. The same runner and texts were used for the historical comparisons.

| Diagnostic | Texts | v4 | v7 | v20 | v21 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical multilingual application set | 200 | 170 / 0 | 169 / 0 | 183 / 0 | 183 / 0 |
| Historical EN/DE/PL/CS set | 198 | 172 / 2 | 173 / 2 | 174 / 2 | 174 / 2 |
| Earlier application set | 120 | 95 / 0 | 96 / 0 | 104 / 0 | 104 / 0 |
| Earlier application set | 114 | 85 / 0 | 87 / 0 | 97 / 0 | 97 / 0 |
| Authored application diagnostics | 1214 | 880 / 3 | 894 / 3 | 982 / 8 | 986 / 8 |
| Difficult-language diagnostics | 339 | 209 / 13 | 211 / 12 | 217 / 12 | 224 / 11 |
| Public general-text test | 189557 | 177728 / 504 | 177657 / 515 | 179105 / 637 | 180138 / 595 |

On the 117 English inputs within the EN/DE/PL/CS set, v4 accepted 99 correct, v7 100 correct, and v20 101 correct, with no accepted errors; v21 keeps the same 101 because this set's accepted totals are unchanged. This is a small English improvement, not evidence of a large or universal gain.

Historical sets were evaluated repeatedly. Their errors informed new hypotheses, although their rows/labels did not train weights, populate the lexical tables, or fit thresholds. Adaptive development and repeated stopping decisions limit their evidential value. Authored examples are correlated and labels are not independently verified human gold.

## Final held-out application sample

Before final candidate predictions, a separate 64-message sample and its assistant-assigned labels were locked. It was not used for selection or subsequent retuning. Exact normalized overlap with known training/development/evaluation inputs was excluded; related templates can remain.

- v4: 52/64 correct accepted, 0 incorrect accepted.
- v7: 53/64 correct accepted, 0 incorrect accepted.
- v20: 56/64 correct accepted, 0 incorrect accepted.
- v21: 55/64 correct accepted, 0 incorrect accepted.

The sample has 36 German, 23 English, 2 Czech, and 3 ambiguous/metadata inputs. v20 accepted 34 German, 20 English, and 2 Czech responses correctly and rejected all three ambiguous inputs. v21 loses one correct accepted response to the new margin floor but still rejects all three ambiguous inputs. Assistant labels need independent human verification; this is not a statistical certification.

## Public-text tradeoff and cost

- v4: raw accuracy 98.1140%; macro recall 96.5408%; accepted precision 99.7172%.
- v7: raw accuracy 98.0992%; macro recall 96.5425%; accepted precision 99.7110%.
- v20: raw accuracy 98.0876%; macro recall 96.5913%; accepted precision 99.6456%.
- v21: raw accuracy 98.0876%; macro recall 96.5913%; accepted precision 99.6708%.

v21 keeps v20's raw accuracy and macro recall unchanged because the weights are unchanged. On the 189,557 public test texts, v21 accepts more correct responses and fewer errors than v20 (595 versus 637; v7 had 515), with higher coverage (95.34% versus 94.82%). Assess both coverage and precision for your domain. Two network passes and about 67.9 MB of weights replace the earlier single network of about 33.6 MB. No new latency measurement is claimed.

## Public packaging and reproducibility

Bundled model SHA-256: `358586ac346be2f4c263d90a6c521cf766a94a3b60787e8efefd424e4337579b`.
Checkpoint size: 67,879,025 bytes.

The public checkpoint omits working metadata and retains only lexical terms present in public training text. Public runtime/package parity is checked against the locked local v21 application predictions, including the 64-message sample. v21 preserves v20's raw neural weights, calibration temperatures, and ensemble weights, changing only the acceptance calibration. Source application records and full adaptation artifacts are not distributed. The complete adapted ensemble is therefore not reproducible from the base public training commands alone.

The [machine-readable v21 summary](evaluation-v21.json) identifies this checkpoint. [Original public-base measurements](evaluation.json) remain historical results for the original base hash and must not be attributed to v21. See the [model card](model-card.md) for architecture and selection limitations.
