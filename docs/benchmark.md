# Bundled v4 checkpoint evaluation

The bundled weights were adapted locally on 2026-10-08 and published on 2026-10-09. Historical local evaluation results below apply to this v4 checkpoint; they are not a fresh production benchmark. Predictions choose among all 100 languages unless uncertainty mode is stated.

| Evaluation | Public base | Bundled v4 |
| --- | ---: | ---: |
| Public held-out accuracy, 189,557 texts | 98.29% | 98.11% |
| Public macro recall, 100 languages | 96.91% | 96.54% |
| Fresh ticket test, 120 examples | 85.83% | 99.17% (119/120) |
| Fresh tickets up to 100 characters, 80 examples | 85.00% | 98.75% (79/80) |
| Historical ticket test, 114 examples | 86.84% | 97.37% (111/114) |
| Held-out authored synthetic diagnostics, 63 examples | 98.41% | 100% |

Ticket labels were assigned by an assistant before prediction, not independently verified by humans. The fresh set contains 64 German, 49 English, 5 Czech, 1 Portuguese, and 1 Chinese ticket. Exact normalized overlap was excluded, but related templates may remain. These small samples do not establish 99% production accuracy or performance for every language. Historical tests and the public test were already used for earlier diagnostics.

## Uncertainty

Public-validation temperature: 1.349526. Confidence cutoff: 0.929371, selected on 58 ticket-validation examples with 43 accepted and zero observed errors. Calibration does not change the highest-scoring language.

On the fresh ticket test, uncertainty mode accepted 95/120 (79.17%) with 95 correct responses. On the historical test it accepted 85/114 with 85 correct responses and rejected six separately labeled und examples. Zero observed errors is not a guarantee.

The regression example `My VPN stopped working today. Please reinstall it.` changes from English at confidence 0.8418 (rejected by the old 0.9 cutoff) to English at 0.9958 (accepted).

## Provenance and reproducibility

Model SHA-256: `29adde51c8dc54aca5031f61730449c17ed6162ba0f79a209ce34ec36f3ccb0f`.

Checkpoint size: 33,585,423 bytes. Architecture and supported codes remain unchanged. Adaptation and ticket-evaluation records are not published. See the [model card](model-card.md) for adaptation provenance and [training instructions](training.md) for the public base workflow. The complete v4 adaptation workflow is not available for reproduction.

[Original public-base measurements](evaluation.json) are retained as historical results for SHA-256 `ec2bb56baba69d1b71d7cf288d0723199f169dfff45a3425b5544fc41bb2fe30`; their calibration, metrics, and latency do not describe the new bundled weights. No new latency benchmark is claimed here.
