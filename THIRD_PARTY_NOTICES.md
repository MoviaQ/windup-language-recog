# Third-party notices

The source datasets are downloaded on demand for reproduction. Neither their raw archives nor prepared text records are redistributed in this repository. Their licenses are not replaced by the repository's MIT software license or the project's model contribution license.

## WiLI-2018

**Martin Thoma. WiLI-2018 — Wikipedia Language Identification dataset (2018).**

- [Dataset record and source archive](https://zenodo.org/records/841984)
- [Paper: The WiLI benchmark dataset for written language identification](https://arxiv.org/abs/1801.07779)
- Database license stated by the dataset: **Open Data Commons Open Database License 1.0 (ODbL-1.0)**. [Legal text](https://opendatacommons.org/licenses/odbl/1-0/).

WiLI text originates from Wikipedia. Database licensing does not replace the underlying articles' attribution and share-alike terms. Consult the source records and [Wikimedia Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use) when distributing text or derived databases. Historical text may be subject to CC BY-SA 3.0 and/or GFDL; source-specific terms and attribution still apply. Retain available dataset metadata and source attribution when reproducing or adapting the corpus.

## MASSIVE 1.1

**Amazon. MASSIVE: A 1M-example multilingual natural language understanding dataset with 51 typologically-diverse languages (2022).**

- [Official repository, contributors, paper, and license](https://github.com/alexa/massive)
- [Version 1.1 source archive](https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz)
- Dataset license: **Creative Commons Attribution 4.0 International (CC BY 4.0)**. [Legal text](https://creativecommons.org/licenses/by/4.0/legalcode.en).

The preparation script extracts utterances, maps locale prefixes to supported language codes, cleans text, removes exact normalized duplicate/conflicting examples, and crops long text. Those are project modifications. The official train/dev/test assignments are retained before duplicate exclusion. Multiple locales can map to one output language code.

## Runtime and training dependencies

PyTorch, ftfy, NumPy, and pycountry are installed from their respective distributions and retain their own licenses. They are not vendored or relicensed here. Package metadata and source repositories provide complete notices.

## Project-authored material

`data/demo.csv`, `data/synthetic.csv`, and `assets/logo.png` are project-authored demonstration/synthetic material and logo artwork, distributed under the repository's MIT license. Synthetic evaluation is illustrative and cannot substitute for an independently labeled application benchmark.

## v21 adaptation

The bundled v21 weights combine an adapted MLP and an earlier linear network. They include support-domain adaptation with public replay data and project-authored synthetic support requests. v21 keeps v20's weights and changes only the acceptance calibration. Adaptation records are not redistributed. Assistant-reviewed labels are not independently verified human gold. The model contribution license covers project contributions only and does not grant rights to source records.
