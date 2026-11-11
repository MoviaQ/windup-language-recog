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

## Multidomain training data (experimental, not in bundled weights)

`prepare_multidomain_data.py` downloads or streams the sources below into `data/prepared_v2/` for an experimental corpus. The bundled `model.pt` was **not** trained on it, and the prepared records are not redistributed. The script pins dataset revisions and writes the URL, the upstream sha256 (when the host publishes one) and the sha256 of the bytes read for every remote file to `data/prepared_v2/manifest.json` (`sources_lock`). Draft: licences below are taken from each source's own card or licence page and should be reviewed before any weights trained on this corpus are distributed.

| Source | What is used | Licence stated by the source |
|---|---|---|
| [MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400) (`clean`, revision `9d886a76`) | prefix of the first clean shard per language | ODC-BY 1.0; Common Crawl terms of use apply to the underlying pages |
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (revision `af9c1333`) | row groups of uzn_Cyrl, srp_Latn, nno_Latn, ckb_Arab, kmr_Latn | ODC-BY 1.0; Common Crawl terms of use apply |
| [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) dump 20231101 (revision `b04c8d1c`) | row groups per language edition | CC BY-SA 4.0 and GFDL; attribution and share-alike apply |
| [Tatoeba](https://tatoeba.org/) per-language exports | sentences | CC BY 2.0 FR (some sentences CC0). Exports change daily; the manifest records the file sha256 and Last-Modified date |
| [MaCoCu](https://macocu.eu/) bs 1.0, hr 2.0, sr 1.0 ([CLARIN.SI 11356/1806–1808](http://hdl.handle.net/11356/1806)) | prefix of the corpus, paragraphs marked quality=good | CC0 1.0 |
| bsWaC 1.1, hrWaC 2.1, srWaC 1.1 ([CLARIN.SI 11356/1062–1064](http://hdl.handle.net/11356/1064)) | prefix of the first file, paragraphs whose language attribute matches | CC BY-SA 4.0 |
| [SETimes](https://opus.nlpl.eu/SETIMES/corpus/version/SETIMES) v2 monolingual bs/hr/sr (OPUS) | prefix | CC BY-SA 3.0 |
| [OpenLID-v3](https://huggingface.co/datasets/HPLT/OpenLID-v3) (revision `f1d299cf`) | only rows whose source tag is mt560, setimes, globalvoices, nllbSeed, mizan, tico, jojajovai, americas-nli, pali, wili2018, tatoeba, commonvoice | per tag: Apache-2.0 (MT560 compilation), CC BY-SA 3.0/4.0, CC BY 3.0/4.0, CC0, MIT, ODbL-1.0, CC BY 2.0 FR |
| WiLI-2018 and MASSIVE 1.1 (see above) | WiLI train paragraphs, MASSIVE train utterances | as above |
| [FLORES-200](https://github.com/facebookresearch/flores) | **filter only**: every dev/devtest line is removed from the corpus; no FLORES text is used for training | CC BY-SA 4.0 |

**Owner-accepted risk: MT560 / JW300.** OpenLID-v3 rows tagged `mt560` come from the MT560 compilation (Apache-2.0 as a compilation), which includes JW300 and other OPUS subsets whose upstream terms are narrower or unclear (JW300 was withdrawn from OPUS over its terms of use). The project owner accepted this risk for the experiment. Remove `mt560` from `OPENLID_ALLOWED_TAGS` before building a corpus for distributed weights unless the risk is accepted again for that release.

Excluded on purpose: non-commercial sources (Leipzig, XL-Sum, TIL, AfriQA, BLTR, EM corpus, the OpenLID `wiki`/HFWikipedia tag), LTI, TEP, MADAR and other Arabic-dialect or custom-licence sources, UD and GlotStoryBook rows (per-row licence not resolvable), sources without a stated licence, and GlotLID-C (contains FLORES dev). The excluded OpenLID tags and their reasons are listed in the manifest.

The web sources were collected by crawlers and filtered by their publishers; they may contain personal data or copyrighted text. MaCoCu and *WaC split Bosnian, Croatian and Serbian by national web domain or by a paragraph classifier, so some rows may carry the label of a closely related language.

## Runtime and training dependencies

PyTorch, ftfy, NumPy, and pycountry are installed from their respective distributions and retain their own licenses. They are not vendored or relicensed here. Package metadata and source repositories provide complete notices.

## Project-authored material

`data/demo.csv`, `data/synthetic.csv`, and `assets/logo.png` are project-authored demonstration/synthetic material and logo artwork, distributed under the repository's MIT license. Synthetic evaluation is illustrative and cannot substitute for an independently labeled application benchmark.

## v21 adaptation

The bundled v21 weights combine an adapted MLP and an earlier linear network. They include support-domain adaptation with public replay data and project-authored synthetic support requests. v21 keeps v20's weights and changes only the acceptance calibration. Adaptation records are not redistributed. Assistant-reviewed labels are not independently verified human gold. The model contribution license covers project contributions only and does not grant rights to source records.
