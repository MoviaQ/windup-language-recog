# Model weights license

The released `model.npz` (v22) is a single network trained on the project's multidomain corpus. That corpus is built by `prepare_multidomain_data.py` from MADLAD-400, FineWeb-2, Wikipedia, Tatoeba, MaCoCu, the *WaC corpora, SETimes, OpenLID-v3 (including rows tagged `mt560`), WiLI-2018 and MASSIVE 1.1. The weights are not covered by the software's MIT license. Earlier releases (`model.pt`, v21 and before) remain in git history under the same terms as when they were published.

To the extent copyright or similar rights apply to the model weight contributions made by this project, Contributors license those contributions under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.

- [License summary](https://creativecommons.org/licenses/by-sa/4.0/)
- [Full legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)

You may share and adapt those contributions, including commercially, subject to the license's attribution and share-alike requirements. Credit **Wind-Up Language Recognition Contributors**, link to this repository and the license, and indicate modifications. Do not imply endorsement.

This grant does not relicense upstream datasets or grant rights Contributors do not hold. Preserve the upstream attribution and applicable database/content rights described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). In particular, WiLI carries database terms; Wikipedia-, *WaC- and SETimes-derived text has attribution/share-alike terms; MADLAD-400 and FineWeb-2 are ODC-BY with Common Crawl terms of use. The OpenLID `mt560` rows come from a compilation that includes JW300, whose upstream terms of use are unclear. The project owner accepted that risk for v22. These notices are part of the model's provenance, not a claim that upstream data has been converted to a different license.

Suggested attribution:

> Wind-Up Language Recognition model v22, by Contributors, CC BY-SA 4.0. Trained on a multidomain corpus built from MADLAD-400, FineWeb-2, Wikipedia, Tatoeba, MaCoCu, bs/hr/srWaC, SETimes, OpenLID-v3, WiLI-2018 (Thoma, 2018) and MASSIVE 1.1 (Amazon, 2022). See the project's third-party notices for upstream terms. [Repository](https://github.com/MoviaQ/windup-language-recog).

Software, documentation, project-authored demo/synthetic examples, and the logo remain under MIT. Dependency licenses apply to dependencies independently.
