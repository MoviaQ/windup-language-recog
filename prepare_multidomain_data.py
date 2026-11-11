"""Prepare an experimental multidomain training corpus (data/prepared_v2).

The corpus mixes web, Wikipedia, short-sentence and other (parallel/news)
text for the 100 codes in ``data/languages.json``. It is an experiment: the
bundled ``model.pt`` was not trained on it. Sources and licences are listed in
``THIRD_PARTY_NOTICES.md`` ("Multidomain training data").

Sources (all pinned; see ``HF_REVISIONS`` and the URL tables below):

* web:   MADLAD-400 ``clean`` shards (streamed prefix), FineWeb-2 slices for
         uzn_Cyrl, srp_Latn, nno_Latn, ckb_Arab, kmr_Latn (row groups),
         MaCoCu bs/hr/sr and bsWaC/hrWaC/srWaC (streamed prefix).
* wiki:  wikimedia/wikipedia 20231101 (row groups), WiLI-2018 train paragraphs.
* short: Tatoeba per-language exports, MASSIVE 1.1 train, OpenLID-v3 rows
         tagged tatoeba/commonvoice.
* other: OpenLID-v3 rows with permissive tags plus MT560 (owner-accepted
         risk), OPUS SETimes monolingual bs/hr/sr.

Pipeline per language: sentence split -> ftfy + NFC -> length filter ->
letter-ratio and per-label script filter -> exact dedup (dataset_utils.text_key)
and prefix/suffix near-dedup -> FLORES-200 contamination filter (all dev and
devtest files) -> held-out filter (WiLI test/valid, MASSIVE dev/test, project
synthetic and ticket fixtures). Then, across languages, every key seen under
more than one label is dropped. Each language is capped (``--cap``) with a
40/30/15/15 web/wiki/short/other mix for full segments, and 30% of the cap is
short-span crops (4-16 words; 16-60 characters for unspaced scripts).

Every dropped row is counted by reason in ``manifest.json``; every remote file
read is recorded with its URL, upstream sha256 when the host publishes one, the
bytes read and their sha256 (``sources_lock``).

Usage::

    .venv/bin/python -m pip install -r requirements-training.txt
    .venv/bin/python prepare_multidomain_data.py --cap 50000 --workers 4

Train on it with ``train_model.py --data-dir data/prepared_v2`` (without
``--short-only``: the files are ``train.csv``/``valid.csv``). ``valid.csv`` is
an in-corpus split for early stopping only; it is not a benchmark.
``negatives/nn.csv`` holds Norwegian Nynorsk text, which is not a training
label, for later abstention calibration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import multiprocessing
import os
import random
import re
import shutil
import struct
import tarfile
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Iterator

import ftfy
import numpy as np

from dataset_utils import read_csv, text_key, write_csv

ROOT = Path(__file__).resolve().parent
USER_AGENT = "windup-language-recog-data-prep/0.1"

# --------------------------------------------------------------------------
# Pinned sources
# --------------------------------------------------------------------------

HF_REVISIONS = {
    "madlad": ("allenai/MADLAD-400", "9d886a76bd8fa69b294f2dd3843dacb8388ee5a5"),
    "wikipedia": ("wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"),
    "openlid": ("HPLT/OpenLID-v3", "f1d299cf94df009e25bea88664753fb4dbcad1bc"),
    "fineweb2": ("HuggingFaceFW/fineweb-2", "af9c13333eb981300149d5ca60a8e9d659b276b9"),
}
WIKI_DUMP = "20231101"
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
FLORES_FILES_PER_SPLIT = 204
TATOEBA_URL = "https://downloads.tatoeba.org/exports/per_language/{iso}/{iso}_sentences.tsv.bz2"
OPUS_SETIMES_URL = "https://object.pouta.csc.fi/OPUS-SETIMES/v2/mono/{lang}.txt.gz"
CLARIN = "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/"
MACOCU_URLS = {
    "hr": CLARIN + "1806/MaCoCu-hr-2.0.xml.zip",
    "sr": CLARIN + "1807/MaCoCu-sr-1.0.xml.zip",
    "bs": CLARIN + "1808/MaCoCu-bs-1.0.xml.zip",
}
WAC_URLS = {
    "bs": CLARIN + "1062/bsWaC1.1.01.xml.gz",
    "sr": CLARIN + "1063/srWaC1.1.01.xml.gz",
    "hr": CLARIN + "1064/hrWaC2.1.01.xml.gz",
}
WILI_SHA256 = "727e52ca4e13400e6def1b1b594ca12c8b7fd49ad9d45fd5c9e57a1f6d3b7a3f"
MASSIVE_SHA256 = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"

LICENCES = {
    "madlad": "ODC-BY 1.0 (MADLAD-400); underlying Common Crawl terms apply",
    "fineweb2": "ODC-BY 1.0 (FineWeb-2); underlying Common Crawl terms apply",
    "wikipedia": "CC BY-SA 4.0 / GFDL (Wikipedia text via wikimedia/wikipedia)",
    "wili": "ODbL-1.0 (WiLI-2018); underlying Wikipedia terms apply",
    "tatoeba": "CC BY 2.0 FR (Tatoeba; some sentences CC0)",
    "massive": "CC BY 4.0 (MASSIVE 1.1)",
    "macocu": "CC0 1.0 (MaCoCu monolingual corpora)",
    "wac": "CC BY-SA 4.0 (bsWaC 1.1, hrWaC 2.1, srWaC 1.1)",
    "opus_setimes": "CC BY-SA 3.0 (SETimes via OPUS)",
    "openlid": "Per source tag, see OPENLID_ALLOWED_TAGS; OpenLID-v3 compilation",
    "flores": "CC BY-SA 4.0 (FLORES-200; used only to filter, never as training data)",
}

# OpenLID-v3 source tags. Licences from the dataset's LICENSE.md table.
# Value: (licence, domain).
OPENLID_ALLOWED_TAGS = {
    "mt560": ("Apache-2.0 compilation; contains JW300 and other OPUS subsets (owner-accepted risk)", "other"),
    "setimes": ("CC BY-SA 3.0", "other"),
    "globalvoices": ("CC BY 3.0 (website licence)", "other"),
    "nllbSeed": ("CC BY-SA 4.0", "other"),
    "mizan": ("CC BY 4.0", "other"),
    "tico": ("CC0", "other"),
    "jojajovai": ("MIT", "other"),
    "americas-nli": ("CC BY-SA 4.0", "other"),
    "pali": ("MIT", "other"),
    "wili2018": ("ODbL-1.0", "other"),
    "tatoeba": ("CC BY 2.0 FR", "short"),
    "tatoeba+grn": ("CC BY 2.0 FR", "short"),
    "commonvoice": ("CC0 1.0", "short"),
    "commonvoice+grn": ("CC0 1.0", "short"),
}
# Every other tag is excluded. Named here so the manifest says why.
OPENLID_EXCLUDED_TAGS = {
    "xlsum": "CC BY-NC-SA 4.0 (non-commercial)",
    "leipzig": "Leipzig terms: non-commercial only",
    "leipzigwiki": "Leipzig terms: non-commercial only",
    "til": "CC BY-NC-SA 4.0 (non-commercial)",
    "afriqa": "CC BY-NC 4.0 (non-commercial)",
    "lti": "owner decision: LTI excluded",
    "tep": "owner decision: TEP (GPL) excluded",
    "madar": "owner decision: MADAR custom licence excluded",
    "arabDialects": "owner decision: Arabic-dialect sources without licence excluded",
    "iadd": "mixed GPL/LGPL/custom licences (Arabic dialects) excluded",
    "voa": "no explicit licence",
    "guarani": "no explicit licence",
    "masakhanews": "licence not stated in OpenLID-v3 table",
    "nusa": "licence not stated in OpenLID-v3 table",
    "wiki": "HFWikipedia tag listed with CC BY-NC-SA 4.0; owner decision: excluded",
    "ud": "UD treebank licences vary per treebank (some NC); not resolvable per row",
    "GlotStoryBook": "'CC family' per story (some NC); not resolvable per row",
    "ud+grn": "UD treebank licences vary per treebank",
}

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

NEGATIVE_LABELS = ("nn",)
# Raw source codes that map onto a different label. Applied by resolve_label.
LABEL_ALIASES = {
    "ckb": "ku", "ckb_Arab": "ku", "kmr": "ku", "kmr_Latn": "ku", "kur": "ku",
    "fil": "tl", "fil_Latn": "tl", "tgl": "tl", "tgl_Latn": "tl",
    "nno": "nn", "nno_Latn": "nn", "nn": "nn",
    "no": "nb", "nob": "nb", "nob_Latn": "nb",
    "uzn_Cyrl": "uz", "uzn_Latn": "uz", "uzb": "uz",
    "srp_Latn": "sr", "srp_Cyrl": "sr", "srp": "sr",
    "cmn": "zh", "cmn_Hans": "zh", "cmn_Hant": "zh", "zh-CN": "zh", "zh-TW": "zh",
}
# Raw codes that are dropped outright, with the reason recorded.
DROPPED_CODES = {
    "azb": "owner decision: South Azerbaijani (azb) dropped",
    "azb_Arab": "owner decision: South Azerbaijani (azb) dropped",
    "sh": "Serbo-Croatian Wikipedia cannot be split into bs/hr/sr",
    "hbs": "Serbo-Croatian cannot be split into bs/hr/sr",
    "yue": "owner decision: Cantonese excluded from zh",
    "yue_Hant": "owner decision: Cantonese excluded from zh",
    "zh_Latn": "romanised Chinese is not zh text",
    "lzh": "Classical Chinese is not Standard Chinese",
}


def resolve_label(raw: str, labels: Iterable[str]) -> tuple[str | None, str]:
    """Map a source code to one training/negative label; (None, reason) if dropped."""
    if raw in DROPPED_CODES:
        return None, DROPPED_CODES[raw]
    label = LABEL_ALIASES.get(raw, raw)
    if label in NEGATIVE_LABELS or label in set(labels):
        return label, "ok"
    return None, f"not a supported label: {raw}"


# Per-label source names. Codes not listed use the label itself.
MADLAD_CODES = {"nb": ["no"], "tl": ["fil"], "ku": ["ku", "ckb"], "zh": ["zh"]}
WIKI_CODES = {"nb": ["no"], "ku": ["ku", "ckb"], "nn": ["nn"]}
FINEWEB2_SLICES = {"uz": ["uzn_Cyrl"], "sr": ["srp_Latn"], "ku": ["ckb_Arab", "kmr_Latn"], "nn": ["nno_Latn"]}
OPENLID_FILES = {
    "af": ["afr_Latn"], "am": ["amh_Ethi"], "ar": ["ara_Arab"], "as": ["asm_Beng"],
    "az": ["azj_Latn"], "be": ["bel_Cyrl"], "bg": ["bul_Cyrl"], "bn": ["ben_Beng"],
    "bo": ["bod_Tibt"], "bs": ["bos_Latn"], "ca": ["cat_Latn"], "cs": ["ces_Latn"],
    "cy": ["cym_Latn"], "da": ["dan_Latn"], "de": ["deu_Latn"], "el": ["ell_Grek"],
    "en": ["eng_Latn"], "es": ["spa_Latn"], "et": ["ekk_Latn"], "eu": ["eus_Latn"],
    "fa": ["fas_Arab"], "fi": ["fin_Latn"], "fr": ["fra_Latn"], "ga": ["gle_Latn"],
    "gl": ["glg_Latn"], "gn": ["gug_Latn"], "gu": ["guj_Gujr"], "ha": ["hau_Latn"],
    "he": ["heb_Hebr"], "hi": ["hin_Deva"], "hr": ["hrv_Latn"], "ht": ["hat_Latn"],
    "hu": ["hun_Latn"], "hy": ["hye_Armn"], "id": ["ind_Latn"], "ig": ["ibo_Latn"],
    "is": ["isl_Latn"], "it": ["ita_Latn"], "ja": ["jpn_Jpan"], "jv": ["jav_Latn"],
    "ka": ["kat_Geor"], "kk": ["kaz_Cyrl"], "km": ["khm_Khmr"], "kn": ["kan_Knda"],
    "ko": ["kor_Hang"], "ku": ["kmr_Latn", "ckb_Arab"], "ky": ["kir_Cyrl"],
    "lb": ["ltz_Latn"], "lo": ["lao_Laoo"], "lt": ["lit_Latn"], "lv": ["lvs_Latn"],
    "mg": ["plt_Latn"], "mk": ["mkd_Cyrl"], "ml": ["mal_Mlym"], "mn": ["khk_Cyrl"],
    "mr": ["mar_Deva"], "ms": ["zsm_Latn"], "mt": ["mlt_Latn"], "my": ["mya_Mymr"],
    "nb": ["nob_Latn"], "ne": ["npi_Deva"], "nl": ["nld_Latn"], "om": ["gaz_Latn"],
    "or": ["ory_Orya"], "pa": ["pan_Guru"], "pl": ["pol_Latn"], "ps": ["pbt_Arab"],
    "pt": ["por_Latn"], "qu": ["quy_Latn"], "ro": ["ron_Latn"], "ru": ["rus_Cyrl"],
    "rw": ["kin_Latn"], "sd": ["snd_Arab"], "si": ["sin_Sinh"], "sk": ["slk_Latn"],
    "sl": ["slv_Latn"], "sn": ["sna_Latn"], "so": ["som_Latn"], "sq": ["als_Latn"],
    "sr": ["srp_Cyrl", "srp_Latn"], "su": ["sun_Latn"], "sv": ["swe_Latn"],
    "sw": ["swh_Latn"], "ta": ["tam_Taml"], "te": ["tel_Telu"], "tg": ["tgk_Cyrl"],
    "th": ["tha_Thai"], "tk": ["tuk_Latn"], "tl": ["fil_Latn"], "tn": ["tsn_Latn"],
    "tr": ["tur_Latn"], "tt": ["tat_Cyrl"], "ug": ["uig_Arab"], "uk": ["ukr_Cyrl"],
    "ur": ["urd_Arab"], "uz": ["uzn_Latn"], "vi": ["vie_Latn"], "yi": ["ydd_Hebr"],
    "yo": ["yor_Latn"], "zh": ["cmn_Hans", "cmn_Hant"],
}
TATOEBA_CODES = {
    "af": ["afr"], "am": ["amh"], "ar": ["ara"], "as": ["asm"], "az": ["aze"],
    "be": ["bel"], "bg": ["bul"], "bn": ["ben"], "bo": ["bod"], "bs": ["bos"],
    "ca": ["cat"], "cs": ["ces"], "cy": ["cym"], "da": ["dan"], "de": ["deu"],
    "el": ["ell"], "en": ["eng"], "es": ["spa"], "et": ["est"], "eu": ["eus"],
    "fa": ["pes"], "fi": ["fin"], "fr": ["fra"], "ga": ["gle"], "gl": ["glg"],
    "gn": ["grn"], "gu": ["guj"], "ha": ["hau"], "he": ["heb"], "hi": ["hin"],
    "hr": ["hrv"], "ht": ["hat"], "hu": ["hun"], "hy": ["hye"], "id": ["ind"],
    "ig": ["ibo"], "is": ["isl"], "it": ["ita"], "ja": ["jpn"], "jv": ["jav"],
    "ka": ["kat"], "kk": ["kaz"], "km": ["khm"], "kn": ["kan"], "ko": ["kor"],
    "ku": ["kmr", "ckb"], "ky": ["kir"], "lb": ["ltz"], "lo": ["lao"], "lt": ["lit"],
    "lv": ["lvs"], "mg": ["mlg"], "mk": ["mkd"], "ml": ["mal"], "mn": ["mon"],
    "mr": ["mar"], "ms": ["zsm"], "mt": ["mlt"], "my": ["mya"], "nb": ["nob"],
    "ne": ["nep", "npi"], "nl": ["nld"], "om": ["gaz"], "or": ["ori", "ory"],
    "pa": ["pan"], "pl": ["pol"], "ps": ["pus"], "pt": ["por"], "qu": ["que"],
    "ro": ["ron"], "ru": ["rus"], "rw": ["kin"], "sd": ["snd"], "si": ["sin"],
    "sk": ["slk"], "sl": ["slv"], "sn": ["sna"], "so": ["som"], "sq": ["sqi"],
    "sr": ["srp"], "su": ["sun"], "sv": ["swe"], "sw": ["swh"], "ta": ["tam"],
    "te": ["tel"], "tg": ["tgk"], "th": ["tha"], "tk": ["tuk"], "tl": ["tgl"],
    "tn": ["tsn"], "tr": ["tur"], "tt": ["tat"], "ug": ["uig"], "uk": ["ukr"],
    "ur": ["urd"], "uz": ["uzb"], "vi": ["vie"], "yi": ["yid"], "yo": ["yor"],
    "zh": ["cmn"], "nn": ["nno"],
}
OPUS_SETIMES_LABELS = ("bs", "hr", "sr")
# MaCoCu paragraphs carry their own script-level language id.
MACOCU_PARAGRAPH_LANGS = {"bs": {"hbs_lat"}, "hr": {"hbs_lat"}, "sr": {"hbs_cyr", "hbs_lat"}}

# --------------------------------------------------------------------------
# Scripts
# --------------------------------------------------------------------------

SCRIPT_RANGES = {
    "Latin": "A-Za-zÀ-ÖØ-öø-ɏɐ-ʯḀ-ỿⱠ-Ɀ꜠-ꟿꬰ-꭯Ａ-Ｚａ-ｚ",
    "Cyrillic": "Ѐ-ԯᲀ-᲏ⷠ-ⷿꙀ-ꚟ",
    "Greek": "Ͱ-Ͽἀ-῿",
    "Armenian": "԰-֏ﬓ-ﬗ",
    "Hebrew": "֐-׿יִ-ﭏ",
    "Arabic": "؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿",
    "Devanagari": "ऀ-ॿ꣠-ꣿ",
    "Bengali": "ঀ-৿",
    "Gurmukhi": "਀-੿",
    "Gujarati": "઀-૿",
    "Oriya": "଀-୿",
    "Tamil": "஀-௿",
    "Telugu": "ఀ-౿",
    "Kannada": "ಀ-೿",
    "Malayalam": "ഀ-ൿ",
    "Sinhala": "඀-෿",
    "Thai": "฀-๿",
    "Lao": "຀-໿",
    "Tibetan": "ༀ-࿿",
    "Myanmar": "က-႟ꧠ-꧿ꩠ-ꩿ",
    "Georgian": "Ⴀ-ჿᲐ-Ჿⴀ-⴯",
    "Hangul": "ᄀ-ᇿ㄰-㆏ꥠ-꥿가-힯ힰ-퟿",
    "Ethiopic": "ሀ-᎟ⶀ-⷟꬀-꬯",
    "Khmer": "ក-៿᧠-᧿",
    "Han": "⺀-⿟々〇〡-〩〸-〻㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f",
    "Kana": "぀-ヿㇰ-ㇿｦ-ﾟ",
}
SCRIPT_RE = {name: re.compile(f"[{ranges}]") for name, ranges in SCRIPT_RANGES.items()}
_LATIN = ("Latin",)
_CYRL = ("Cyrillic",)
_ARAB = ("Arabic",)
# Allowed scripts per label: at least half of a text's letters must be in them.
EXPECTED_SCRIPTS: dict[str, tuple[str, ...]] = {
    "am": ("Ethiopic",), "ar": _ARAB, "as": ("Bengali",), "be": _CYRL, "bg": _CYRL,
    "bn": ("Bengali",), "bo": ("Tibetan",), "el": ("Greek",), "fa": _ARAB,
    "gu": ("Gujarati",), "he": ("Hebrew",), "hi": ("Devanagari",), "hy": ("Armenian",),
    "ja": ("Han", "Kana"), "ka": ("Georgian",), "kk": _CYRL, "km": ("Khmer",),
    "kn": ("Kannada",), "ko": ("Hangul", "Han"), "ku": ("Latin", "Arabic"), "ky": _CYRL,
    "lo": ("Lao",), "mk": _CYRL, "ml": ("Malayalam",), "mn": _CYRL, "mr": ("Devanagari",),
    "my": ("Myanmar",), "ne": ("Devanagari",), "or": ("Oriya",), "pa": ("Gurmukhi",),
    "ps": _ARAB, "ru": _CYRL, "sd": _ARAB, "si": ("Sinhala",),
    "sr": ("Cyrillic", "Latin"), "ta": ("Tamil",), "te": ("Telugu",), "tg": _CYRL,
    "th": ("Thai",), "tt": _CYRL, "ug": _ARAB, "uk": _CYRL, "ur": _ARAB,
    "uz": ("Latin", "Cyrillic"), "yi": ("Hebrew",), "zh": ("Han",),
}
# Labels written without spaces between words: crops are character windows.
UNSPACED_LABELS = frozenset({"bo", "ja", "km", "lo", "my", "th", "zh"})


def expected_scripts(label: str) -> tuple[str, ...]:
    return EXPECTED_SCRIPTS.get(label, _LATIN)


def script_check(text: str, label: str, min_share: float = 0.5) -> str | None:
    """None if the text fits the label's scripts, else a drop reason."""
    letters = sum(1 for c in text if c.isalpha())
    if letters == 0:
        return "no_letters"
    in_script = 0
    for name in expected_scripts(label):
        in_script += sum(1 for c in SCRIPT_RE[name].findall(text) if c.isalpha())
    if in_script < min_share * letters:
        return "script"
    if label == "ja" and not SCRIPT_RE["Kana"].search(text):
        return "script"  # Han-only text is not distinguishable from Chinese.
    return None


# --------------------------------------------------------------------------
# Text cleanup, segmentation, keys
# --------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?;।॥۔؟።፧။։។៕།])\s+|(?<=[。！？])|\n+")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_URL = re.compile(r"https?://|www\.|\S+@\S+\.\w")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", ftfy.fix_text(text))
    return " ".join(_CONTROL.sub(" ", text).split())


def split_segments(text: str, max_chars: int) -> list[str]:
    """Sentence-split; chunk segments longer than max_chars at whitespace."""
    out = []
    for segment in _SENTENCE_END.split(text):
        segment = segment.strip()
        if not segment:
            continue
        if len(segment) <= max_chars:
            out.append(segment)
            continue
        words = segment.split()
        if len(words) == 1:
            step = max(16, max_chars // 2)
            out.extend(segment[i : i + step] for i in range(0, len(segment), step))
            continue
        chunk: list[str] = []
        size = 0
        for word in words:
            if chunk and size + len(word) + 1 > max_chars // 2 + 50:
                out.append(" ".join(chunk))
                chunk, size = [], 0
            chunk.append(word)
            size += len(word) + 1
        if chunk:
            out.append(" ".join(chunk))
    return out


def key_hash(key: str) -> int:
    """Stable 63-bit hash of a dataset_utils.text_key (same in every process)."""
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little") >> 1


def near_signatures(key: str, width: int = 40) -> tuple[int, ...]:
    """Prefix/suffix signatures of a normalized key (cheap near-duplicate check).

    Two segments sharing their first or their last ``width`` normalized letters
    are treated as near-duplicates. This catches the dominant web pattern
    (templates differing in a number, a name or a tail) at O(1) per segment;
    MinHash would also catch reordered or substituted middles, at a cost the
    exact-plus-contamination filters do not need for language-id data.
    """
    if len(key) < width:
        return ()
    return (key_hash("<" + key[:width]), key_hash(">" + key[-width:]))


class Filter:
    """Per-segment checks shared by all sources; returns (text, key) or a reason."""

    def __init__(self, min_chars: int, max_chars: int, min_words: int):
        self.min_chars, self.max_chars, self.min_words = min_chars, max_chars, min_words

    def __call__(self, raw: str, label: str, need_words: bool) -> tuple[str, str] | str:
        text = normalize_text(raw)
        if len(text) < self.min_chars:
            return "too_short"
        if len(text) > self.max_chars:
            return "too_long"
        if _URL.search(text):
            return "url_or_email"
        non_space = sum(1 for c in text if not c.isspace())
        letters = sum(1 for c in text if c.isalpha())
        if letters < 0.5 * non_space:
            return "too_few_letters"
        reason = script_check(text, label)
        if reason:
            return reason
        if need_words and label not in UNSPACED_LABELS and len(text.split()) < self.min_words:
            return "too_few_words"
        key = text_key(text)
        if not key:
            return "empty_key"
        return text, key


# --------------------------------------------------------------------------
# Contamination index
# --------------------------------------------------------------------------


class ShingleIndex:
    """Exact keys plus 8-word and 30-character shingles of reference texts.

    ``check`` drops a text whose normalized key equals a reference key, or of
    whose own shingles at least ``threshold`` are reference shingles: 8-word
    shingles when the key has at least 8 words, else 30-character shingles
    when it has at least 30 characters. Shorter texts get the exact check only.
    Shingle hashes use Python's ``hash``: the index must be queried in the
    process that built it or in forked children.
    """

    def __init__(self, texts: Iterable[str], words: int = 8, chars: int = 30, threshold: float = 0.5):
        self.words, self.chars, self.threshold = words, chars, threshold
        exact, word_parts, char_parts = [], [], []
        buffer_w: list[int] = []
        buffer_c: list[int] = []
        self.size = 0
        for text in texts:
            key = text_key(text)
            if not key:
                continue
            self.size += 1
            exact.append(key_hash(key))
            buffer_w.extend(self._word_shingles(key))
            buffer_c.extend(self._char_shingles(key))
            if len(buffer_c) > 2_000_000:
                word_parts.append(np.unique(np.array(buffer_w, dtype=np.int64)))
                char_parts.append(np.unique(np.array(buffer_c, dtype=np.int64)))
                buffer_w, buffer_c = [], []
        word_parts.append(np.array(buffer_w, dtype=np.int64))
        char_parts.append(np.array(buffer_c, dtype=np.int64))
        self.exact = np.unique(np.array(exact, dtype=np.int64))
        self.word = np.unique(np.concatenate(word_parts))
        self.char = np.unique(np.concatenate(char_parts))

    def _word_shingles(self, key: str) -> list[int]:
        tokens = key.split()
        n = self.words
        return [hash(" ".join(tokens[i : i + n])) for i in range(len(tokens) - n + 1)]

    def _char_shingles(self, key: str) -> list[int]:
        n = self.chars
        return [hash(key[i : i + n]) for i in range(len(key) - n + 1)]

    @staticmethod
    def _contains(table: np.ndarray, values: list[int]) -> np.ndarray:
        query = np.array(values, dtype=np.int64)
        if not len(table):
            return np.zeros(len(query), dtype=bool)
        position = np.minimum(np.searchsorted(table, query), len(table) - 1)
        return table[position] == query

    def check(self, key: str) -> str | None:
        """None when clean, 'exact' or 'shingle' when the key is contaminated."""
        if self._contains(self.exact, [key_hash(key)])[0]:
            return "exact"
        if len(key.split()) >= self.words:
            shingles, table = self._word_shingles(key), self.word
        elif len(key) >= self.chars:
            shingles, table = self._char_shingles(key), self.char
        else:
            return None
        if self._contains(table, shingles).mean() >= self.threshold:
            return "shingle"
        return None


def flores_lines(flores_dir: Path, expected_per_split: int = FLORES_FILES_PER_SPLIT) -> Iterator[str]:
    """Every line of every FLORES-200 dev and devtest file; refuses partial copies."""
    for split in ("dev", "devtest"):
        files = sorted((flores_dir / split).glob(f"*.{split}"))
        if len(files) != expected_per_split:
            raise ValueError(
                f"{flores_dir / split}: expected {expected_per_split} FLORES files, found {len(files)}"
            )
        for path in files:
            yield from path.read_text(encoding="utf-8").splitlines()


# --------------------------------------------------------------------------
# Crops and cross-label rule
# --------------------------------------------------------------------------


def make_crop(text: str, label: str, rng: random.Random) -> str | None:
    """A strictly shorter span: 4-16 words, or 16-60 characters for unspaced labels."""
    if label in UNSPACED_LABELS:
        compact = text.strip()
        if len(compact) < 17:
            return None
        length = rng.randint(16, min(60, len(compact) - 1))
        start = rng.randrange(0, len(compact) - length + 1)
        return compact[start : start + length].strip() or None
    words = text.split()
    if len(words) < 5:
        return None
    length = rng.randint(4, min(16, len(words) - 1))
    start = rng.randrange(0, len(words) - length + 1)
    return " ".join(words[start : start + length])


def cross_label_conflicts(hashes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Sorted key hashes that occur under more than one label."""
    if not len(hashes):
        return np.array([], dtype=np.int64)
    pairs = np.unique(np.stack([hashes.astype(np.int64), labels.astype(np.int64)], axis=1), axis=0)
    values, counts = np.unique(pairs[:, 0], return_counts=True)
    return values[counts > 1]


def allocate(mix: dict[str, float], available: dict[str, int], total: int) -> dict[str, int]:
    """Split ``total`` over domains by ``mix``; deficits move to domains with spare rows."""
    alloc = {d: 0 for d in mix}
    remaining = total
    open_domains = {d for d in mix if available.get(d, 0) > 0}
    while remaining > 0 and open_domains:
        weight = sum(mix[d] for d in open_domains)
        given = 0
        for d in sorted(open_domains):
            share = max(1, int(round(remaining * mix[d] / weight)))
            take = min(share, available[d] - alloc[d], remaining - given)
            alloc[d] += take
            given += take
        remaining -= given
        open_domains = {d for d in open_domains if available[d] > alloc[d]}
        if given == 0:
            break
    return alloc


# --------------------------------------------------------------------------
# HTTP and lock records
# --------------------------------------------------------------------------


class Lock:
    """Collects one record per remote file read."""

    def __init__(self):
        self.records: list[dict] = []

    def add(self, **record):
        self.records.append(record)


def _request(url: str, start: int = 0, end: int | None = None, timeout: int = 180):
    headers = {"User-Agent": USER_AGENT}
    if start or end is not None:
        headers["Range"] = f"bytes={start}-" + ("" if end is None else str(end))
    delay = 5
    for attempt in range(6):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)
        except urllib.error.HTTPError as error:
            if error.code in (401, 403, 404, 410):
                raise RuntimeError(f"HTTP {error.code} for {url} (blocked or missing)") from error
            retry_after = error.headers.get("Retry-After") if error.headers else None
            wait = int(retry_after) if retry_after and retry_after.isdigit() else delay
            print(f"[http] {error.code} {url}; retry {attempt + 1} in {wait}s", flush=True)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            wait = delay
            print(f"[http] {error!r} {url}; retry {attempt + 1} in {wait}s", flush=True)
        time.sleep(wait)
        delay = min(delay * 2, 120)
    raise RuntimeError(f"giving up on {url} after 6 attempts")


def http_json(url: str, data: dict | None = None):
    headers = {"User-Agent": USER_AGENT}
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=headers), timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as error:
            if error.code in (401, 403, 404):
                raise RuntimeError(f"HTTP {error.code} for {url}") from error
            print(f"[http] {error.code} {url}; retry {attempt + 1}", flush=True)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            print(f"[http] {error!r} {url}; retry {attempt + 1}", flush=True)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}")


class HashedStream(io.RawIOBase):
    """Sequential HTTP body reader that resumes with Range and hashes every byte."""

    def __init__(self, url: str):
        self.url = url
        self.digest = hashlib.sha256()
        self.offset = 0
        self.response = _request(url)
        self.total = self.response.headers.get("Content-Length")

    def readable(self):
        return True

    def readinto(self, buffer) -> int:
        for attempt in range(6):
            try:
                data = self.response.read(len(buffer))
                break
            except (OSError, TimeoutError, ConnectionError) as error:
                print(f"[http] resume {self.url} at {self.offset}: {error!r}", flush=True)
                time.sleep(5 * (attempt + 1))
                self.response = _request(self.url, start=self.offset)
        else:
            raise RuntimeError(f"stream {self.url} failed at {self.offset}")
        buffer[: len(data)] = data
        self.digest.update(data)
        self.offset += len(data)
        return len(data)

    def close(self):
        try:
            self.response.close()
        finally:
            super().close()


class RangeFile(io.RawIOBase):
    """Seekable HTTP file for pyarrow; records the ranges fetched and their sha256."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.position = url, size, 0
        self.ranges: list[tuple[int, int, str]] = []
        self.resolved = url

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.position, 2: self.size}[whence]
        self.position = base + offset
        return self.position

    def readinto(self, buffer) -> int:
        count = min(len(buffer), self.size - self.position)
        if count <= 0:
            return 0
        for attempt in range(3):
            try:
                with _request(self.resolved, start=self.position, end=self.position + count - 1) as r:
                    self.resolved = r.geturl()
                    data = r.read()
                break
            except RuntimeError:
                if self.resolved == self.url or attempt == 2:
                    raise
                self.resolved = self.url  # signed CDN URL expired: resolve again
        if len(data) != count:
            raise RuntimeError(f"short range read {self.url}: {len(data)} != {count}")
        buffer[:count] = data
        self.ranges.append((self.position, count, hashlib.sha256(data).hexdigest()))
        self.position += count
        return count

    def ranges_digest(self) -> str:
        lines = "".join(f"{o}:{n}:{h}\n" for o, n, h in self.ranges)
        return hashlib.sha256(lines.encode()).hexdigest()

    def bytes_read(self) -> int:
        return sum(n for _, n, _ in self.ranges)


def hf_url(source: str, path: str) -> str:
    repo, revision = HF_REVISIONS[source]
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


def hf_paths_info(source: str, paths: list[str]) -> dict[str, dict]:
    repo, revision = HF_REVISIONS[source]
    info = http_json(f"https://huggingface.co/api/datasets/{repo}/paths-info/{revision}", {"paths": paths})
    return {entry["path"]: entry for entry in info}


def hf_tree(source: str, path: str) -> list[dict]:
    repo, revision = HF_REVISIONS[source]
    try:
        return http_json(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{path}")
    except RuntimeError as error:
        if "HTTP 404" in str(error):
            return []
        raise


# --------------------------------------------------------------------------
# Source readers. Each yields documents (lists of raw strings) and records
# what it read into ``lock`` in a finally block.
# --------------------------------------------------------------------------


def iter_gzip_lines(stream: io.RawIOBase, stats: Counter) -> Iterator[str]:
    decoder = zlib.decompressobj(zlib.MAX_WBITS | 16)
    pending = b""
    while True:
        chunk = stream.read(1 << 20)
        if not chunk:
            break
        data = decoder.decompress(chunk)
        while decoder.eof and decoder.unused_data:  # concatenated gzip members
            rest = decoder.unused_data
            decoder = zlib.decompressobj(zlib.MAX_WBITS | 16)
            data += decoder.decompress(rest)
        pending += data
        *lines, pending = pending.split(b"\n")
        for line in lines:
            try:
                yield line.decode("utf-8")
            except UnicodeDecodeError:
                stats["decode_error"] += 1
    if pending:
        try:
            yield pending.decode("utf-8")
        except UnicodeDecodeError:
            stats["decode_error"] += 1


def iter_zip_first_member_lines(raw: io.RawIOBase, stats: Counter) -> Iterator[str]:
    stream = io.BufferedReader(raw, buffer_size=1 << 20)
    header = stream.read(30)
    signature, _, flags, method, *_rest, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", header)
    if signature != 0x04034B50 or method != 8:
        raise RuntimeError(f"unsupported zip stream (signature {signature:#x}, method {method})")
    stream.read(name_len + extra_len)
    decoder = zlib.decompressobj(-15)
    pending = b""
    while not decoder.eof:
        chunk = stream.read(1 << 20)
        if not chunk:
            break
        pending += decoder.decompress(chunk)
        *lines, pending = pending.split(b"\n")
        for line in lines:
            try:
                yield line.decode("utf-8")
            except UnicodeDecodeError:
                stats["decode_error"] += 1


def stream_record(lock: Lock, source: str, stream: HashedStream, upstream_sha256: str | None = None):
    total = int(stream.total) if stream.total else None
    complete = total is not None and stream.offset >= total
    record = {
        "source": source,
        "url": stream.url,
        "size": total,
        "bytes_read": stream.offset,
        "sha256_of_bytes_read": stream.digest.hexdigest(),
        "complete": complete,
    }
    if upstream_sha256:
        record["upstream_sha256"] = upstream_sha256
        if complete and stream.digest.hexdigest() != upstream_sha256:
            raise RuntimeError(f"sha256 mismatch for {stream.url}")
        record["verified"] = complete
    lock.add(**record)


def madlad_docs(code: str, lock: Lock, stats: Counter) -> Iterator[list[str]]:
    entries = [e for e in hf_tree("madlad", f"data/{code}") if "_clean_" in e["path"]]
    if not entries:
        stats[f"source_missing:madlad:{code}"] += 1
        return
    for entry in sorted(entries, key=lambda e: e["path"]):
        stream = HashedStream(hf_url("madlad", entry["path"]))
        try:
            for line in iter_gzip_lines(stream, stats):
                if line.strip():
                    yield json.loads(line)["text"].replace("\\n", "\n").split("\n")
        finally:
            stream_record(lock, f"madlad:{entry['path']}", stream, entry.get("lfs", {}).get("oid"))
            stream.close()


def parquet_docs(source: str, files: list[dict], column: str, lock: Lock, rng: random.Random,
                 stats: Counter) -> Iterator[list[str]]:
    """Documents from row groups visited in a seeded random order across files."""
    import pyarrow.parquet as pq

    handles = []
    order = []
    try:
        for index, entry in enumerate(files):
            raw = RangeFile(hf_url(source, entry["path"]), entry["size"])
            reader = pq.ParquetFile(io.BufferedReader(raw, buffer_size=1 << 20), pre_buffer=False)
            handles.append((entry, raw, reader, set()))
            order.extend((index, group) for group in range(reader.metadata.num_row_groups))
        rng.shuffle(order)
        for index, group in order:
            entry, raw, reader, used = handles[index]
            used.add(group)
            texts = reader.read_row_group(group, columns=[column]).column(column).to_pylist()
            rng.shuffle(texts)
            for text in texts:
                if text:
                    yield text.split("\n")
    finally:
        for entry, raw, reader, used in handles:
            lock.add(
                source=f"{source}:{entry['path']}",
                url=hf_url(source, entry["path"]),
                size=entry["size"],
                upstream_sha256=entry.get("lfs", {}).get("oid"),
                row_groups_read=sorted(used),
                row_groups_total=reader.metadata.num_row_groups,
                bytes_read=raw.bytes_read(),
                sha256_of_ranges=raw.ranges_digest(),
                complete=False,
            )


def wikipedia_docs(code: str, lock: Lock, rng: random.Random, stats: Counter) -> Iterator[list[str]]:
    files = [e for e in hf_tree("wikipedia", f"{WIKI_DUMP}.{code}") if e["path"].endswith(".parquet")]
    if not files:
        stats[f"source_missing:wikipedia:{code}"] += 1
        return
    yield from parquet_docs("wikipedia", files, "text", lock, rng, stats)


def fineweb2_docs(name: str, lock: Lock, rng: random.Random, stats: Counter) -> Iterator[list[str]]:
    files = [e for e in hf_tree("fineweb2", f"data/{name}/train") if e["path"].endswith(".parquet")]
    if not files:
        stats[f"source_missing:fineweb2:{name}"] += 1
        return
    yield from parquet_docs("fineweb2", files, "text", lock, rng, stats)


def openlid_rows(name: str, quota: dict[str, int], raw_dir: Path, lock: Lock, seed: int,
                 stats: Counter) -> dict[str, list[tuple[str, str]]]:
    """Uniform sample (bottom-k on seeded random keys) of permitted rows by domain.

    The ``source`` column is probed with range reads first; the file is
    downloaded and sha256-verified only when it holds permitted rows.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    path = f"data/{name}.parquet"
    info = hf_paths_info("openlid", [path]).get(path)
    if not info:
        stats[f"source_missing:openlid:{name}"] += 1
        return {}
    url, size, sha = hf_url("openlid", path), info["size"], info["lfs"]["oid"]
    probe = RangeFile(url, size)
    tags = Counter(
        pq.ParquetFile(io.BufferedReader(probe, 1 << 20)).read(columns=["source"]).column("source").to_pylist()
    )
    for tag, count in tags.items():
        if tag not in OPENLID_ALLOWED_TAGS:
            if tag not in OPENLID_EXCLUDED_TAGS:
                raise RuntimeError(f"OpenLID-v3 tag {tag!r} in {name} has no licence decision")
            stats[f"openlid_excluded_tag:{tag}"] += count
    allowed = [tag for tag in tags if tag in OPENLID_ALLOWED_TAGS]
    record = dict(source=f"openlid:{path}", url=url, size=size, upstream_sha256=sha,
                  source_column_bytes=probe.bytes_read(), source_column_ranges_sha256=probe.ranges_digest(),
                  tags=dict(tags))
    if not allowed:
        lock.add(**record, bytes_read=probe.bytes_read(), complete=False, note="no permitted tags; text not read")
        return {}
    temporary = raw_dir / "tmp" / f"openlid_{name}_{os.getpid()}.parquet"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = HashedStream(url)
        with temporary.open("wb") as handle:
            while chunk := stream.read(1 << 22):
                handle.write(chunk)
        stream.close()
        digest = stream.digest.hexdigest()
        if digest != sha:
            raise RuntimeError(f"sha256 mismatch for {url}")
        lock.add(**record, bytes_read=stream.offset, sha256_of_bytes_read=digest, complete=True, verified=True)
        generator = np.random.default_rng(seed)
        kept: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
        for batch in pq.ParquetFile(temporary).iter_batches(batch_size=65536, columns=["text", "source"]):
            sources = batch.column(1)
            mask = pc.is_in(sources, value_set=pa.array(allowed))
            selected = batch.filter(mask)
            if not selected.num_rows:
                continue
            weights = generator.random(selected.num_rows)
            for weight, text, tag in zip(weights, selected.column(0).to_pylist(), selected.column(1).to_pylist()):
                domain = OPENLID_ALLOWED_TAGS[tag][1]
                kept[domain].append((weight, text, tag))
            for domain in list(kept):
                limit = quota.get(domain, 0)
                if len(kept[domain]) > 4 * limit + 100_000:
                    kept[domain].sort()
                    del kept[domain][limit:]
        result = {}
        for domain, rows in kept.items():
            rows.sort()
            result[domain] = [(text, tag) for _, text, tag in rows[: quota.get(domain, 0)]]
        return result
    finally:
        temporary.unlink(missing_ok=True)


def tatoeba_docs(iso: str, lock: Lock, stats: Counter) -> Iterator[list[str]]:
    import bz2

    url = TATOEBA_URL.format(iso=iso)
    try:
        stream = HashedStream(url)
    except RuntimeError as error:
        if "HTTP 404" in str(error):
            stats[f"source_missing:tatoeba:{iso}"] += 1
            return
        raise
    data = stream.readall()
    last_modified = stream.response.headers.get("Last-Modified")
    stream_record(lock, f"tatoeba:{iso}", stream)
    lock.records[-1]["last_modified"] = last_modified
    stream.close()
    lines = bz2.decompress(data).decode("utf-8").splitlines()
    rng = random.Random(f"tatoeba:{iso}")
    rng.shuffle(lines)
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 3:
            stats["malformed_row"] += 1
            continue
        yield [parts[2]]


def gzip_line_docs(source: str, url: str, lock: Lock, stats: Counter) -> Iterator[list[str]]:
    stream = HashedStream(url)
    try:
        for line in iter_gzip_lines(stream, stats):
            yield [line]
    finally:
        stream_record(lock, source, stream)
        stream.close()


_P_ATTRS = re.compile(r'(\w+)="([^"]*)"')


def macocu_docs(label: str, lock: Lock, stats: Counter) -> Iterator[list[str]]:
    url = MACOCU_URLS[label]
    stream = HashedStream(url)
    allowed = MACOCU_PARAGRAPH_LANGS[label]
    try:
        document: list[str] = []
        keep = False
        for line in iter_zip_first_member_lines(stream, stats):
            if line.startswith("<doc "):
                if document:
                    yield document
                document = []
            elif line.startswith("<p "):
                attrs = dict(_P_ATTRS.findall(line))
                keep = attrs.get("quality") == "good" and attrs.get("lang") in allowed
                if not keep:
                    stats["macocu_paragraph_skipped"] += 1
            elif line.startswith("</p>"):
                keep = False
            elif keep and not line.startswith("<"):
                document.append(line)
        if document:
            yield document
    finally:
        stream_record(lock, f"macocu:{label}", stream)
        stream.close()


def wac_docs(label: str, lock: Lock, stats: Counter) -> Iterator[list[str]]:
    url = WAC_URLS[label]
    stream = HashedStream(url)
    try:
        sentences: list[str] = []
        tokens: list[str] = []
        keep = False
        glue = False
        for line in iter_gzip_lines(stream, stats):
            if line.startswith("<p "):
                keep = dict(_P_ATTRS.findall(line)).get("lang") == label
                if not keep:
                    stats["wac_paragraph_other_lang"] += 1
                sentences = []
            elif line.startswith("</p>"):
                if keep and sentences:
                    yield sentences
                keep = False
            elif line.startswith("<s>"):
                tokens, glue = [], False
            elif line.startswith("</s>"):
                if keep and tokens:
                    sentences.append("".join(tokens).strip())
            elif line.startswith("<g/>"):
                glue = True
            elif keep and line and not line.startswith("<"):
                token = line.split("\t", 1)[0]
                tokens.append(token if glue or not tokens else " " + token)
                glue = False
    finally:
        stream_record(lock, f"wac:{label}", stream)
        stream.close()


# --------------------------------------------------------------------------
# WiLI and MASSIVE (shared with prepare_public_data.py downloads)
# --------------------------------------------------------------------------


def load_public(raw_dir: Path, labels: list[str]) -> tuple[dict, dict, list[str], list[dict]]:
    """WiLI train paragraphs and MASSIVE train utterances by label, plus held-out texts.

    Held out: WiLI test, the 50 WiLI paragraphs per language that
    prepare_public_data.py assigns to validation, MASSIVE dev and test.
    """
    import pycountry
    import prepare_public_data as public

    public.RAW = raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    wili = public.download("wili-2018.zip")
    massive = public.download("massive-1.1.tar.gz")
    lock = []
    for path, expected in ((wili, WILI_SHA256), (massive, MASSIVE_SHA256)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"checksum mismatch for {path}")
        lock.append({"source": path.name, "url": public.SOURCES[path.name], "sha256": digest,
                     "size": path.stat().st_size, "complete": True, "verified": True})
    wiki_train: dict[str, list[str]] = defaultdict(list)
    short_train: dict[str, list[str]] = defaultdict(list)
    heldout: list[str] = []
    with zipfile.ZipFile(wili) as archive:
        table = list(csv.DictReader(io.StringIO(archive.read("labels.csv").decode()), delimiter=";"))
        mapping = {}
        for row in table:
            iso = row["ISO 369-3"].strip()
            language = pycountry.languages.get(alpha_3=iso) if len(iso) == 3 else None
            mapping[row["Label"]] = getattr(language, "alpha_2", None) or iso
        for split in ("train", "test"):
            texts = archive.read(f"x_{split}.txt").decode().splitlines()
            ys = archive.read(f"y_{split}.txt").decode().splitlines()
            grouped = defaultdict(list)
            for text, y in zip(texts, ys):
                grouped[mapping[y]].append(text[:2000])
            for code, samples in sorted(grouped.items()):
                label, _ = resolve_label(code, labels)
                if label is None:
                    continue
                if split == "test":
                    heldout.extend(samples)
                    continue
                # Same per-code order and seed as prepare_public_data.py.
                rng = random.Random(42)
                rng.shuffle(samples)
                heldout.extend(samples[:50])
                wiki_train[label].extend(samples[50:])
    with tarfile.open(massive) as archive:
        for member in archive:
            if not member.name.endswith(".jsonl"):
                continue
            code = member.name.rsplit("/", 1)[-1].split("-")[0]
            label, _ = resolve_label(code, labels)
            if label is None:
                continue
            for line in archive.extractfile(member):
                entry = json.loads(line)
                if entry["partition"] == "train":
                    short_train[label].append(entry["utt"])
                else:
                    heldout.append(entry["utt"])
    return dict(wiki_train), dict(short_train), heldout, lock


def project_heldout_texts() -> list[str]:
    texts = [row["text"] for row in read_csv(ROOT / "data/synthetic.csv")]
    fixture = ROOT / "tests/fixtures/authored_tickets.jsonl"
    if fixture.exists():
        for line in fixture.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                texts.extend(t for t in (entry.get("title"), entry.get("description")) if t)
    return texts


# --------------------------------------------------------------------------
# Per-language collection (phase A) and selection (phase C)
# --------------------------------------------------------------------------

DOMAINS = ("web", "wiki", "short", "other")
COLLECT_ORDER = ("short", "other", "wiki", "web")  # scarcest first; web absorbs deficits

COUNTER_LEGEND = {
    "too_short/too_long/url_or_email/too_few_letters/script/no_letters/too_few_words/empty_key":
        "segment dropped by the per-segment filter",
    "duplicate_exact": "same text_key already kept for this label",
    "duplicate_near": "same first or last 40 key characters already kept for this label",
    "flores_exact/flores_shingle": "dropped: matches FLORES-200 dev/devtest",
    "heldout_exact/heldout_shingle": "dropped: matches a held-out evaluation set",
    "cross_label": "dropped: key seen as a candidate under another label",
    "cross_label_final:<kind>": "dropped after crops: key also present under another label",
    "decode_error/malformed_row": "unreadable source line",
    "crop_*": "crop candidate rejected (reason after the prefix) or crop shortfall",
    "crop_shortfall_filled_with_sentences:<split>": "rows filled with spare segments because crops ran out",
    "below_target:<split>": "rows missing to reach the cap",
    "spare_unused:<split>": "selected spare rows not needed (bookkeeping, not a quality drop)",
    "collect_deficit:<domain>": "candidates missing for a domain; carried to the next domain",
    "openlid_excluded_tag:<tag>": "OpenLID-v3 rows with an excluded licence tag (whole file)",
    "macocu_paragraph_skipped": "MaCoCu paragraph not quality=good or not in the label's script id",
    "wac_paragraph_other_lang": "*WaC paragraph whose language attribute is another label",
    "source_missing:<source>": "source has no data for this label",
    "source_error:<source>": "source failed (blocked or network); see source_errors",
}

# Set in the parent before forking workers.
CONTEXT: dict = {}


def source_plan(label: str, ctx: dict) -> dict[str, list[tuple[str, Callable]]]:
    """Sources per domain for one label as (name, factory) pairs."""
    lock, stats, seed, raw_dir = ctx["lock"], ctx["stats"], ctx["seed"], ctx["raw_dir"]
    plan: dict[str, list] = {d: [] for d in DOMAINS}
    for code in MADLAD_CODES.get(label, [label] if label not in NEGATIVE_LABELS else []):
        plan["web"].append((f"madlad:{code}", lambda code=code: madlad_docs(code, lock, stats)))
    for name in FINEWEB2_SLICES.get(label, []):
        plan["web"].append((f"fineweb2:{name}", lambda name=name: fineweb2_docs(
            name, lock, random.Random(f"{seed}:fw2:{name}"), stats)))
    if label in MACOCU_URLS:
        plan["web"].append((f"macocu:{label}", lambda: macocu_docs(label, lock, stats)))
    if label in WAC_URLS:
        plan["web"].append((f"wac:{label}", lambda: wac_docs(label, lock, stats)))
    for code in WIKI_CODES.get(label, [label]):
        plan["wiki"].append((f"wikipedia:{code}", lambda code=code: wikipedia_docs(
            code, lock, random.Random(f"{seed}:wiki:{code}"), stats)))
    if ctx["public_wiki"].get(label):
        plan["wiki"].append(("wili2018", lambda: iter([[p] for p in ctx["public_wiki"][label]])))
    for iso in TATOEBA_CODES.get(label, []):
        plan["short"].append((f"tatoeba:{iso}", lambda iso=iso: tatoeba_docs(iso, lock, stats)))
    if ctx["public_short"].get(label):
        plan["short"].append(("massive", lambda: iter([[u] for u in ctx["public_short"][label]])))
    if label in OPUS_SETIMES_LABELS:
        plan["other"].append((f"opus_setimes:{label}", lambda: gzip_line_docs(
            f"opus_setimes:{label}", OPUS_SETIMES_URL.format(lang=label), lock, stats)))
    return plan


def collect_language(label: str) -> dict:
    """Phase A: candidates for one label, written to work/cand/<label>.tsv."""
    ctx = CONTEXT
    started = time.time()
    lock, stats = Lock(), Counter()
    seed = ctx["seed"]
    ctx = {**ctx, "lock": lock, "stats": stats}
    total = ctx["cap"] + (0 if label in NEGATIVE_LABELS else ctx["valid"])
    if label in NEGATIVE_LABELS:
        total = ctx["negatives_cap"]
    mix = ctx["mix"]
    targets = {d: int(math.ceil(total * mix[d] * ctx["oversample"])) for d in DOMAINS}
    check = Filter(ctx["min_chars"], ctx["max_chars"], ctx["min_words"])
    flores, heldout = ctx["flores"], ctx["heldout"]
    seen: set[int] = set()
    near: set[int] = set()
    rows: list[tuple[str, str, str]] = []
    got = Counter()
    by_source = Counter()
    plan = source_plan(label, ctx)
    # OpenLID rows feed both "short" and "other"; sample them once up front.
    errors: list[str] = []
    openlid: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name in OPENLID_FILES.get(label, []):
        try:
            sampled = openlid_rows(name, {d: targets[d] * 2 for d in ("short", "other")},
                                   ctx["raw_dir"], lock, int(key_hash(f"{seed}:{name}") % 2**32), stats)
        except RuntimeError as error:  # blocked/failed download: recorded, reported, not substituted
            errors.append(f"openlid:{name}: {error}")
            stats[f"source_error:openlid:{name}"] += 1
            print(f"[error] {label} openlid:{name}: {error}", flush=True)
            continue
        for domain, items in sampled.items():
            openlid[domain].extend(items)
    for domain in ("short", "other"):
        if openlid.get(domain):
            items = openlid[domain]
            plan[domain].append(("openlid", lambda items=items: iter([[t] for t, _ in items])))
    carry = 0
    for position, domain in enumerate(COLLECT_ORDER):
        later = COLLECT_ORDER[position + 1 :]
        need = targets[domain] + carry
        specs = plan[domain]
        rng = random.Random(f"{seed}:{label}:{domain}")
        for index, (name, factory) in enumerate(specs):
            quota = int(math.ceil((need - got[domain]) / (len(specs) - index)))
            if quota <= 0:
                break
            taken = 0
            per_doc = ctx["per_doc"] if domain in ("web", "wiki") else 1_000_000
            documents = None
            try:
                documents = factory()
                for document in documents:
                    segments = []
                    for part in document:
                        segments.extend(split_segments(part, ctx["max_chars"]))
                    rng.shuffle(segments)
                    accepted = 0
                    for segment in segments:
                        if accepted >= per_doc or taken >= quota:
                            break
                        result = check(segment, label, need_words=domain in ("web", "wiki"))
                        if isinstance(result, str):
                            stats[result] += 1
                            continue
                        text, key = result
                        hashed = key_hash(key)
                        if hashed in seen:
                            stats["duplicate_exact"] += 1
                            continue
                        signatures = near_signatures(key)
                        if any(s in near for s in signatures):
                            stats["duplicate_near"] += 1
                            seen.add(hashed)
                            continue
                        verdict = flores.check(key)
                        if verdict:
                            stats[f"flores_{verdict}"] += 1
                            seen.add(hashed)
                            continue
                        verdict = heldout.check(key) if heldout else None
                        if verdict:
                            stats[f"heldout_{verdict}"] += 1
                            seen.add(hashed)
                            continue
                        seen.add(hashed)
                        near.update(signatures)
                        rows.append((domain, name, text))
                        accepted += 1
                        taken += 1
                    if taken >= quota:
                        break
            except RuntimeError as error:  # blocked/failed source: recorded, reported, not substituted
                errors.append(f"{name}: {error}")
                stats[f"source_error:{name}"] += 1
                print(f"[error] {label} {name}: {error}", flush=True)
            finally:
                close = getattr(documents, "close", None)
                if close:
                    close()
            got[domain] += taken
            by_source[name] += taken
        deficit = max(0, need - got[domain])
        if deficit:
            stats[f"collect_deficit:{domain}"] += deficit
        carry = deficit if later else 0
    work = ctx["work"] / "cand"
    work.mkdir(parents=True, exist_ok=True)
    with (work / f"{label}.tsv").open("w", encoding="utf-8") as handle:
        for domain, name, text in rows:
            handle.write(f"{domain}\t{name}\t{text}\n")
    hashes = np.array([key_hash(text_key(text)) for _, _, text in rows], dtype=np.int64)
    np.save(work / f"{label}.npy", hashes)
    print(f"[collect] {label}: {len(rows)} candidates {dict(got)} in {time.time() - started:.0f}s", flush=True)
    return {"label": label, "candidates": dict(got), "by_source": dict(by_source),
            "drops": dict(stats), "lock": lock.records, "errors": errors,
            "seconds": round(time.time() - started, 1)}


def select_language(label: str) -> dict:
    """Phase C: drop cross-label keys, pick segments by domain mix, add crops."""
    ctx = CONTEXT
    seed = ctx["seed"]
    conflicts = ctx["conflicts"]
    stats = Counter()
    rng = random.Random(f"{seed}:select:{label}")
    pools: dict[str, list[tuple[str, str, str, int]]] = defaultdict(list)
    path = ctx["work"] / "cand" / f"{label}.tsv"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            domain, name, text = line.rstrip("\n").split("\t", 2)
            hashed = key_hash(text_key(text))
            if conflicts.size and conflicts[min(np.searchsorted(conflicts, hashed), conflicts.size - 1)] == hashed:
                stats["cross_label"] += 1
                continue
            pools[domain].append((domain, name, text, hashed))
    negative = label in NEGATIVE_LABELS
    splits = {"train": ctx["negatives_cap"] if negative else ctx["cap"]}
    if not negative:
        splits["valid"] = ctx["valid"]
    crop_share = ctx["crop_share"]
    sentence_need = {s: n - int(round(n * crop_share)) for s, n in splits.items()}
    crop_need = {s: splits[s] - sentence_need[s] for s in splits}
    spare = {s: int(math.ceil(n * 0.03)) + 5 for s, n in splits.items()}
    for domain in pools:
        rng.shuffle(pools[domain])
    available = {d: len(pools.get(d, [])) for d in DOMAINS}
    wanted = sum(sentence_need.values()) + sum(spare.values())
    alloc = allocate(ctx["mix"], available, wanted)
    chosen = []
    for domain in DOMAINS:
        chosen.extend(pools.get(domain, [])[: alloc[domain]])
    rng.shuffle(chosen)
    unused = [row for d in DOMAINS for row in pools.get(d, [])[alloc[d] :]]
    rng.shuffle(unused)
    seen = {row[3] for d in pools for row in pools[d]}
    out: dict[str, list[tuple]] = {s: [] for s in splits}  # (kind, domain, source, text, hash)
    cursor = 0
    for split in ("valid", "train"):
        if split not in splits:
            continue
        take = sentence_need[split] + spare[split]
        for domain, name, text, hashed in chosen[cursor : cursor + take]:
            out[split].append(("sentence", domain, name, text, hashed))
        cursor += take
    leftovers = chosen[cursor:]
    check = Filter(ctx["min_chars"], ctx["max_chars"], ctx["min_words"])

    def crops_from(parents, need, split):
        made = 0
        for domain, name, text, _ in parents:
            if made >= need:
                break
            if domain == "short":
                continue
            crop = make_crop(text, label, rng)
            if crop is None:
                stats["crop_parent_too_short"] += 1
                continue
            result = check(crop, label, need_words=False)
            if isinstance(result, str):
                stats[f"crop_{result}"] += 1
                continue
            crop_text, key = result
            hashed = key_hash(key)
            if hashed in seen:
                stats["crop_duplicate"] += 1
                continue
            if ctx["flores"].check(key) == "exact" or (ctx["heldout"] and ctx["heldout"].check(key) == "exact"):
                stats["crop_contaminated_exact"] += 1
                continue
            seen.add(hashed)
            out[split].append(("crop", f"crop/{domain}", name, crop_text, hashed))
            made += 1
        return made

    parents = unused + leftovers
    position = 0
    for split in ("valid", "train"):
        if split not in splits:
            continue
        need = crop_need[split] + spare[split]
        made = 0
        while made < need and position < len(parents):
            batch = parents[position : position + (need - made)]
            position += len(batch)
            made += crops_from(batch, need - made, split)
        if split == "train" and made < need:
            # Second pass: crops of selected training segments (other windows).
            train_parents = [(r[1], r[2], r[3], r[4]) for r in out["train"] if r[0] == "sentence"]
            for _ in range(3):
                if made >= need:
                    break
                made += crops_from(train_parents, need - made, split)
        if made < need:
            stats[f"crop_shortfall:{split}"] += need - made
    return {"label": label, "rows": out, "drops": dict(stats),
            "alloc": alloc, "available": available,
            "targets": {"sentences": sentence_need, "crops": crop_need}}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def ensure_flores(raw_dir: Path, flores_dir: Path | None, lock: list) -> Path:
    if flores_dir is None:
        flores_dir = raw_dir / "flores200_dataset"
        if not flores_dir.exists():
            archive = raw_dir / "flores200_dataset.tar.gz"
            stream = HashedStream(FLORES_URL)
            with archive.open("wb") as handle:
                while chunk := stream.read(1 << 22):
                    handle.write(chunk)
            stream.close()
            if stream.digest.hexdigest() != FLORES_SHA256:
                archive.unlink()
                raise ValueError("FLORES-200 checksum mismatch")
            with tarfile.open(archive) as tar:
                tar.extractall(raw_dir, filter="data")
            archive.unlink()  # raw download removed after extraction
    digest = hashlib.sha256()
    for path in sorted(flores_dir.glob("dev*/*")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    lock.append({"source": "flores200 (filter only)", "url": FLORES_URL, "archive_sha256": FLORES_SHA256,
                 "dir": str(flores_dir), "sha256_of_dev_devtest_files": digest.hexdigest()})
    return flores_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "data/prepared_v2")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw_multidomain",
                        help="Scratch for temporary downloads and FLORES (gitignored).")
    parser.add_argument("--public-raw-dir", type=Path, default=ROOT / "data/raw",
                        help="Where prepare_public_data.py keeps the WiLI/MASSIVE archives.")
    parser.add_argument("--flores-dir", type=Path, help="Extracted flores200_dataset (else downloaded).")
    parser.add_argument("--cap", type=int, default=50000, help="Training rows per language.")
    parser.add_argument("--valid", type=int, default=1000, help="In-corpus validation rows per language.")
    parser.add_argument("--negatives-cap", type=int, default=20000, help="Rows in negatives/nn.csv.")
    parser.add_argument("--crop-share", type=float, default=0.3)
    parser.add_argument("--mix", default="web=0.40,wiki=0.30,short=0.15,other=0.15")
    parser.add_argument("--oversample", type=float, default=1.5,
                        help="Candidates collected per domain relative to its share of cap+valid.")
    parser.add_argument("--per-doc", type=int, default=5, help="Max segments per web/wiki document.")
    parser.add_argument("--min-chars", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=300)
    parser.add_argument("--min-words", type=int, default=3, help="Web/wiki segments, spaced scripts.")
    parser.add_argument("--languages", help="Comma-separated subset (smoke runs; manifest marks partial).")
    parser.add_argument("--skip-public", action="store_true",
                        help="Do not use WiLI/MASSIVE (their test splits are then NOT filtered out).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=945)
    args = parser.parse_args(argv)
    mix = {k: float(v) for k, v in (item.split("=") for item in args.mix.split(","))}
    if set(mix) != set(DOMAINS) or abs(sum(mix.values()) - 1) > 1e-6:
        parser.error(f"--mix must give {DOMAINS} summing to 1")
    if not 0 <= args.crop_share < 1 or args.cap < 1 or args.valid < 0:
        parser.error("invalid --crop-share/--cap/--valid")

    started = time.time()
    with (ROOT / "data/languages.json").open(encoding="utf-8") as stream:
        languages = list(json.load(stream))
    if languages != sorted(languages) or len(languages) != 100:
        raise ValueError("data/languages.json must list 100 codes in sorted order")
    labels = languages + list(NEGATIVE_LABELS)
    if args.languages:
        requested = args.languages.split(",")
        unknown = [c for c in requested if c not in labels]
        if unknown:
            parser.error(f"unknown labels {unknown}")
        labels = requested
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    work = args.raw_dir / "work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    global_lock: list[dict] = []
    flores_dir = ensure_flores(args.raw_dir, args.flores_dir, global_lock)
    print("[index] FLORES-200 ...", flush=True)
    flores = ShingleIndex(flores_lines(flores_dir))
    control = (flores_dir / "dev" / "eng_Latn.dev").read_text(encoding="utf-8").splitlines()[0]
    if flores.check(text_key(control)) != "exact":
        raise RuntimeError("FLORES positive control failed: a dev line is not detected")
    print(f"[index] FLORES lines {flores.size}, word shingles {flores.word.size}, char shingles {flores.char.size}", flush=True)
    public_wiki, public_short, heldout_texts = {}, {}, project_heldout_texts()
    heldout_sets = ["data/synthetic.csv", "tests/fixtures/authored_tickets.jsonl"]
    if not args.skip_public:
        public_wiki, public_short, public_heldout, public_lock = load_public(args.public_raw_dir, languages)
        heldout_texts += public_heldout
        global_lock += public_lock
        heldout_sets += ["WiLI-2018 test", "WiLI-2018 prepare_public_data validation (50/lang)",
                         "MASSIVE dev", "MASSIVE test"]
    heldout = ShingleIndex(heldout_texts)
    print(f"[index] held-out texts {heldout.size}", flush=True)
    CONTEXT.update(
        cap=args.cap, valid=args.valid, negatives_cap=args.negatives_cap, mix=mix,
        oversample=args.oversample, per_doc=args.per_doc, min_chars=args.min_chars,
        max_chars=args.max_chars, min_words=args.min_words, seed=args.seed,
        raw_dir=args.raw_dir, work=work, flores=flores, heldout=heldout,
        public_wiki=public_wiki, public_short=public_short, crop_share=args.crop_share,
    )
    context = multiprocessing.get_context("fork")  # indexes are shared copy-on-write
    with context.Pool(args.workers) as pool:
        collected = {r["label"]: r for r in pool.imap_unordered(collect_language, labels)}
    print(f"[collect] done in {time.time() - started:.0f}s", flush=True)

    label_index = {label: i for i, label in enumerate(labels)}
    hashes, owners = [], []
    for label in labels:
        values = np.load(work / "cand" / f"{label}.npy")
        hashes.append(values)
        owners.append(np.full(values.size, label_index[label], dtype=np.int64))
    conflicts = cross_label_conflicts(np.concatenate(hashes), np.concatenate(owners))
    print(f"[dedup] cross-label keys: {conflicts.size}", flush=True)
    CONTEXT["conflicts"] = conflicts
    with context.Pool(args.workers) as pool:
        selected = {r["label"]: r for r in pool.imap_unordered(select_language, labels)}

    # Phase D: crops and sentences must not share a key with any other label.
    pair_h, pair_l = [np.concatenate(hashes)], [np.concatenate(owners)]
    for label in labels:
        for split_rows in selected[label]["rows"].values():
            values = np.array([r[4] for r in split_rows if r[0] == "crop"], dtype=np.int64)
            pair_h.append(values)
            pair_l.append(np.full(values.size, label_index[label], dtype=np.int64))
    final_conflicts = cross_label_conflicts(np.concatenate(pair_h), np.concatenate(pair_l))
    blocked = set(final_conflicts.tolist())
    outputs: dict[str, list[dict]] = {"train": [], "valid": [], "negatives": []}
    meta: dict[str, list[dict]] = {"train": [], "valid": [], "negatives": []}
    report: dict[str, dict] = {}
    for label in labels:
        result = selected[label]
        drops = Counter(collected[label]["drops"]) + Counter(result["drops"])
        counts = {}
        for split, rows in result["rows"].items():
            target = result["targets"]["sentences"][split] + result["targets"]["crops"][split]
            kept_sent, kept_crop = [], []
            for row in rows:
                if row[4] in blocked:
                    drops[f"cross_label_final:{row[0]}"] += 1
                    continue
                (kept_crop if row[0] == "crop" else kept_sent).append(row)
            crop_target = result["targets"]["crops"][split]
            crops = kept_crop[:crop_target]
            # A crop shortfall is filled with spare segments (counted, not silent).
            sentences = kept_sent[: target - len(crops)]
            final = sentences + crops
            filled = max(0, len(sentences) - (target - crop_target))
            if filled:
                drops[f"crop_shortfall_filled_with_sentences:{split}"] += filled
            if len(final) < target:
                drops[f"below_target:{split}"] += target - len(final)
            drops[f"spare_unused:{split}"] += len(kept_sent) + len(kept_crop) - len(final)
            destination = "negatives" if label in NEGATIVE_LABELS else split
            for kind, domain, name, text, _ in final:
                outputs[destination].append({"text": text, "language": label})
                meta[destination].append({"language": label, "kind": kind, "domain": domain, "source": name})
            counts[split] = {
                "rows": len(final), "target": target,
                "crops": sum(1 for r in final if r[0] == "crop"),
                "by_domain": dict(Counter(r[1] for r in final)),
                "by_source": dict(Counter(r[2] for r in final)),
            }
        report[label] = {
            "counts": counts,
            "candidates": collected[label]["candidates"],
            "candidates_by_source": collected[label]["by_source"],
            "domain_alloc": result["alloc"], "domain_available": result["available"],
            "drops": dict(sorted(drops.items())),
            "source_errors": collected[label]["errors"],
            "collect_seconds": collected[label]["seconds"],
        }
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        order = list(range(len(outputs[split])))
        random.Random(f"{args.seed}:{split}").shuffle(order)
        write_csv(args.out / f"{split}.csv", [outputs[split][i] for i in order])
        write_sidecar(args.out / f"{split}_sources.tsv", [meta[split][i] for i in order])
    if outputs["negatives"]:
        write_csv(args.out / "negatives" / "nn.csv", outputs["negatives"])
        write_sidecar(args.out / "negatives" / "nn_sources.tsv", meta["negatives"])
    lock_records = global_lock + [r for label in labels for r in collected[label]["lock"]]
    totals = Counter()
    for label in labels:
        totals.update(report[label]["drops"])
    manifest = {
        "languages": languages,
        "language_count": len(languages),
        "partial": bool(args.languages),
        "labels_built": labels,
        "negatives": {"nn": "negatives/nn.csv (not a training label)"},
        "experimental": "Not used for the bundled model.pt.",
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "domain_mix": mix,
        "label_decisions": {"aliases": LABEL_ALIASES, "dropped_codes": DROPPED_CODES,
                            "expected_scripts": {l: list(expected_scripts(l)) for l in labels},
                            "unspaced_labels": sorted(UNSPACED_LABELS)},
        "filters": {
            "flores": "all 204 dev + 204 devtest files; exact text_key match, or >=50% of the "
                      "segment's 8-word (>=8 words) / 30-char shingles found in FLORES",
            "heldout_sets": heldout_sets,
            "near_dedup": "first/last 40 characters of text_key, per label",
            "cross_label": "keys (dataset_utils.text_key) seen under >1 label are dropped everywhere",
        },
        "counts": {label: report[label]["counts"] for label in labels},
        "per_language": report,
        "drop_totals": dict(sorted(totals.items())),
        "counter_legend": COUNTER_LEGEND,
        "cross_label_keys": int(conflicts.size),
        "cross_label_keys_final": int(final_conflicts.size),
        "licences": LICENCES,
        "openlid_allowed_tags": {t: v[0] for t, v in OPENLID_ALLOWED_TAGS.items()},
        "openlid_excluded_tags": OPENLID_EXCLUDED_TAGS,
        "hf_revisions": {k: {"repo": r, "revision": v} for k, (r, v) in HF_REVISIONS.items()},
        "sources_lock": lock_records,
        "files": {name: hashlib.sha256((args.out / name).read_bytes()).hexdigest()
                  for name in ("train.csv", "valid.csv", "negatives/nn.csv") if (args.out / name).exists()},
        "seconds": round(time.time() - started, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.rmtree(work)
    print(f"[done] train {len(outputs['train'])} valid {len(outputs['valid'])} "
          f"negatives {len(outputs['negatives'])} in {time.time() - started:.0f}s", flush=True)


def write_sidecar(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["language", "kind", "domain", "source"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
