"""Optional ticket input and sentence guard (frozen policy, runtime).

Operates only on caller-supplied title/description strings. The guard
sees exactly the same request source as extract_ticket_text (shared
selection in ticket_text, never a different field), split into segments
after structural cleanup but before whitespace collapse, so plain error
lines stay visible while fenced logs never become evidence. Segments
need >=3 Unicode letter words; short tickets stay in with zero segments.

Only the first 2000 characters (the model-visible prefix) can supply
segments: a line straddling the edge is clipped, later lines are
excluded. The veto fires on one foreign-confident sentence only;
uncertain and lexical add-ons stay disabled.

Offline and dependency-light: standard library plus ftfy/ticket_text
only — never fastText.
"""

import re
from dataclasses import dataclass
from typing import Callable, List, Mapping, Sequence, Tuple

from ticket_text import (
    extract_ticket_text,
    has_request_substance,
    normalize_ticket_text,
    select_request_source,
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
FULL_PREFIX_CHARS = 2000


@dataclass(frozen=True)
class RawPrediction:
    """Raw (pre-acceptance) model vote: best label and confidence."""

    best: str
    confidence: float
    margin: float = 0.0
    agreement: bool = True


@dataclass(frozen=True)
class PreparedTicket:
    """Cleaned whole-ticket text plus guard segments (>=3 words each)."""

    text: str
    sentences: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TicketPolicy:
    """Minimal guard settings: name and sentence cutoff in [0, 1]."""

    name: str
    sentence_cutoff: float = 0.65

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("policy name must be a nonempty string")
        cut = self.sentence_cutoff
        if not isinstance(cut, (int, float)) or isinstance(cut, bool):
            raise ValueError("sentence_cutoff must be a number in [0, 1]")
        if not 0.0 <= float(cut) <= 1.0:
            raise ValueError("sentence_cutoff must be within [0, 1]")


def letter_words(text: str) -> List[str]:
    """Letter-only tokens; digit mixes and underscores never count."""
    out: List[str] = []
    for token in re.findall(r"\S+", text or ""):
        if any(ch.isdigit() for ch in token) or "_" in token:
            continue
        letters = "".join(ch for ch in token if ch.isalpha())
        if letters:
            out.append(letters)
    return out


def _split_prefix_sentences(raw: str) -> List[str]:
    """Guard sentences from raw source lines within the 2000-char prefix."""
    sentences: List[str] = []
    offset = 0
    for raw_line in re.sub(r"\r\n?", "\n", raw).split("\n"):
        cleaned = normalize_ticket_text(raw_line)
        if not cleaned:
            continue
        start = offset
        offset += len(cleaned) + 1
        if start >= FULL_PREFIX_CHARS:
            continue
        cleaned = cleaned[:FULL_PREFIX_CHARS - start]
        for part in SENTENCE_SPLIT.split(cleaned):
            part = part.strip()
            if part and len(letter_words(part)) >= 3:
                sentences.append(part)
    return sentences


def _split_whole_text(text: str) -> List[str]:
    """Fallback segmentation of the cleaned whole text (same filter)."""
    out: List[str] = []
    for part in SENTENCE_SPLIT.split(text or ""):
        part = part.strip()
        if part and len(letter_words(part)) >= 3:
            out.append(part)
    return out


def prepare_ticket(title: str, description: str = "") -> PreparedTicket:
    """Build the common input text plus guard segments for one ticket."""
    text = extract_ticket_text(title or "", description or "")
    raw = select_request_source(description or "")
    if not has_request_substance(normalize_ticket_text(raw)):
        raw = title or ""
    sentences = _split_prefix_sentences(raw)
    if any(s not in text for s in sentences):
        sentences = _split_whole_text(text)
    return PreparedTicket(text=text, sentences=tuple(sentences))


def guarded_language(
    whole: str,
    segments: Sequence[str],
    predict_fn: Callable[[str], RawPrediction],
    cutoff: float,
) -> str:
    """Veto the whole-ticket answer on a confident foreign segment."""
    TicketPolicy(name="ticket-guard", sentence_cutoff=cutoff)
    if whole == "und":
        return "und"
    for segment in segments or ():
        prediction = predict_fn(segment)
        if prediction.best == "und" or prediction.best == whole:
            continue
        if prediction.confidence >= float(cutoff):
            return "und"
    return whole
