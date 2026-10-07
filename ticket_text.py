"""Local text cleanup for support requests and email messages."""

import html
import re

import ftfy


def normalize_ticket_text(text: str) -> str:
    text = html.unescape(ftfy.fix_text(text))
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+", " ", text)
    text = re.sub(r"<[^>]{0,200}>", " ", text)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", " ", text)
    text = re.sub(
        r"\b[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}\b", " ", text
    )
    text = re.sub(r"[A-Za-z]:\\\S+", " ", text)
    # Remove request numbers, hostnames with digits, and technical identifiers.
    text = re.sub(r"\b(?=\w*[A-Za-z])(?=\w*\d)\w+\b|\b\w+_\w+\b", " ", text)
    text = re.sub(r"\b\d+(?:[.:/-]\d+)*\b", " ", text)
    return " ".join(text.split())


def extract_ticket_text(title: str, description: str = "") -> str:
    """Prefer description content after removing system and email headers."""
    body = ftfy.fix_text(description or "")
    body = re.sub(
        r"^\s*Following Service Request.*?Users Comment:\s*",
        "",
        body,
        flags=re.DOTALL | re.IGNORECASE,
    )
    body = re.sub(
        r"-{5,}\s*System:.*?-{5,}", " ", body, flags=re.DOTALL | re.IGNORECASE
    )
    if re.match(r"\s*(?:From|Von):", body, flags=re.IGNORECASE):
        separator = re.search(r"\s---\s", body)
        if separator:
            body = body[separator.end() :]
        else:
            subject = re.search(r"\b(?:Subject|Betreff):", body, flags=re.IGNORECASE)
            if subject:
                greeting = re.search(
                    r"\b(?:Hi|Dear|Hello|Hallo|Guten Morgen|Guten Tag|Ahoj)\b",
                    body[subject.end() :],
                    flags=re.IGNORECASE,
                )
                if greeting:
                    body = body[subject.end() + greeting.start() :]
    # Ignore quoted replies when extracting the original request.
    body = re.split(
        r"-{8,}\s*(?=From:|Von:|发件人:)", body, maxsplit=1, flags=re.IGNORECASE
    )[0]
    body = re.split(
        r"(?<!\w)(?:Best regards|Freundliche Grüße|Mit freundlichen Grüßen|Kind regards|Warm Regards|S pozdravem)\b",
        body,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    body = normalize_ticket_text(body)
    title = normalize_ticket_text(title)
    if (
        sum(word.isalpha() for word in body.split()) >= 3
        or sum(char.isalpha() for char in body) >= 12
    ):
        return body[:2000]
    return title[:2000]
