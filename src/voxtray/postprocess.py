from __future__ import annotations

import re


_MULTISPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")


def normalize_transcript(text: str) -> str:
    cleaned = text.replace("\r", " ").replace("\n", " ").strip()
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return cleaned
