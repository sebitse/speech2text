"""Text normalization utilities for Romanian ASR evaluation."""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Dict


ROMANIAN_DIACRITICS = str.maketrans(
    {
        "ă": "a",
        "â": "a",
        "î": "i",
        "ș": "s",
        "ş": "s",
        "ț": "t",
        "ţ": "t",
        "Ă": "A",
        "Â": "A",
        "Î": "I",
        "Ș": "S",
        "Ş": "S",
        "Ț": "T",
        "Ţ": "T",
    }
)


def strip_diacritics(text: str) -> str:
    """Remove Romanian diacritics and any remaining combining marks."""
    text = text.translate(ROMANIAN_DIACRITICS)
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def remove_punctuation(text: str) -> str:
    """Replace punctuation with spaces while keeping letters and digits."""
    punctuation = string.punctuation + "„”«»…–—"
    translation = str.maketrans({char: " " for char in punctuation})
    return text.translate(translation)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim ends."""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_for_wer(text: object, config: Dict[str, object]) -> str:
    """Normalize text according to config['normalization'] for WER/CER."""
    norm_config = config.get("normalization", {}) or {}
    value = "" if text is None else str(text)

    if bool(norm_config.get("lowercase", True)):
        value = value.lower()
    if not bool(norm_config.get("keep_diacritics", True)):
        value = strip_diacritics(value)
    if bool(norm_config.get("remove_punctuation", True)):
        value = remove_punctuation(value)

    return normalize_whitespace(value)
