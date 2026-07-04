"""Feature extraction for future XGBoost gating experiments."""

from __future__ import annotations

import string
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import soundfile as sf


DIACRITICS = set("ăâîșşțţĂÂÎȘŞȚŢ")
PUNCTUATION = set(string.punctuation + "„”«»…–—")


def audio_duration_sec(audio_path: str) -> Optional[float]:
    """Return audio duration in seconds when the file is readable."""
    try:
        info = sf.info(str(audio_path))
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        return None
    return None


def text_features(text: object, duration_sec: Optional[float]) -> Dict[str, float]:
    """Extract simple transcript-level features."""
    value = "" if text is None else str(text)
    words = value.split()
    num_chars = len(value)
    num_words = len(words)
    unique_words = len(set(word.lower() for word in words))
    repeated_words = max(num_words - unique_words, 0)
    duration = duration_sec if duration_sec and duration_sec > 0 else None

    return {
        "num_chars": float(num_chars),
        "num_words": float(num_words),
        "avg_word_length": float(np.mean([len(word) for word in words])) if words else 0.0,
        "chars_per_second": float(num_chars / duration) if duration else np.nan,
        "words_per_second": float(num_words / duration) if duration else np.nan,
        "diacritics_ratio": float(sum(ch in DIACRITICS for ch in value) / num_chars) if num_chars else 0.0,
        "digit_ratio": float(sum(ch.isdigit() for ch in value) / num_chars) if num_chars else 0.0,
        "punctuation_ratio": float(sum(ch in PUNCTUATION for ch in value) / num_chars) if num_chars else 0.0,
        "unique_word_ratio": float(unique_words / num_words) if num_words else 0.0,
        "repetition_ratio": float(repeated_words / num_words) if num_words else 0.0,
    }


def extract_features_for_xgboost(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Build a feature table for a later LLM-gating model."""
    rows = []
    for _, row in metrics_df.iterrows():
        duration = audio_duration_sec(row.get("audio_abs_path")) if Path(str(row.get("audio_abs_path", ""))).exists() else None
        features = text_features(row.get("whisper_prediction", ""), duration)
        features.update(
            {
                "sample_id": row.get("sample_id"),
                "source": row.get("source"),
                "changed_by_llm": bool(row.get("changed_by_llm", False)),
                "raw_to_llm_cer": float(row.get("raw_to_llm_cer", 0.0)),
                "wer_before": float(row.get("wer_before", 0.0)),
                "wer_after": float(row.get("wer_after", 0.0)),
                "llm_helped": int(float(row.get("wer_after", 0.0)) < float(row.get("wer_before", 0.0))),
            }
        )
        rows.append(features)
    return pd.DataFrame(rows)
