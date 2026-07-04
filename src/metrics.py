"""WER/CER metrics and summary tables."""

from __future__ import annotations

from typing import Dict, Iterable, List

import jiwer
import numpy as np
import pandas as pd

from src.text_normalization import normalize_for_wer


EPS = 1e-12


def safe_wer(reference: str, hypothesis: str) -> float:
    """Compute WER with stable behavior for empty references."""
    reference = reference or ""
    hypothesis = hypothesis or ""
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    return float(jiwer.wer(reference, hypothesis))


def safe_cer(reference: str, hypothesis: str) -> float:
    """Compute CER with stable behavior for empty references."""
    reference = reference or ""
    hypothesis = hypothesis or ""
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    return float(jiwer.cer(reference, hypothesis))


def status_from_scores(before: float, after: float) -> str:
    """Classify whether a metric improved, degraded, or stayed unchanged."""
    if after < before - EPS:
        return "improved"
    if after > before + EPS:
        return "degraded"
    return "unchanged"


def normalize_columns(df: pd.DataFrame, config: Dict[str, object]) -> pd.DataFrame:
    """Add normalized reference, Whisper, and LLM prediction columns."""
    result = df.copy()
    result["reference_norm"] = result["reference"].apply(lambda x: normalize_for_wer(x, config))
    result["whisper_prediction_norm"] = result["whisper_prediction"].apply(
        lambda x: normalize_for_wer(x, config)
    )
    result["llm_corrected_prediction_norm"] = result["llm_corrected_prediction"].apply(
        lambda x: normalize_for_wer(x, config)
    )
    return result


def add_per_sample_metrics(df: pd.DataFrame, config: Dict[str, object]) -> pd.DataFrame:
    """Add WER/CER before and after LLM correction for every sample."""
    result = normalize_columns(df, config)
    result["wer_before"] = [
        safe_wer(ref, hyp)
        for ref, hyp in zip(result["reference_norm"], result["whisper_prediction_norm"])
    ]
    result["wer_after"] = [
        safe_wer(ref, hyp)
        for ref, hyp in zip(result["reference_norm"], result["llm_corrected_prediction_norm"])
    ]
    result["cer_before"] = [
        safe_cer(ref, hyp)
        for ref, hyp in zip(result["reference_norm"], result["whisper_prediction_norm"])
    ]
    result["cer_after"] = [
        safe_cer(ref, hyp)
        for ref, hyp in zip(result["reference_norm"], result["llm_corrected_prediction_norm"])
    ]
    result["wer_gain"] = result["wer_before"] - result["wer_after"]
    result["cer_gain"] = result["cer_before"] - result["cer_after"]
    result["status_wer"] = [
        status_from_scores(before, after)
        for before, after in zip(result["wer_before"], result["wer_after"])
    ]
    result["status_cer"] = [
        status_from_scores(before, after)
        for before, after in zip(result["cer_before"], result["cer_after"])
    ]
    return result


def micro_metric(refs: Iterable[str], hyps: Iterable[str], metric: str) -> float:
    """Compute corpus-level WER or CER."""
    refs_list: List[str] = [str(x or "") for x in refs]
    hyps_list: List[str] = [str(x or "") for x in hyps]
    if not refs_list:
        return 0.0
    if metric == "wer":
        return float(jiwer.wer(refs_list, hyps_list))
    if metric == "cer":
        return float(jiwer.cer(refs_list, hyps_list))
    raise ValueError(f"Unsupported metric: {metric}")


def pct(value: float) -> float:
    """Convert a ratio to percent."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return float(value) * 100.0


def summarize_by_source(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Create per-source metric summary."""
    rows = []
    for source, group in metrics_df.groupby("source", dropna=False):
        wer_before = micro_metric(group["reference_norm"], group["whisper_prediction_norm"], "wer")
        wer_after = micro_metric(group["reference_norm"], group["llm_corrected_prediction_norm"], "wer")
        cer_before = micro_metric(group["reference_norm"], group["whisper_prediction_norm"], "cer")
        cer_after = micro_metric(group["reference_norm"], group["llm_corrected_prediction_norm"], "cer")
        rows.append(
            {
                "source": source,
                "n": int(len(group)),
                "wer_before_percent": pct(wer_before),
                "wer_after_percent": pct(wer_after),
                "wer_gain_percent": pct(wer_before - wer_after),
                "cer_before_percent": pct(cer_before),
                "cer_after_percent": pct(cer_after),
                "cer_gain_percent": pct(cer_before - cer_after),
                "improved_percent": pct((group["status_wer"] == "improved").mean()),
                "degraded_percent": pct((group["status_wer"] == "degraded").mean()),
                "unchanged_percent": pct((group["status_wer"] == "unchanged").mean()),
                "changed_by_llm_percent": pct(group["changed_by_llm"].astype(bool).mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_global(metrics_df: pd.DataFrame, by_source_df: pd.DataFrame) -> Dict[str, float]:
    """Create global micro/macro metric summary."""
    micro_wer_before = micro_metric(metrics_df["reference_norm"], metrics_df["whisper_prediction_norm"], "wer")
    micro_wer_after = micro_metric(
        metrics_df["reference_norm"], metrics_df["llm_corrected_prediction_norm"], "wer"
    )
    micro_cer_before = micro_metric(metrics_df["reference_norm"], metrics_df["whisper_prediction_norm"], "cer")
    micro_cer_after = micro_metric(
        metrics_df["reference_norm"], metrics_df["llm_corrected_prediction_norm"], "cer"
    )

    changed = metrics_df["changed_by_llm"].astype(bool)
    worsened_when_changed = changed & (metrics_df["wer_after"] > metrics_df["wer_before"] + EPS)
    overcorrection_rate = (
        float(worsened_when_changed.sum()) / float(changed.sum()) if int(changed.sum()) > 0 else 0.0
    )

    return {
        "n": int(len(metrics_df)),
        "micro_wer_before_percent": pct(micro_wer_before),
        "micro_wer_after_percent": pct(micro_wer_after),
        "micro_wer_gain_percent": pct(micro_wer_before - micro_wer_after),
        "macro_wer_before_percent": float(by_source_df["wer_before_percent"].mean()) if len(by_source_df) else 0.0,
        "macro_wer_after_percent": float(by_source_df["wer_after_percent"].mean()) if len(by_source_df) else 0.0,
        "macro_wer_gain_percent": float(by_source_df["wer_gain_percent"].mean()) if len(by_source_df) else 0.0,
        "micro_cer_before_percent": pct(micro_cer_before),
        "micro_cer_after_percent": pct(micro_cer_after),
        "micro_cer_gain_percent": pct(micro_cer_before - micro_cer_after),
        "macro_cer_before_percent": float(by_source_df["cer_before_percent"].mean()) if len(by_source_df) else 0.0,
        "macro_cer_after_percent": float(by_source_df["cer_after_percent"].mean()) if len(by_source_df) else 0.0,
        "macro_cer_gain_percent": float(by_source_df["cer_gain_percent"].mean()) if len(by_source_df) else 0.0,
        "improved_percent": pct((metrics_df["status_wer"] == "improved").mean()),
        "degraded_percent": pct((metrics_df["status_wer"] == "degraded").mean()),
        "unchanged_percent": pct((metrics_df["status_wer"] == "unchanged").mean()),
        "changed_by_llm_percent": pct(changed.mean()),
        "overcorrection_rate_percent": pct(overcorrection_rate),
    }
