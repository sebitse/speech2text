"""WER/CER metrics and summary tables."""

from __future__ import annotations

from typing import Dict, Iterable, List

import jiwer
import numpy as np
import pandas as pd

from src.text_normalization import normalize_for_wer


EPS = 1e-12


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


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
    """Add normalized reference and all prediction variant columns."""
    result = ensure_evaluation_columns(df)
    result["reference_norm"] = result["reference"].apply(lambda x: normalize_for_wer(x, config))
    result["whisper_prediction_norm"] = result["whisper_prediction"].apply(
        lambda x: normalize_for_wer(x, config)
    )
    result["llm_corrected_prediction_norm"] = result["llm_corrected_prediction"].apply(
        lambda x: normalize_for_wer(x, config)
    )
    result["final_prediction_norm"] = result["final_prediction"].apply(
        lambda x: normalize_for_wer(x, config)
    )
    return result


def _bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    true_values = {"true", "1", "yes", "y", "accepted"}
    false_values = {"false", "0", "no", "n", "rejected", "skipped", ""}

    def convert(value: object) -> bool:
        if pd.isna(value):
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in true_values:
            return True
        if text in false_values:
            return False
        return default

    return series.apply(convert).astype(bool)


def ensure_evaluation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill optional LLM columns when evaluating older correction CSVs."""
    result = df.copy()
    required = ["reference", "whisper_prediction"]
    missing_required = [column for column in required if column not in result.columns]
    if missing_required:
        raise ValueError(
            "Corrections CSV is missing required columns: " + ", ".join(missing_required)
        )

    if "source" not in result.columns:
        result["source"] = "unknown"
    result["source"] = result["source"].apply(lambda value: _text(value) or "unknown")
    if "llm_corrected_prediction" not in result.columns:
        result["llm_corrected_prediction"] = result["whisper_prediction"]
    for column in ("reference", "whisper_prediction", "llm_corrected_prediction"):
        result[column] = result[column].apply(_text)
    if "llm_status" not in result.columns:
        result["llm_status"] = "ok"
    if "llm_skip_reason" not in result.columns:
        result["llm_skip_reason"] = np.where(result["llm_status"] == "skipped", "unknown", "not_skipped")
    if "llm_accepted" not in result.columns:
        result["llm_accepted"] = result["llm_status"].astype(str).str.lower().eq("ok")
    result["llm_accepted"] = _bool_series(result["llm_accepted"])

    skipped = result["llm_status"].astype(str).str.lower().eq("skipped")
    if "final_prediction" not in result.columns:
        result["final_prediction"] = np.where(
            skipped | ~result["llm_accepted"],
            result["whisper_prediction"],
            result["llm_corrected_prediction"],
        )
    result["final_prediction"] = result["final_prediction"].apply(_text)
    if "llm_accept_reason" not in result.columns:
        result["llm_accept_reason"] = np.where(
            skipped,
            "skipped",
            np.where(result["llm_accepted"], "accepted_or_legacy", "rejected_or_legacy"),
        )
    if "llm_rejected" not in result.columns:
        result["llm_rejected"] = (~skipped) & (~result["llm_accepted"])
    result["llm_rejected"] = _bool_series(result["llm_rejected"])

    if "changed_by_llm" not in result.columns:
        result["changed_by_llm"] = [
            _text(raw).strip() != _text(corrected).strip()
            for raw, corrected in zip(result["whisper_prediction"], result["llm_corrected_prediction"])
        ]
    result["changed_by_llm"] = _bool_series(result["changed_by_llm"])

    if "raw_to_llm_cer" not in result.columns:
        result["raw_to_llm_cer"] = np.nan
    if "length_ratio" not in result.columns:
        result["length_ratio"] = np.nan

    for idx, row in result.iterrows():
        raw = _text(row.get("whisper_prediction"))
        corrected = _text(row.get("llm_corrected_prediction"))
        if pd.isna(result.at[idx, "raw_to_llm_cer"]):
            result.at[idx, "raw_to_llm_cer"] = safe_cer(raw, corrected)
        if pd.isna(result.at[idx, "length_ratio"]):
            result.at[idx, "length_ratio"] = len(corrected.strip()) / max(len(raw.strip()), 1)

    return result


def add_per_sample_metrics(df: pd.DataFrame, config: Dict[str, object]) -> pd.DataFrame:
    """Add WER/CER metrics for Whisper raw, LLM all, and LLM verified."""
    result = normalize_columns(df, config)
    result["wer_before"] = [
        safe_wer(ref, hyp)
        for ref, hyp in zip(result["reference_norm"], result["whisper_prediction_norm"])
    ]
    result["wer_llm_all"] = [
        safe_wer(ref, hyp)
        for ref, hyp in zip(result["reference_norm"], result["llm_corrected_prediction_norm"])
    ]
    result["wer_verified"] = [
        safe_wer(ref, hyp)
        for ref, hyp in zip(result["reference_norm"], result["final_prediction_norm"])
    ]
    result["cer_before"] = [
        safe_cer(ref, hyp)
        for ref, hyp in zip(result["reference_norm"], result["whisper_prediction_norm"])
    ]
    result["cer_llm_all"] = [
        safe_cer(ref, hyp)
        for ref, hyp in zip(result["reference_norm"], result["llm_corrected_prediction_norm"])
    ]
    result["cer_verified"] = [
        safe_cer(ref, hyp)
        for ref, hyp in zip(result["reference_norm"], result["final_prediction_norm"])
    ]
    result["wer_gain_llm_all"] = result["wer_before"] - result["wer_llm_all"]
    result["wer_gain_llm_verified"] = result["wer_before"] - result["wer_verified"]
    result["cer_gain_llm_all"] = result["cer_before"] - result["cer_llm_all"]
    result["cer_gain_llm_verified"] = result["cer_before"] - result["cer_verified"]
    result["status_llm_all"] = [
        status_from_scores(before, after)
        for before, after in zip(result["wer_before"], result["wer_llm_all"])
    ]
    result["status_llm_verified"] = [
        status_from_scores(before, after)
        for before, after in zip(result["wer_before"], result["wer_verified"])
    ]
    result["status_cer_llm_all"] = [
        status_from_scores(before, after)
        for before, after in zip(result["cer_before"], result["cer_llm_all"])
    ]
    result["status_cer_llm_verified"] = [
        status_from_scores(before, after)
        for before, after in zip(result["cer_before"], result["cer_verified"])
    ]

    result["wer_after"] = result["wer_verified"]
    result["cer_after"] = result["cer_verified"]
    result["wer_gain"] = result["wer_gain_llm_verified"]
    result["cer_gain"] = result["cer_gain_llm_verified"]
    result["status_wer"] = result["status_llm_verified"]
    result["status_cer"] = result["status_cer_llm_verified"]
    return result


def micro_metric(refs: Iterable[str], hyps: Iterable[str], metric: str) -> float:
    """Compute corpus-level WER or CER."""
    refs_list: List[str] = [_text(x) for x in refs]
    hyps_list: List[str] = [_text(x) for x in hyps]
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


def _overcorrection_rate(group: pd.DataFrame, candidate_mask: pd.Series, wer_column: str) -> float:
    candidates = candidate_mask.astype(bool)
    if int(candidates.sum()) == 0:
        return 0.0
    worsened = candidates & (group[wer_column] > group["wer_before"] + EPS)
    return float(worsened.sum()) / float(candidates.sum())


def _summary_flags(group: pd.DataFrame) -> Dict[str, float]:
    skipped = group["llm_status"].astype(str).str.lower().eq("skipped")
    accepted = _bool_series(group["llm_accepted"])
    rejected = _bool_series(group["llm_rejected"])
    changed = _bool_series(group["changed_by_llm"])
    return {
        "llm_skipped_percent": pct(skipped.mean()),
        "llm_run_percent": pct((~skipped).mean()),
        "llm_accepted_percent": pct(accepted.mean()),
        "llm_rejected_percent": pct(rejected.mean()),
        "changed_by_llm_percent": pct(changed.mean()),
        "overcorrection_rate_llm_all_percent": pct(
            _overcorrection_rate(group, changed, "wer_llm_all")
        ),
        "overcorrection_rate_llm_verified_percent": pct(
            _overcorrection_rate(group, accepted, "wer_verified")
        ),
    }


def summarize_by_source(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Create per-source metric summary."""
    rows = []
    for source, group in metrics_df.groupby("source", dropna=False):
        wer_whisper = micro_metric(group["reference_norm"], group["whisper_prediction_norm"], "wer")
        wer_llm_all = micro_metric(group["reference_norm"], group["llm_corrected_prediction_norm"], "wer")
        wer_llm_verified = micro_metric(group["reference_norm"], group["final_prediction_norm"], "wer")
        cer_whisper = micro_metric(group["reference_norm"], group["whisper_prediction_norm"], "cer")
        cer_llm_all = micro_metric(group["reference_norm"], group["llm_corrected_prediction_norm"], "cer")
        cer_llm_verified = micro_metric(group["reference_norm"], group["final_prediction_norm"], "cer")
        flags = _summary_flags(group)
        rows.append(
            {
                "source": source,
                "n": int(len(group)),
                "wer_whisper_percent": pct(wer_whisper),
                "wer_llm_all_percent": pct(wer_llm_all),
                "wer_llm_verified_percent": pct(wer_llm_verified),
                "cer_whisper_percent": pct(cer_whisper),
                "cer_llm_all_percent": pct(cer_llm_all),
                "cer_llm_verified_percent": pct(cer_llm_verified),
                "wer_gain_llm_all_percent": pct(wer_whisper - wer_llm_all),
                "wer_gain_llm_verified_percent": pct(wer_whisper - wer_llm_verified),
                "cer_gain_llm_all_percent": pct(cer_whisper - cer_llm_all),
                "cer_gain_llm_verified_percent": pct(cer_whisper - cer_llm_verified),
                "improved_llm_all_percent": pct((group["status_llm_all"] == "improved").mean()),
                "degraded_llm_all_percent": pct((group["status_llm_all"] == "degraded").mean()),
                "improved_llm_verified_percent": pct((group["status_llm_verified"] == "improved").mean()),
                "degraded_llm_verified_percent": pct((group["status_llm_verified"] == "degraded").mean()),
                "unchanged_llm_all_percent": pct((group["status_llm_all"] == "unchanged").mean()),
                "unchanged_llm_verified_percent": pct((group["status_llm_verified"] == "unchanged").mean()),
                **flags,
            }
        )
    return pd.DataFrame(rows)


def _mean_column(df: pd.DataFrame, column: str) -> float:
    return float(df[column].mean()) if len(df) and column in df.columns else 0.0


def summarize_global(metrics_df: pd.DataFrame, by_source_df: pd.DataFrame) -> Dict[str, float]:
    """Create global micro/macro metric summary."""
    micro_wer_whisper = micro_metric(metrics_df["reference_norm"], metrics_df["whisper_prediction_norm"], "wer")
    micro_wer_llm_all = micro_metric(
        metrics_df["reference_norm"], metrics_df["llm_corrected_prediction_norm"], "wer"
    )
    micro_wer_llm_verified = micro_metric(metrics_df["reference_norm"], metrics_df["final_prediction_norm"], "wer")
    micro_cer_whisper = micro_metric(metrics_df["reference_norm"], metrics_df["whisper_prediction_norm"], "cer")
    micro_cer_llm_all = micro_metric(
        metrics_df["reference_norm"], metrics_df["llm_corrected_prediction_norm"], "cer"
    )
    micro_cer_llm_verified = micro_metric(metrics_df["reference_norm"], metrics_df["final_prediction_norm"], "cer")
    flags = _summary_flags(metrics_df)

    return {
        "n": int(len(metrics_df)),
        "micro_wer_whisper_raw_percent": pct(micro_wer_whisper),
        "micro_wer_llm_all_percent": pct(micro_wer_llm_all),
        "micro_wer_llm_verified_percent": pct(micro_wer_llm_verified),
        "macro_wer_whisper_raw_percent": _mean_column(by_source_df, "wer_whisper_percent"),
        "macro_wer_llm_all_percent": _mean_column(by_source_df, "wer_llm_all_percent"),
        "macro_wer_llm_verified_percent": _mean_column(by_source_df, "wer_llm_verified_percent"),
        "wer_gain_llm_all_percent": pct(micro_wer_whisper - micro_wer_llm_all),
        "wer_gain_llm_verified_percent": pct(micro_wer_whisper - micro_wer_llm_verified),
        "micro_cer_whisper_raw_percent": pct(micro_cer_whisper),
        "micro_cer_llm_all_percent": pct(micro_cer_llm_all),
        "micro_cer_llm_verified_percent": pct(micro_cer_llm_verified),
        "macro_cer_whisper_raw_percent": _mean_column(by_source_df, "cer_whisper_percent"),
        "macro_cer_llm_all_percent": _mean_column(by_source_df, "cer_llm_all_percent"),
        "macro_cer_llm_verified_percent": _mean_column(by_source_df, "cer_llm_verified_percent"),
        "cer_gain_llm_all_percent": pct(micro_cer_whisper - micro_cer_llm_all),
        "cer_gain_llm_verified_percent": pct(micro_cer_whisper - micro_cer_llm_verified),
        "improved_llm_all_percent": pct((metrics_df["status_llm_all"] == "improved").mean()),
        "degraded_llm_all_percent": pct((metrics_df["status_llm_all"] == "degraded").mean()),
        "unchanged_llm_all_percent": pct((metrics_df["status_llm_all"] == "unchanged").mean()),
        "improved_llm_verified_percent": pct((metrics_df["status_llm_verified"] == "improved").mean()),
        "degraded_llm_verified_percent": pct((metrics_df["status_llm_verified"] == "degraded").mean()),
        "unchanged_llm_verified_percent": pct((metrics_df["status_llm_verified"] == "unchanged").mean()),
        **flags,
        "per_source": by_source_df.to_dict(orient="records"),
    }
