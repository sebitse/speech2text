"""Evaluate Whisper predictions before and after LLM correction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis import extract_features_for_xgboost
from src.config import load_config, output_path
from src.io_utils import save_dataframe, save_json, setup_logging
from src.metrics import add_per_sample_metrics, summarize_by_source, summarize_global


def run(config_path: str) -> Path:
    config = load_config(config_path)
    logger = setup_logging(config["output_dir"], "evaluation")
    corrections_path = output_path(config, "corrections", str(config["corrections_filename"]))

    if not corrections_path.exists():
        raise FileNotFoundError(f"LLM corrections not found: {corrections_path}")

    corrections = pd.read_csv(corrections_path)
    optional_columns = [
        "llm_corrected_prediction",
        "llm_status",
        "llm_skip_reason",
        "llm_accepted",
        "llm_accept_reason",
        "raw_to_llm_cer",
        "length_ratio",
        "final_prediction",
    ]
    missing_optional = [column for column in optional_columns if column not in corrections.columns]
    if missing_optional:
        logger.warning(
            "Corrections CSV is missing optional LLM columns; evaluation will infer defaults: %s",
            ", ".join(missing_optional),
        )

    metrics_df = add_per_sample_metrics(corrections, config)
    by_source = summarize_by_source(metrics_df)
    global_metrics = summarize_global(metrics_df, by_source)
    features = extract_features_for_xgboost(metrics_df)

    metrics_dir = Path(config["output_dir"]) / "metrics"
    per_sample_path = metrics_dir / "per_sample_metrics.csv"
    by_source_path = metrics_dir / "per_source_metrics.csv"
    global_path = metrics_dir / "global_metrics.json"
    features_path = metrics_dir / "features_for_xgboost.csv"

    save_dataframe(per_sample_path, metrics_df)
    save_dataframe(by_source_path, by_source)
    save_json(global_path, global_metrics)
    save_dataframe(features_path, features)

    logger.info("Per-sample metrics saved to %s", per_sample_path)
    logger.info("Per-source metrics saved to %s", by_source_path)
    logger.info("Global metrics saved to %s", global_path)
    logger.info("XGBoost feature table saved to %s", features_path)
    return global_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
