"""Run local LLM conservative post-correction for Whisper predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import load_config, output_path
from src.io_utils import append_row_csv, read_csv_if_exists, setup_logging
from src.llm_corrector import PROMPT_VERSION, correct_transcript, correction_row_fields


EXTRA_COLUMNS = [
    "llm_corrected_prediction",
    "llm_status",
    "llm_model",
    "llm_prompt_version",
    "llm_latency_sec",
    "changed_by_llm",
    "raw_to_llm_cer",
]


def run(config_path: str) -> Path:
    config = load_config(config_path)
    logger = setup_logging(config["output_dir"], "llm_correction")
    predictions_path = output_path(config, "predictions", str(config["predictions_filename"]))
    corrections_path = output_path(config, "corrections", str(config["corrections_filename"]))
    if not config.get("resume", True) and corrections_path.exists():
        corrections_path.unlink()

    if not predictions_path.exists():
        raise FileNotFoundError(f"Whisper predictions not found: {predictions_path}")

    predictions = pd.read_csv(predictions_path)
    existing = read_csv_if_exists(corrections_path)
    done_ids = set(existing["sample_id"].astype(str)) if not existing.empty and config.get("resume", True) else set()
    pending = predictions[~predictions["sample_id"].astype(str).isin(done_ids)].copy()
    output_columns = list(predictions.columns) + EXTRA_COLUMNS

    logger.info("Loaded %s predictions; %s pending for LLM.", len(predictions), len(pending))
    for _, row in tqdm(pending.iterrows(), total=len(pending), desc="LLM correction"):
        transcript = "" if pd.isna(row.get("whisper_prediction")) else str(row.get("whisper_prediction"))
        result = correct_transcript(transcript, config)
        fields = correction_row_fields(transcript, result.corrected_text)

        output_row = row.to_dict()
        output_row.update(
            {
                "llm_corrected_prediction": result.corrected_text,
                "llm_status": result.status,
                "llm_model": config.get("llm_model"),
                "llm_prompt_version": PROMPT_VERSION,
                "llm_latency_sec": result.latency_sec,
                **fields,
            }
        )
        append_row_csv(corrections_path, output_row, output_columns)

    logger.info("LLM corrections saved to %s", corrections_path)
    return corrections_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
