"""Run Whisper transcription on the configured manifest subset."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.config import load_config, output_path
from src.data import load_dataset
from src.io_utils import append_row_csv, read_csv_if_exists, setup_logging
from src.whisper_asr import WhisperASR


OUTPUT_COLUMNS = [
    "sample_id",
    "source",
    "split",
    "audio_path",
    "audio_abs_path",
    "reference",
    "whisper_prediction",
]


def run(config_path: str) -> Path:
    config = load_config(config_path)
    logger = setup_logging(config["output_dir"], "whisper_transcription")
    predictions_path = output_path(config, "predictions", str(config["predictions_filename"]))
    if not config.get("resume", True) and predictions_path.exists():
        predictions_path.unlink()

    dataset = load_dataset(config)
    existing = read_csv_if_exists(predictions_path)
    done_ids = set(existing["sample_id"].astype(str)) if not existing.empty and config.get("resume", True) else set()
    pending = dataset[~dataset["sample_id"].astype(str).isin(done_ids)].copy()

    logger.info("Loaded %s samples; %s pending for Whisper.", len(dataset), len(pending))
    asr = WhisperASR(config)

    for _, row in tqdm(pending.iterrows(), total=len(pending), desc="Whisper"):
        sample_id = str(row["sample_id"])
        try:
            prediction = asr.transcribe_file(str(row["audio_abs_path"]))
        except Exception as exc:
            logger.exception("Whisper failed for sample_id=%s: %s", sample_id, exc)
            prediction = ""

        append_row_csv(
            predictions_path,
            {
                "sample_id": sample_id,
                "source": row["source"],
                "split": row["split"],
                "audio_path": row["audio_path"],
                "audio_abs_path": row["audio_abs_path"],
                "reference": row["text"],
                "whisper_prediction": prediction,
            },
            OUTPUT_COLUMNS,
        )

    logger.info("Whisper predictions saved to %s", predictions_path)
    return predictions_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
