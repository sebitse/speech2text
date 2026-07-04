"""Run the complete Whisper -> LLM correction -> evaluation pipeline."""

from __future__ import annotations

import argparse

from evaluate_results import run as run_evaluation
from run_llm_correction import run as run_llm_correction
from run_whisper_transcription import run as run_whisper_transcription


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    run_whisper_transcription(args.config)
    run_llm_correction(args.config)
    run_evaluation(args.config)


if __name__ == "__main__":
    main()
