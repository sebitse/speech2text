"""I/O helpers for logging, directories, and incremental CSV writes."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


OUTPUT_SUBDIRS = ("predictions", "corrections", "metrics", "logs")


def ensure_output_dirs(output_dir: str | Path) -> None:
    """Create the standard output directory tree."""
    root = Path(output_dir)
    for subdir in OUTPUT_SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)


def setup_logging(output_dir: str | Path, name: str = "whisper_llm") -> logging.Logger:
    """Configure console and file logging."""
    ensure_output_dirs(output_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = Path(output_dir) / "logs" / f"{name}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    """Read a CSV if present; otherwise return an empty DataFrame."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def append_row_csv(path: str | Path, row: Dict[str, object], columns: Iterable[str]) -> None:
    """Append one row to a CSV, writing the header only once."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    pd.DataFrame([row], columns=list(columns)).to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8",
    )


def save_dataframe(path: str | Path, df: pd.DataFrame) -> None:
    """Save a dataframe to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def save_json(path: str | Path, payload: Dict[str, object]) -> None:
    """Save a JSON payload with UTF-8 encoding."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
