"""Manifest loading, validation, filtering, and sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


REQUIRED_COLUMNS = {"audio_path", "text", "split", "source"}


def _make_audio_abs_path(audio_path: object, audio_root: str | None) -> str:
    path = Path(str(audio_path)).expanduser()
    if not path.is_absolute():
        root = Path(audio_root or ".").expanduser()
        path = root / path
    return str(path.resolve())


def read_manifest(config: Dict[str, object]) -> pd.DataFrame:
    """Read and validate the manifest CSV."""
    manifest_path = config.get("manifest_path")
    if not manifest_path:
        raise ValueError("manifest_path is not set in config.yaml")
    path = Path(str(manifest_path))
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    if "sample_id" not in df.columns:
        df.insert(0, "sample_id", [str(i) for i in df.index])
    else:
        df["sample_id"] = df["sample_id"].astype(str)

    df["audio_abs_path"] = df["audio_path"].apply(
        lambda value: _make_audio_abs_path(value, str(config.get("audio_root") or "."))
    )
    return df


def filter_manifest(df: pd.DataFrame, config: Dict[str, object]) -> pd.DataFrame:
    """Apply split, source, audio-existence, and sampling filters."""
    result = df.copy()
    split_to_use = config.get("split_to_use")
    if split_to_use:
        result = result[result["split"].astype(str) == str(split_to_use)]

    sources_to_use = config.get("sources_to_use")
    if sources_to_use:
        allowed = {str(source) for source in sources_to_use}
        result = result[result["source"].astype(str).isin(allowed)]

    exists_mask = result["audio_abs_path"].apply(lambda path: Path(str(path)).exists())
    result = result[exists_mask].copy()

    return sample_manifest(result, config).reset_index(drop=True)


def sample_manifest(df: pd.DataFrame, config: Dict[str, object]) -> pd.DataFrame:
    """Sample rows using the configured strategy."""
    max_samples = config.get("max_samples_per_source")
    if max_samples is None or pd.isna(max_samples):
        return df.copy()

    max_samples = int(max_samples)
    seed = int(config.get("random_seed", 42))
    strategy = str(config.get("sampling_strategy", "balanced_per_source"))

    if max_samples <= 0:
        return df.head(0).copy()

    if strategy == "balanced_per_source":
        return (
            df.groupby("source", group_keys=False)
            .apply(lambda group: group.sample(n=min(len(group), max_samples), random_state=seed))
            .reset_index(drop=True)
        )

    if strategy == "first_n":
        return df.head(max_samples).copy()

    raise ValueError(f"Unsupported sampling_strategy: {strategy}")


def load_dataset(config: Dict[str, object]) -> pd.DataFrame:
    """Read, validate, filter, and sample the configured manifest."""
    return filter_manifest(read_manifest(config), config)
