"""Configuration loading and validation utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "manifest_path": None,
    "audio_root": ".",
    "whisper_model_path": None,
    "output_dir": "outputs",
    "split_to_use": "test",
    "sources_to_use": None,
    "max_samples_per_source": None,
    "random_seed": 42,
    "sampling_strategy": "balanced_per_source",
    "resume": True,
    "predictions_filename": "whisper_predictions.csv",
    "corrections_filename": "llm_corrections.csv",
    "ollama_url": "http://localhost:11434/api/generate",
    "llm_model": "llama3.1:8b",
    "llm_temperature": 0.0,
    "llm_num_predict": 256,
    "llm_max_retries": 3,
    "llm_timeout_sec": 120,
    "normalization": {
        "lowercase": True,
        "remove_punctuation": True,
        "keep_diacritics": True,
    },
    "whisper": {
        "sampling_rate": 16000,
        "language": "ro",
        "task": "transcribe",
        "max_new_tokens": 256,
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge nested dictionaries without mutating either input."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_path(value: Any, base_dir: Path) -> Any:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file, apply defaults, and resolve local paths."""
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    config = deep_merge(DEFAULT_CONFIG, user_config)
    base_dir = config_path.parent
    for key in ("manifest_path", "audio_root", "whisper_model_path", "output_dir"):
        config[key] = _resolve_path(config.get(key), base_dir)

    strategy = config.get("sampling_strategy")
    if strategy not in {"balanced_per_source", "first_n"}:
        raise ValueError("sampling_strategy must be 'balanced_per_source' or 'first_n'")

    sources = config.get("sources_to_use")
    if sources is not None and not isinstance(sources, list):
        raise ValueError("sources_to_use must be a list or null")

    return config


def output_path(config: Dict[str, Any], subdir: str, filename: str) -> Path:
    """Return an output path under output_dir/subdir."""
    return Path(config["output_dir"]) / subdir / filename
