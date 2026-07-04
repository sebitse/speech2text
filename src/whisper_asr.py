"""Whisper ASR inference using a local HuggingFace checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class WhisperASR:
    """Small wrapper around a fine-tuned Whisper checkpoint."""

    def __init__(self, config: Dict[str, object]):
        self.config = config
        model_path = config.get("whisper_model_path")
        if not model_path:
            raise ValueError("whisper_model_path is not set in config.yaml")
        if not Path(str(model_path)).exists():
            raise FileNotFoundError(f"Whisper model path not found: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(str(model_path))
        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio with soundfile/librosa and resample to the configured rate."""
        target_sr = int((self.config.get("whisper", {}) or {}).get("sampling_rate", 16000))
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return np.asarray(audio, dtype=np.float32)

    def _forced_decoder_ids(self) -> Optional[list]:
        whisper_cfg = self.config.get("whisper", {}) or {}
        language = whisper_cfg.get("language")
        task = whisper_cfg.get("task", "transcribe")
        if not language:
            return None
        try:
            return self.processor.get_decoder_prompt_ids(language=str(language), task=str(task))
        except Exception:
            return None

    @torch.inference_mode()
    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe one audio file."""
        target_sr = int((self.config.get("whisper", {}) or {}).get("sampling_rate", 16000))
        audio = self.load_audio(audio_path)
        inputs = self.processor(audio, sampling_rate=target_sr, return_tensors="pt")
        input_features = inputs.input_features.to(device=self.device, dtype=self.dtype)

        whisper_cfg = self.config.get("whisper", {}) or {}
        generate_kwargs = {
            "max_new_tokens": int(whisper_cfg.get("max_new_tokens", 256)),
        }
        forced_decoder_ids = self._forced_decoder_ids()
        if forced_decoder_ids is not None:
            generate_kwargs["forced_decoder_ids"] = forced_decoder_ids

        predicted_ids = self.model.generate(input_features, **generate_kwargs)
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
