"""Ollama-based conservative transcript correction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import requests

from src.metrics import safe_cer


PROMPT_VERSION = "ro_conservative_v1"


PROMPT_TEMPLATE = """Corecteaza transcriptul ASR de mai jos.

Reguli:
1. Corecteaza doar greselile evidente de transcriere, diacriticele, punctuatia, capitalizarea, numerele si termenii tehnici.
2. Nu reformula propozitiile.
3. Nu adauga informatie noua.
4. Nu elimina informatie.
5. Pastreaza ordinea cuvintelor cat mai aproape de transcriptul initial.
6. Daca transcriptul este deja acceptabil, returneaza-l neschimbat.
7. Returneaza doar transcriptul corectat, fara explicatii.

Transcript ASR:
\"\"\"{transcript}\"\"\"

Transcript corectat:"""


@dataclass
class CorrectionResult:
    corrected_text: str
    status: str
    latency_sec: float
    prompt_version: str = PROMPT_VERSION


def build_prompt(transcript: str) -> str:
    """Build the conservative Romanian correction prompt."""
    return PROMPT_TEMPLATE.format(transcript=transcript or "")


def _clean_response(text: str) -> str:
    text = (text or "").strip()
    if text.startswith('"""') and text.endswith('"""') and len(text) >= 6:
        text = text[3:-3].strip()
    return text


def correct_transcript(transcript: str, config: Dict[str, object]) -> CorrectionResult:
    """Correct one transcript through local Ollama, falling back to input on failure."""
    transcript = "" if transcript is None else str(transcript)
    prompt = build_prompt(transcript)
    url = str(config.get("ollama_url"))
    model = str(config.get("llm_model", "llama3.1:8b"))
    max_retries = int(config.get("llm_max_retries", 3))
    timeout = int(config.get("llm_timeout_sec", 120))
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(config.get("llm_temperature", 0.0)),
            "num_predict": int(config.get("llm_num_predict", 256)),
        },
    }

    start = time.time()
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            corrected = _clean_response(str(data.get("response", "")))
            if not corrected:
                raise ValueError("Ollama returned an empty correction")
            return CorrectionResult(
                corrected_text=corrected,
                status="ok",
                latency_sec=time.time() - start,
            )
        except Exception:
            if attempt == max_retries:
                return CorrectionResult(
                    corrected_text=transcript,
                    status="failed",
                    latency_sec=time.time() - start,
                )
            time.sleep(min(2**attempt, 10))

    return CorrectionResult(transcript, "failed", time.time() - start)


def correction_row_fields(transcript: str, corrected: str) -> Dict[str, object]:
    """Compute common correction metadata fields."""
    transcript = "" if transcript is None else str(transcript)
    corrected = "" if corrected is None else str(corrected)
    return {
        "changed_by_llm": transcript.strip() != corrected.strip(),
        "raw_to_llm_cer": safe_cer(transcript, corrected),
    }
