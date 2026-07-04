# Whisper + LLM post-correction pentru Romanian ASR

Proiect pentru testarea unui flux experimental simplu:

```text
Audio -> Whisper ASR -> LLM conservative correction -> WER/CER before vs after
```

Scopul este sa verifici daca un LLM local rulat prin Ollama poate corecta conservator transcrierile unui model Whisper fine-tuned pentru limba romana, fara API-uri platite.

## Structura

```text
project/
├── config.yaml
├── requirements.txt
├── README.md
├── run_whisper_transcription.py
├── run_llm_correction.py
├── evaluate_results.py
├── run_full_pipeline.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── text_normalization.py
│   ├── whisper_asr.py
│   ├── llm_corrector.py
│   ├── metrics.py
│   ├── analysis.py
│   └── io_utils.py
└── outputs/
    ├── predictions/
    ├── corrections/
    ├── metrics/
    └── logs/
```

## Instalare

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Instaleaza versiunea de `torch` potrivita pentru GPU-ul tau daca vrei inferenta CUDA. Codul foloseste automat GPU daca `torch.cuda.is_available()` este true.

## Manifest CSV

Manifestul trebuie sa contina coloanele:

```text
audio_path,text,split,source
```

Optional, poate contine si `sample_id`. Daca lipseste, proiectul foloseste indexul randului din CSV.

`audio_path` poate fi absolut sau relativ la `audio_root`.

## Configurare

Editeaza `config.yaml`:

```yaml
manifest_path: "data/manifest.csv"
audio_root: "data/audio"
whisper_model_path: "FT_ALL_whisper_large_v2"
output_dir: "outputs"

split_to_use: "test"
sources_to_use: null
max_samples_per_source: 100
random_seed: 42
sampling_strategy: "balanced_per_source"
resume: true

ollama_url: "http://localhost:11434/api/generate"
llm_model: "llama3.1:8b"
llm_temperature: 0.0
llm_num_predict: 256
llm_max_retries: 3
llm_timeout_sec: 120

normalization:
  lowercase: true
  remove_punctuation: true
  keep_diacritics: true
```

Sampling:

- `balanced_per_source`: ia maximum `max_samples_per_source` exemple din fiecare `source`.
- `first_n`: ia primele `max_samples_per_source` exemple total.
- `max_samples_per_source: null`: pastreaza toate exemplele.

## Ollama

Porneste Ollama local si descarca modelul:

```bash
ollama serve
ollama pull llama3.1:8b
```

Test rapid:

```bash
ollama run llama3.1:8b
```

## Rulare

Transcriere Whisper:

```bash
python run_whisper_transcription.py --config config.yaml
```

Corectie LLM:

```bash
python run_llm_correction.py --config config.yaml
```

Evaluare:

```bash
python evaluate_results.py --config config.yaml
```

Pipeline complet:

```bash
python run_full_pipeline.py --config config.yaml
```

## Output-uri

`outputs/predictions/whisper_predictions.csv`:

- `sample_id`
- `source`
- `split`
- `audio_path`
- `audio_abs_path`
- `reference`
- `whisper_prediction`

`outputs/corrections/llm_corrections.csv`:

- toate coloanele din predictii
- `llm_corrected_prediction`
- `llm_status`
- `llm_model`
- `llm_prompt_version`
- `llm_latency_sec`
- `changed_by_llm`
- `raw_to_llm_cer`

`outputs/metrics/metrics_per_sample.csv`:

- WER/CER inainte si dupa LLM
- gain-uri
- status `improved`, `degraded` sau `unchanged`

`outputs/metrics/metrics_per_source.csv`:

- sumar pe fiecare `source`
- WER/CER in procente
- procente improved/degraded/unchanged
- procent de randuri modificate de LLM

`outputs/metrics/metrics_global.json`:

- metrici micro globale
- metrici macro mediate pe surse
- `overcorrection_rate_percent`, adica procentul exemplelor modificate de LLM unde WER s-a inrautatit

`outputs/metrics/features_for_xgboost.csv`:

- feature table pentru un viitor model de gating
- include lungimi, rate de caractere, rate de diacritice/cifre/punctuatie, `raw_to_llm_cer`, `wer_before`, `wer_after`, `llm_helped`

## Resume si robustete

Scripturile scriu incremental rand cu rand.

Daca transcrierea sau corectia se opreste, ruleaza din nou acelasi script cu `resume: true`. Randurile deja existente in CSV sunt sarite pe baza de `sample_id`.

Daca Ollama esueaza pentru un exemplu, transcriptul original este pastrat, iar `llm_status` devine `failed`.

## Interpretarea metricilor

- `wer_before_percent`: eroarea Whisper fata de referinta.
- `wer_after_percent`: eroarea dupa corectia LLM.
- `wer_gain_percent`: `before - after`; valoare pozitiva inseamna imbunatatire.
- `status_wer`: per sample, arata daca LLM a ajutat, a stricat sau nu a schimbat scorul.
- `changed_by_llm_percent`: cat de des LLM-ul modifica transcriptul.
- `overcorrection_rate_percent`: risc de supra-corectie pe randurile pe care LLM le-a modificat.

## Extensie XGBoost gating

Faza urmatoare poate folosi `features_for_xgboost.csv` pentru a antrena un model binar care prezice `llm_helped`.

Ideea: la inferenta, trimiti la LLM doar transcripturile pentru care gating-ul estimeaza ca post-corectia are sanse sa reduca WER, reducand latenta si riscul de supra-corectie.
