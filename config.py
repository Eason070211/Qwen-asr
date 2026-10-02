"""
Qwen3-ASR Server - configuration (env vars + sensible defaults).

All settings can be overridden via environment variables prefixed with QWEN_ASR_.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # --- model -----------------------------------------------------------
    # NOTE: use the NATIVE-format repo (Qwen/Qwen3-ASR-0.6B), which matches
    # qwen-asr 0.0.6 (transformers 4.x). The *-hf repos need transformers 5.x.
    model_path: str = str(PROJECT_ROOT / "models" / "Qwen3-ASR-0.6B")
    forced_aligner_path: str = ""          # empty = no timestamps support
    dtype: str = "float32"                 # float32 | float16 | bfloat16
    device: str = "cpu"                    # cpu | cuda:0 | ...
    max_new_tokens: int = 512
    max_inference_batch_size: int = 1      # CPU: 1 is safest

    # --- server ----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8910
    workers: int = 1                       # CPU inference is single-threaded
    max_concurrent: int = 2                # semaphore limit for /v1/audio/*
    request_timeout: int = 300             # seconds

    # --- audio -----------------------------------------------------------
    sample_rate: int = 16000
    max_audio_seconds: int = 1200          # 20 minutes

    # --- misc ------------------------------------------------------------
    log_level: str = "INFO"
    model_id: str = "qwen3-asr-0.6b"      # friendly name for /v1/models

    model_config = {"env_prefix": "QWEN_ASR_"}


settings = Settings()

# Resolve model_path relative to project root if not absolute
_mp = Path(settings.model_path)
if not _mp.is_absolute():
    settings.model_path = str(PROJECT_ROOT / _mp)

# Resolved paths
MODEL_DIR = Path(settings.model_path)
ALIGNER_DIR = Path(settings.forced_aligner_path) if settings.forced_aligner_path else None
