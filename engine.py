"""
Qwen3-ASR inference engine — single-process, CPU-optimized.

Design choices:
  * Model is loaded ONCE at startup; all requests share the same instance.
  * A threading.Semaphore serialises inference to avoid memory blow on CPU.
  * The qwen_asr package handles audio normalization internally, but we
    still call audio.prepare_audio() for uploaded files to be safe.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from config import settings, MODEL_DIR, ALIGNER_DIR
from audio import prepare_audio, SAMPLE_RATE

logger = logging.getLogger("qwen3-asr.engine")

# Global state
_model = None
_aligner = None
_executor = ThreadPoolExecutor(max_workers=1)
_semaphore = asyncio.Semaphore(settings.max_concurrent)


async def load_model():
    """Load the Qwen3-ASR model (called once at startup)."""
    global _model, _aligner
    import torch
    from qwen_asr import Qwen3ASRModel

    t0 = time.time()
    logger.info("Loading model from %s ...", MODEL_DIR)

    # Resolve dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(settings.dtype, torch.float32)

    # Build kwargs
    kwargs = dict(
        dtype=dtype,
        device_map=settings.device,
        max_inference_batch_size=settings.max_inference_batch_size,
        max_new_tokens=settings.max_new_tokens,
    )

    # Forced aligner (optional — enables timestamps)
    forced_aligner_path = None
    if ALIGNER_DIR and ALIGNER_DIR.exists():
        forced_aligner_path = str(ALIGNER_DIR)
        logger.info("Loading forced aligner from %s ...", forced_aligner_path)
        kwargs["forced_aligner"] = forced_aligner_path
        kwargs["forced_aligner_kwargs"] = dict(
            dtype=dtype,
            device_map=settings.device,
        )

    _model = Qwen3ASRModel.from_pretrained(str(MODEL_DIR), **kwargs)
    elapsed = time.time() - t0
    logger.info("Model loaded in %.1f s", elapsed)


def transcribe_sync(
    audio: np.ndarray,
    language: Optional[str] = None,
    context: str = "",
    return_timestamps: bool = False,
):
    """Synchronous transcription — runs in the executor thread."""
    if _model is None:
        raise RuntimeError("Model not loaded yet")

    # qwen_asr accepts (np.ndarray, sr) tuples directly
    audio_input = (audio, SAMPLE_RATE)

    results = _model.transcribe(
        audio=audio_input,
        language=language,
        context=context,
        return_time_stamps=return_timestamps,
    )
    return results[0]  # single audio -> single result


async def transcribe(
    audio: np.ndarray,
    language: Optional[str] = None,
    context: str = "",
    return_timestamps: bool = False,
):
    """Async wrapper — acquires semaphore, runs inference in thread pool."""
    async with _semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: transcribe_sync(audio, language, context, return_timestamps),
        )


def get_model_info() -> dict:
    """Return model metadata for the /v1/models endpoint."""
    if _model is None:
        return {"id": settings.model_id, "status": "loading"}
    return {
        "id": settings.model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "qwen",
        "permission": [],
        "root": str(MODEL_DIR),
        "backend": _model.backend,
        "device": str(getattr(_model, "device", "cpu")),
        "dtype": str(getattr(_model, "dtype", "float32")),
        "forced_aligner": _aligner is not None,
    }
