"""
Audio decoding utilities for Qwen3-ASR server.

Handles: file upload bytes, URL download, base64 data-URL, and raw numpy.
Always returns mono float32 numpy at 16 kHz.
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests
import soundfile as sf

SAMPLE_RATE = 16000


# ------------------------------------------------------------------ public

def decode_audio(
    content: Optional[bytes] = None,
    url: Optional[str] = None,
    data_url: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """Return (waveform, sample_rate) from any supported input source."""
    if content is not None:
        return _decode_bytes(content)
    if url is not None:
        return _decode_url(url)
    if data_url is not None:
        return _decode_data_url(data_url)
    raise ValueError("No audio input provided (provide content, url, or data_url)")


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Ensure audio is mono float32."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return audio.astype(np.float32)


def resample_if_needed(audio: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simple resample via linear interpolation (good enough for ASR)."""
    if sr == target_sr:
        return audio
    # librosa.resample is the preferred path; fall back to linear interp
    try:
        import librosa
        return librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    except ImportError:
        pass
    # lightweight fallback
    duration = len(audio) / sr
    target_len = int(duration * target_sr)
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def prepare_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Full pipeline: mono -> float32 -> resample to 16 kHz."""
    audio = to_mono_float32(audio)
    audio = resample_if_needed(audio, sr, SAMPLE_RATE)
    return audio


# ----------------------------------------------------------------- private

def _decode_bytes(raw: bytes) -> Tuple[np.ndarray, int]:
    """Decode audio from raw bytes via an in-memory buffer.

    (NamedTemporaryFile fails on some Windows setups — libsndfile reports
    "System error" when opening the still-open temp file; BytesIO avoids
    the filesystem entirely.)
    """
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    return audio, sr


def _decode_url(url: str) -> Tuple[np.ndarray, int]:
    """Download audio from a URL and decode."""
    resp = requests.get(url, timeout=60, headers={"User-Agent": "qwen3-asr-server/1.0"})
    resp.raise_for_status()
    return _decode_bytes(resp.content)


def _decode_data_url(data_url: str) -> Tuple[np.ndarray, int]:
    """Decode a data:audio/...;base64,... URL."""
    match = re.match(r"data:audio/[^;]+;base64,(.+)", data_url, re.DOTALL)
    if not match:
        raise ValueError("Invalid data URL format")
    raw = base64.b64decode(match.group(1))
    return _decode_bytes(raw)
