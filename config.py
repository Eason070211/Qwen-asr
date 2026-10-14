"""
Qwen3-ASR Server - configuration (env vars + sensible defaults).

All settings can be overridden via environment variables prefixed with QWEN_ASR_.

The `voice_*` group below configures the client-side voice input tool
(voice_input.py): silence detection ("VAD"), the transcription queue and the
input-level handling. Every one of them can also be overridden per-run on the
command line, e.g.

    python voice_input.py --silence-db -42 --silence-hold 0.8

or permanently via environment variables, e.g.

    set QWEN_ASR_VOICE_SILENCE_DB=-42
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
    workers: int = 1                       # process count; CPU inference is serial
    # NOTE: inference is strictly serialised (one shared model instance, one
    # inference thread), so this does NOT give you parallel transcription.
    # Measured: 6 concurrent requests finish one after another. Raise it only
    # if you replace the backend with something thread-safe.
    max_concurrent: int = 2
    request_timeout: int = 300             # seconds

    # --- audio -----------------------------------------------------------
    sample_rate: int = 16000
    max_audio_seconds: int = 1200          # 20 minutes

    # --- voice input tool (voice_input.py) -------------------------------
    # Silence-gated segmentation. A "segment" is one chunk of speech that is
    # sent to the server as one transcription job; segments are split apart
    # whenever the input stays quiet for `voice_silence_hold` seconds.
    voice_hotkey: str = "right ctrl"
    voice_cancel_hotkey: str = ""          # "" = disabled; e.g. "right alt"
    voice_language: str = ""               # "" = auto-detect
    voice_paste: bool = True               # auto Ctrl+V at the cursor
    voice_max_seconds: int = 300           # whole-session auto-stop watchdog
    voice_device: str = "16"               # "" = system default; or index/"name"

    voice_vad: bool = True                 # False = never split, one job on stop
    voice_adaptive_noise: bool = True      # measure the room, derive the gates
    voice_silence_db: float = -45.0        # fixed-mode fallback: quiet below this
    voice_speech_db: float = -38.0         # fixed-mode fallback: speech above this
    voice_silence_hold: float = 0.65       # quiet this long -> close segment
    voice_min_speech: float = 0.30         # ignore shorter blips (clicks/noise)
    voice_preroll: float = 0.35            # audio kept before speech starts
    voice_max_segment: float = 30.0        # force-split runaway monologue
    voice_min_segment: float = 0.35        # drop segments shorter than this
    voice_restore_clipboard: bool = True   # put the old clipboard back after paste

    # --- queue -----------------------------------------------------------
    # Transcription jobs are processed in the background, so recording never
    # blocks on inference. One worker keeps the server's serialized CPU
    # inference busy without piling requests on top of each other.
    voice_queue_workers: int = 1
    voice_queue_max: int = 64              # backlog cap (oldest dropped)
    voice_backlog_warn: int = 5            # warn when this many jobs are pending
    voice_retry: int = 2                   # retries per segment when the server fails

    # --- result safety net ------------------------------------------------
    # Transcribed text is appended to a transcript log and failed segments keep
    # their audio, so nothing is lost if a paste misses, the window is closed or
    # the server dies mid-session.
    voice_log_transcripts: bool = True
    voice_transcripts_dir: str = str(PROJECT_ROOT / "transcripts")
    voice_failed_audio_dir: str = str(PROJECT_ROOT / "failed_audio")

    # --- input level / language -------------------------------------------
    # Windows devices with automatic gain control (the AMD mic array on this
    # machine drifts over ~33 dB) hand the model anything from -50 to -15 dBFS,
    # and the model's answer tracks that level: a quiet chunk comes back as a
    # single filler character or an empty string. Normalise each segment to a
    # target RMS before sending it.
    voice_auto_gain: bool = True
    voice_gain_target_db: float = -22.0
    # Language auto-detection returns nothing for Chinese on this model, so an
    # empty result is retried once with this language ("" disables the retry).
    voice_fallback_language: str = "Chinese"

    # --- misc ------------------------------------------------------------
    log_level: str = "INFO"
    # Print the level and result of every transcription request. Enabled by
    # `server.py --debug-requests`: a quiet chunk makes the model answer with
    # filler or nothing, and these numbers show that immediately.
    debug_requests: bool = False
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
