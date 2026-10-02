"""
Qwen3-ASR inference server — FastAPI, OpenAI-compatible + native endpoints.

Endpoints
---------
POST /v1/audio/transcriptions   — OpenAI Whisper-compatible
POST /v1/audio/translations     — same shape, forces English output
POST /v1/chat/completions       — OpenAI multimodal chat (audio_url)
GET  /v1/models                 — model listing
GET  /health                    — liveness + model status
WS   /v1/audio/stream           — real-time streaming (16 kHz PCM chunks)
POST /transcribe                — native JSON API
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import struct
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response

import config
from audio import decode_audio, prepare_audio, SAMPLE_RATE
import engine

logger = logging.getLogger("qwen3-asr.server")


# ============================================================ lifecycle

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, clean up at shutdown."""
    logger.info("Starting Qwen3-ASR server on %s:%s", config.settings.host, config.settings.port)
    await engine.load_model()
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Qwen3-ASR Server",
    description="CPU inference server for Qwen3-ASR with OpenAI-compatible API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================ helpers

def _srt_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_timestamp(seconds: float) -> str:
    """Format seconds as VTT timestamp HH:MM:SS.mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _format_timestamps_srt(result) -> str:
    """Convert forced aligner output to SRT format."""
    if not hasattr(result, "time_stamps") or result.time_stamps is None:
        return result.text
    lines = []
    for i, ts in enumerate(result.time_stamps, 1):
        start = _srt_timestamp(ts.start_time)
        end = _srt_timestamp(ts.end_time)
        lines.append(f"{i}\n{start} --> {end}\n{ts.text}\n")
    return "\n".join(lines)


def _format_timestamps_vtt(result) -> str:
    """Convert forced aligner output to WebVTT format."""
    if not hasattr(result, "time_stamps") or result.time_stamps is None:
        return result.text
    lines = ["WEBVTT\n"]
    for i, ts in enumerate(result.time_stamps, 1):
        start = _vtt_timestamp(ts.start_time)
        end = _vtt_timestamp(ts.end_time)
        lines.append(f"{i}\n{start} --> {end}\n{ts.text}\n")
    return "\n".join(lines)


def _format_timestamps_json(result) -> dict:
    """Convert forced aligner output to JSON with segments."""
    segs = []
    if hasattr(result, "time_stamps") and result.time_stamps:
        for ts in result.time_stamps:
            segs.append({
                "start": ts.start_time,
                "end": ts.end_time,
                "text": ts.text,
            })
    return {"text": result.text, "language": result.language, "segments": segs}


# ============================================================ POST /v1/audio/transcriptions

@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: Optional[UploadFile] = File(None),
    model: str = Form("qwen3-asr-0.6b"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(""),
    response_format: Optional[str] = Form("json"),
    timestamp_granularities: Optional[str] = Form(None),
):
    """OpenAI-compatible audio transcription endpoint."""
    # --- validate input ---
    if file is None:
        raise HTTPException(400, "No audio file provided")
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty audio file")

    # --- decode ---
    try:
        raw_audio, sr = decode_audio(content=content)
    except Exception as e:
        raise HTTPException(400, f"Failed to decode audio: {e}")
    audio = prepare_audio(raw_audio, sr)

    # --- language ---
    lang = language if language else None

    # --- timestamps ---
    want_ts = response_format in ("verbose_json", "srt", "vtt") or \
              (timestamp_granularities and "segment" in timestamp_granularities)

    # --- infer ---
    try:
        result = await engine.transcribe(
            audio=audio,
            language=lang,
            context=prompt or "",
            return_timestamps=want_ts,
        )
    except Exception as e:
        logger.exception("transcription failed")
        raise HTTPException(500, f"Inference error: {e}")

    # --- format response ---
    if response_format == "text":
        return PlainTextResponse(result.text)
    elif response_format == "verbose_json":
        return JSONResponse(_format_timestamps_json(result))
    elif response_format == "srt":
        return PlainTextResponse(_format_timestamps_srt(result))
    elif response_format == "vtt":
        return PlainTextResponse(_format_timestamps_vtt(result))
    else:  # json (default)
        return JSONResponse({"text": result.text, "language": result.language})


# ============================================================ POST /v1/audio/translations

@app.post("/v1/audio/translations")
async def openai_translations(
    file: Optional[UploadFile] = File(None),
    model: str = Form("qwen3-asr-0.6b"),
    prompt: Optional[str] = Form(""),
    response_format: Optional[str] = Form("json"),
):
    """OpenAI-compatible translation endpoint (forces English output)."""
    if file is None:
        raise HTTPException(400, "No audio file provided")
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty audio file")

    try:
        raw_audio, sr = decode_audio(content=content)
    except Exception as e:
        raise HTTPException(400, f"Failed to decode audio: {e}")
    audio = prepare_audio(raw_audio, sr)

    try:
        result = await engine.transcribe(
            audio=audio,
            language="English",
            context=prompt or "",
        )
    except Exception as e:
        logger.exception("translation failed")
        raise HTTPException(500, f"Inference error: {e}")

    if response_format == "text":
        return PlainTextResponse(result.text)
    return JSONResponse({"text": result.text, "language": "English"})


# ============================================================ POST /v1/chat/completions

@app.post("/v1/chat/completions")
async def openai_chat_completions(request_body: dict):
    """
    OpenAI-compatible chat completions endpoint for audio.
    Accepts audio_url or base64 audio in the message content.
    """
    messages = request_body.get("messages", [])
    if not messages:
        raise HTTPException(400, "No messages provided")

    # Extract audio from the last user message
    audio_url = None
    audio_data_url = None
    lang = None
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "audio_url":
                    audio_url = part.get("audio_url", {}).get("url")
                elif part.get("type") == "input_audio":
                    audio_data_url = part.get("input_audio", {}).get("data")
        break

    if not audio_url and not audio_data_url:
        raise HTTPException(400, "No audio found in messages")

    # Decode
    try:
        if audio_url:
            raw_audio, sr = decode_audio(url=audio_url)
        else:
            raw_audio, sr = decode_audio(data_url=audio_data_url)
    except Exception as e:
        raise HTTPException(400, f"Failed to decode audio: {e}")

    audio = prepare_audio(raw_audio, sr)

    # Infer
    try:
        result = await engine.transcribe(audio=audio, language=lang)
    except Exception as e:
        logger.exception("chat completion failed")
        raise HTTPException(500, f"Inference error: {e}")

    return JSONResponse({
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_body.get("model", config.settings.model_id),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


# ============================================================ GET /v1/models

@app.get("/v1/models")
async def list_models():
    """List available models."""
    info = engine.get_model_info()
    return JSONResponse({
        "object": "list",
        "data": [info],
    })


# ============================================================ GET /health

@app.get("/health")
async def health():
    """Liveness probe."""
    info = engine.get_model_info()
    status = "ok" if info.get("status") != "loading" else "loading"
    return JSONResponse({"status": status, "model": info})


# ============================================================ POST /transcribe

@app.post("/transcribe")
async def native_transcribe(
    file: Optional[UploadFile] = File(None),
    audio_url: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    context: Optional[str] = Form(""),
    return_timestamps: bool = Form(False),
):
    """
    Native transcription endpoint — richer than the OpenAI-compatible one.
    Accepts file upload OR audio_url (mutually exclusive).
    """
    if file is None and audio_url is None:
        raise HTTPException(400, "Provide either file or audio_url")

    try:
        if file:
            content = await file.read()
            raw_audio, sr = decode_audio(content=content)
        else:
            raw_audio, sr = decode_audio(url=audio_url)
    except Exception as e:
        raise HTTPException(400, f"Failed to decode audio: {e}")

    audio = prepare_audio(raw_audio, sr)

    try:
        result = await engine.transcribe(
            audio=audio,
            language=language if language else None,
            context=context or "",
            return_timestamps=return_timestamps,
        )
    except Exception as e:
        logger.exception("transcription failed")
        raise HTTPException(500, f"Inference error: {e}")

    resp = {"text": result.text, "language": result.language}
    if return_timestamps and hasattr(result, "time_stamps") and result.time_stamps:
        resp["segments"] = [{
            "start": ts.start_time,
            "end": ts.end_time,
            "text": ts.text,
        } for ts in result.time_stamps]

    return JSONResponse(resp)


# ============================================================ WebSocket /v1/audio/stream

@app.websocket("/v1/audio/stream")
async def audio_stream(websocket: WebSocket):
    """
    Real-time streaming transcription via WebSocket.

    Client sends:
      1. JSON config:  {"language": "Chinese", "sample_rate": 16000}
      2. Binary PCM chunks (16-bit signed LE, mono, 16 kHz)
      3. JSON {"action": "flush"} to force transcription of buffered audio
      4. JSON {"action": "end"} to finalize

    Server sends:
      {"type": "transcript", "text": "...", "language": "...", "is_final": bool}
    """
    await websocket.accept()
    import struct

    buf = bytearray()
    language = None
    chunk_samples = SAMPLE_RATE * 2  # 2 seconds buffer

    try:
        while True:
            msg = await websocket.receive()

            if msg["type"] == "websocket.receive":
                if "text" in msg and msg["text"]:
                    data = json.loads(msg["text"])
                    action = data.get("action")

                    if action == "end":
                        break
                    elif action == "flush":
                        if buf:
                            audio = np.frombuffer(bytes(buf), dtype=np.int16).astype(np.float32) / 32768.0
                            try:
                                result = await engine.transcribe(audio=audio, language=language)
                                await websocket.send_json({
                                    "type": "transcript",
                                    "text": result.text,
                                    "language": result.language,
                                    "is_final": True,
                                })
                            except Exception as e:
                                await websocket.send_json({"type": "error", "message": str(e)})
                            buf.clear()
                    elif "language" in data:
                        language = data["language"]

                elif "bytes" in msg and msg["bytes"]:
                    buf.extend(msg["bytes"])
                    # Auto-flush every 2 seconds of audio
                    if len(buf) >= chunk_samples * 2:  # *2 because int16 = 2 bytes/sample
                        audio = np.frombuffer(bytes(buf[:chunk_samples * 2]), dtype=np.int16).astype(np.float32) / 32768.0
                        buf = buf[chunk_samples * 2:]
                        try:
                            result = await engine.transcribe(audio=audio, language=language)
                            await websocket.send_json({
                                "type": "transcript",
                                "text": result.text,
                                "language": result.language,
                                "is_final": False,
                            })
                        except Exception as e:
                            await websocket.send_json({"type": "error", "message": str(e)})

    except Exception as e:
        logger.exception("streaming error")
    finally:
        await websocket.close()


# ============================================================ main

if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    uvicorn.run(
        "server:app",
        host=config.settings.host,
        port=config.settings.port,
        workers=config.settings.workers,
        log_level=config.settings.log_level.lower(),
    )
