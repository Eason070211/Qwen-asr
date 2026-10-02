"""
Qwen3-ASR local voice input tool (global hotkey).

Hold the hotkey to record, release to transcribe:
  1. text is copied to the clipboard
  2. with --paste it is also typed into the focused app (simulated Ctrl+V)

Requires the Qwen3-ASR server to be running (see start_voice_input.bat).

Usage:
    python voice_input.py                          # default hotkey: ctrl+alt+space
    python voice_input.py --hotkey "ctrl+alt+f1"
    python voice_input.py --language Chinese       # force language (default: auto-detect)
    python voice_input.py --paste                  # auto-paste after transcribing
    python voice_input.py --test-file speech.wav   # offline test: transcribe a file, no hotkey
"""

from __future__ import annotations

import argparse
import io
import threading
import time
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

SERVER_BASE = "http://127.0.0.1:8910"
TRANSCRIBE_URL = SERVER_BASE + "/v1/audio/transcriptions"
HEALTH_URL = SERVER_BASE + "/health"
RATE = 16000

_state = {"recording": False}
_frames: list[np.ndarray] = []
_stream = None
_lock = threading.Lock()


def beep(freq: int, dur: int) -> None:
    try:
        import winsound
        winsound.Beep(freq, dur)
    except Exception:
        pass


def wait_for_server(timeout: int = 360) -> bool:
    """Block until the ASR server reports status=ok (model loaded)."""
    print("[server] 等待服务器就绪（首次需加载模型，约 1-2 分钟）", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(HEALTH_URL, timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print("\n[server] 就绪: " + SERVER_BASE, flush=True)
                return True
        except requests.RequestException:
            pass
        print(".", end="", flush=True)
        time.sleep(5)
    print("\n[server] 等待超时：请确认 server.py 已启动且模型加载成功", flush=True)
    return False


def transcribe(audio: np.ndarray, args) -> str | None:
    if len(audio) == 0:
        return ""
    buf = io.BytesIO()
    sf.write(buf, audio, RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    data = {"model": args.model}
    if args.language:
        data["language"] = args.language
    try:
        r = requests.post(
            TRANSCRIBE_URL,
            files={"file": ("speech.wav", buf, "audio/wav")},
            data=data,
            timeout=300,
        )
        r.raise_for_status()
        return r.json().get("text", "")
    except Exception as e:
        print(f"[ERR] 转写请求失败: {e}", flush=True)
        return None


def audio_callback(indata, frames, time_info, status):
    if _state["recording"]:
        _frames.append(indata.copy())


def start_recording() -> None:
    global _stream
    with _lock:
        if _state["recording"]:
            return
        _frames.clear()
        _state["recording"] = True
    beep(1200, 100)
    print("[REC] 录音中…（松开快捷键结束）", flush=True)
    try:
        _stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", callback=audio_callback)
        _stream.start()
    except Exception as e:
        print(f"[ERR] 无法打开麦克风: {e}", flush=True)
        with _lock:
            _state["recording"] = False


def stop_recording(args) -> None:
    global _stream
    with _lock:
        if not _state["recording"]:
            return
        _state["recording"] = False
    if _stream is not None:
        try:
            _stream.stop()
            _stream.close()
        except Exception:
            pass
        _stream = None
    beep(1600, 100)
    audio = np.concatenate(_frames) if _frames else np.zeros(0, dtype=np.float32)
    print(f"[DONE] 录音 {len(audio) / RATE:.1f}s，正在转写…", flush=True)

    text = transcribe(audio, args)
    if text is None:
        print("[FAIL] 转写失败", flush=True)
        return
    if text.strip():
        import pyperclip
        pyperclip.copy(text)
        print(f"[TEXT] {text}", flush=True)
        if args.paste:
            time.sleep(0.2)
            try:
                import keyboard as kb
                kb.press_and_release("ctrl+v")
                print("[PASTE] 已粘贴到当前窗口", flush=True)
            except Exception as e:
                print(f"[WARN] 自动粘贴失败（请手动 Ctrl+V）: {e}", flush=True)
    else:
        print("[TEXT] （未识别到语音）", flush=True)


def build_hotkey_handler(trigger: str, mods: list[str], args):
    import keyboard as kb

    def handler(e):
        if e.name != trigger:
            return
        if e.event_type == "down":
            if all(kb.is_pressed(m) for m in mods):
                start_recording()
        elif e.event_type == "up" and _state["recording"]:
            threading.Thread(target=stop_recording, args=(args,), daemon=True).start()

    return handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-ASR local voice input (global hotkey)")
    parser.add_argument("--hotkey", default="right ctrl",
                        help="hotkey to hold while speaking, e.g. 'right ctrl' (default) or 'ctrl+alt+f1'")
    parser.add_argument("--language", default=None, help="force language, e.g. Chinese (default: auto-detect)")
    parser.add_argument("--model", default="qwen3-asr-0.6b", help="model id reported to the server")
    parser.add_argument("--paste", action="store_true", help="auto-paste (Ctrl+V) into the focused app")
    parser.add_argument("--test-file", type=str, default=None,
                        help="transcribe a local audio file and exit (no hotkey)")
    args = parser.parse_args()

    # offline test mode
    if args.test_file:
        audio, sr = sf.read(args.test_file, dtype="float32", always_2d=False)
        if sr != RATE:
            print(f"[WARN] 输入采样率 {sr}Hz，将重采样到 {RATE}Hz", flush=True)
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=RATE)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        audio = audio.astype(np.float32)
        print(f"[TEST] 文件 {args.test_file}，时长 {len(audio) / RATE:.1f}s，转写中…", flush=True)
        text = transcribe(audio, args)
        print(f"[RESULT] {text}" if text is not None else "[RESULT] 失败", flush=True)
        return

    if not wait_for_server():
        sys_exit(1)

    # microphone sanity check
    try:
        devs = sd.query_devices()
        inputs = [d for d in devs if d.get("max_input_channels", 0) > 0]
        if not inputs:
            print("[WARN] 未检测到麦克风输入设备", flush=True)
    except Exception:
        pass

    parts = [p.strip().lower() for p in args.hotkey.split("+")]
    trigger = parts[-1]
    mods = parts[:-1]

    import keyboard as kb
    kb.hook(build_hotkey_handler(trigger, mods, args))
    print("=" * 52)
    print(f"  按住 [{args.hotkey}] 说话，松开自动转写")
    print("  （右 Ctrl 单键即可；想换别的可用 --hotkey，如 ctrl+alt+f1）")
    print("  结果自动复制到剪贴板" + ("，并粘贴到当前窗口" if args.paste else ""))
    print("  退出：关闭本窗口 / Ctrl+C")
    print("=" * 52, flush=True)
    threading.Event().wait()


def sys_exit(code: int) -> None:
    import sys
    sys.exit(code)


if __name__ == "__main__":
    main()
