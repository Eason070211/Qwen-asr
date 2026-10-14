"""Check which input device actually captures your voice at a usable level.

    D:\\Anaconda\\envs\\yolo-asr\\python.exe check_mic_levels.py

Speaks are recorded from every real (non-virtual) input device in turn and both
the measured level and what the ASR server makes of it are printed. Use it when
transcription keeps coming back empty: on this machine the AMD microphone array
has automatic gain control that drifts over ~33 dB, and one of its entries
captured 20 dB quieter than the other.

Options:
    --seconds 3        recording length per device
    --devices 16 1     test only these indices
    --skip-server      only measure levels, do not call the ASR server
"""
from __future__ import annotations

import argparse
import io
import time

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

RATE = 16000
SERVER = "http://127.0.0.1:8910"
VIRTUAL_HINTS = ("voicemeeter", "cable", "mapper", "point", "senary", "wave microphone")


def dbfs(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    return 20 * np.log10(max(r, 1e-9))


def candidate_devices(limit: int) -> list[int]:
    out = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        name = dev.get("name", "").lower()
        if any(h in name for h in VIRTUAL_HINTS):
            continue
        out.append(idx)
    return out[:limit]


def record(idx: int, seconds: float) -> np.ndarray:
    frames: list[np.ndarray] = []
    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", device=idx,
                        callback=lambda i, n, t, s: frames.append(i[:, 0].copy())):
        for left in range(int(seconds), 0, -1):
            print(f"      {left}...", flush=True)
            time.sleep(1.0)
    return np.concatenate(frames) if frames else np.zeros(0, np.float32)


def ask_server(audio: np.ndarray, language: str = "Chinese") -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, RATE, format="WAV", subtype="PCM_16")
    r = requests.post(f"{SERVER}/v1/audio/transcriptions",
                      files={"file": ("d.wav", buf.getvalue(), "audio/wav")},
                      data={"model": "qwen3-asr-0.6b", "language": language},
                      timeout=300)
    r.raise_for_status()
    return r.json().get("text", "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--devices", type=int, nargs="*", default=None)
    ap.add_argument("--skip-server", action="store_true")
    args = ap.parse_args()

    indices = args.devices if args.devices else candidate_devices(4)
    print(f"将测试 {len(indices)} 个设备，每个 {args.seconds:.0f} 秒。")
    print("每次提示后请对着麦克风说同一句话，例如：今天天气不错。\n")

    rows = []
    for idx in indices:
        try:
            name = sd.query_devices(idx)["name"]
        except Exception:
            print(f"[{idx}] 无效设备，跳过")
            continue
        print(f"--- [{idx}] {name[:56]} ---")
        try:
            audio = record(idx, args.seconds)
        except Exception as e:
            print(f"    打不开: {type(e).__name__}: {e}\n")
            continue
        if audio.size == 0:
            print("    没采到数据\n")
            continue
        win = RATE // 4
        prof = np.array([dbfs(audio[i:i + win]) for i in range(0, audio.size, win)])
        median = float(np.median(prof))
        text = "(跳过)"
        if not args.skip_server:
            try:
                text = ask_server(audio)
            except Exception as e:
                text = f"<服务器不可用: {e}>"
        print(f"    整段 {dbfs(audio):6.1f} dBFS | 0.25s 中位 {median:6.1f} "
              f"最大 {prof.max():6.1f} | 峰值样本 {np.abs(audio).max():.4f}")
        print(f"    识别: {text!r}\n")
        rows.append((idx, name, dbfs(audio), median, text))

    print("=" * 78)
    print("汇总（按电平从高到低）—— 选电平高、并且识别出文字的那个：")
    for idx, name, whole, median, text in sorted(rows, key=lambda r: -r[3]):
        print(f"  [{idx:>3}] {name[:48]:48} 中位 {median:6.1f} dBFS  -> {text!r}")
    if rows:
        best = max(rows, key=lambda r: r[3])
        print(f"\n建议：用 --device {best[0]}（或把 config.py 里 voice_device 设为 \"{best[0]}\"）")
        print("注：设备编号在增删音频设备后可能变化，可用 --list-devices 复核。")


if __name__ == "__main__":
    main()
