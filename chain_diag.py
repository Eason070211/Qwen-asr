"""One-shot full-chain diagnosis: mic -> level -> normalise -> server -> text.

Records one utterance and walks the entire path the tool uses, printing what
happens at every stage, so it is obvious which stage loses the words.

Run from the project directory:
    & D:\Anaconda\envs\yolo-asr\python.exe chain_diag.py
"""
import io
import os
import sys
import time

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RATE = 16000
DEVICE = 16
SECONDS = 6.0
URL = "http://127.0.0.1:8910/v1/audio/transcriptions"


def db(x):
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    return 20 * np.log10(max(r, 1e-9))


print(f"=== 用设备 [{DEVICE}] 录 {SECONDS:.0f} 秒，请说一句完整的话 ===", flush=True)
frames = []
with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", device=DEVICE,
                    callback=lambda i, n, t, s: frames.append(i[:, 0].copy())):
    for left in range(int(SECONDS), 0, -1):
        print(f"   {left}...", flush=True)
        time.sleep(1.0)
audio = np.concatenate(frames)

print("\n--- 第 1 环：麦克风原始信号 ---")
print(f"  整段 RMS      {db(audio):7.1f} dBFS")
print(f"  峰值样本      {float(np.abs(audio).max()):7.4f}")
w = RATE // 4
prof = np.array([db(audio[i:i + w]) for i in range(0, audio.size, w)])
print(f"  0.25s 电平    min {prof.min():.1f} / 中位 {np.median(prof):.1f} / max {prof.max():.1f} dBFS")
print(f"  帧数          {audio.size}（应为 {int(SECONDS * RATE)}）")

spec = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
freqs = np.fft.rfftfreq(audio.size, 1 / RATE)
print(f"  频谱重心      {float((spec * freqs).sum() / (spec.sum() + 1e-12)):.0f} Hz")
top = np.argsort(spec)[-5:][::-1]
print(f"  最强频率分量  {[int(freqs[i]) for i in top]} Hz")

print("\n--- 第 2 环：工具的切段 ---")
import voice_input as vi
seg = vi.SilenceSegmenter(silence_db=-45., speech_db=-38., silence_hold=0.65,
                          min_speech=0.30, preroll=0.35, max_segment=30., min_segment=0.35,
                          adaptive=True)
seg.reset()
segs = []
for i in range(0, audio.size, 320):
    got = seg.feed(audio[i:i + 320])
    if got is not None:
        segs.append(got)
tail = seg.flush()
if tail is not None:
    segs.append(tail)
print(f"  噪声底 {seg.floor_db():.1f} dBFS，门限 {tuple(round(x,1) for x in seg.thresholds())}")
print(f"  切出 {len(segs)} 段" + ("" if segs else "   <-- 这里就丢了：一句完整的话应该至少切出 1 段"))

print("\n--- 第 3 环：归一化 ---")
for s in segs:
    gain_audio, gain = vi.normalize_audio(s.audio)
    print(f"  #{s.index} {s.seconds:.2f}s  {db(s.audio):6.1f} -> {db(gain_audio):6.1f} dBFS "
          f"(+{gain:.1f} dB)")

print("\n--- 第 4 环：服务器识别 ---")
for s in segs:
    for label, aud in (("原始未归一化", s.audio),
                       ("归一化后", vi.normalize_audio(s.audio)[0])):
        for lang in ("Chinese", None):
            buf = io.BytesIO()
            sf.write(buf, aud, RATE, format="WAV", subtype="PCM_16")
            data = {"model": "qwen3-asr-0.6b"}
            if lang:
                data["language"] = lang
            t0 = time.time()
            try:
                r = requests.post(URL, files={"file": ("a.wav", buf.getvalue(), "audio/wav")},
                                  data=data, timeout=300)
                out = r.json().get("text", "")
            except Exception as e:
                out = f"<失败 {e}>"
            print(f"  #{s.index} {label:14} lang={str(lang):8} {time.time()-t0:4.1f}s -> {out!r}")

print("\n--- 第 5 环：工具的完整调用（就是你按热键时走的路径）---")
client = vi.ASRClient(timeout=300)
for s in segs:
    print(f"  #{s.index} -> {client.transcribe(s.audio, model='qwen3-asr-0.6b', language=None)!r}")
n, lo, hi = client.gain_stats()
print(f"  本次补电平 {n} 次，范围 +{lo:.1f}~+{hi:.1f} dB；语言回退 {client.fallbacks} 次")
