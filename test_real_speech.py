"""Synthesise real speech with SAPI and verify the whole VAD, queue, ASR chain.

Produces a WAV with three Chinese utterances separated by real silence (each
utterance is rendered to its own file, then stitched with inserted pauses —
SAPI's synchronous Speak would otherwise write three sentences back-to-back
with no gap for the VAD to split on), then:
  1. runs the real SilenceSegmenter over it,
  2. checks the pause structure was recovered,
  3. transcribes each segment through the live server,
  4. checks the words of each sentence landed in the right segment.

Run from the project directory (the ASR server must be up):
    & D:\Anaconda\envs\yolo-asr\python.exe test_real_speech.py
"""
from __future__ import annotations

import difflib
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import voice_input as vi  # noqa: E402

RATE = vi.RATE
HERE = os.path.dirname(os.path.abspath(__file__))
WAV = os.path.join(HERE, "_speech_test.wav")
PS1 = os.path.join(HERE, "_speak.ps1")
PS = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"

# (phrase, pause_after_seconds). Chinese: this machine's SAPI has Huihui
# (zh-CN), and the tool is used for Chinese input, so test the real scenario.
# Each phrase is rendered to its own WAV; the pauses below are inserted as real
# silence during stitching.
SCRIPT = [
    ("今天天气真不错", 1.5),
    ("我们去公园走一走吧", 1.5),
    ("记得把报告发给我", 0.8),
]

# Written as UTF-8 *with BOM*: Windows PowerShell 5.1 reads a BOM-less file as
# ANSI, which would mangle the Chinese text.
per_phrase_wavs = [os.path.join(HERE, f"_speech_test_{i}.wav") for i in range(len(SCRIPT))]
lines = [
    "Add-Type -AssemblyName System.Speech",
    "$ErrorActionPreference = 'Stop'",
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
    "$zh = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'zh*' } | Select-Object -First 1",
    "$picked = $null",
    "foreach ($n in @($zh.VoiceInfo.Name, 'Microsoft Huihui Desktop', 'Microsoft Zira Desktop', 'Microsoft Hazel Desktop')) {",
    "  if (-not $n) { continue }",
    "  try { $s.SelectVoice($n); $picked = $n; break } catch { }",
    "}",
    "if (-not $picked) { throw 'no usable SAPI voice' }",
    "Write-Output \"voice=$picked\"",
    "$s.Rate = -1",
]
for i, (text, _pause) in enumerate(SCRIPT):
    lines.append(f"$s.SetOutputToWaveFile('{per_phrase_wavs[i]}')")
    lines.append(f"$s.Speak('{text}')")
lines += ["$s.Dispose()", "Write-Output 'SAPI_OK'"]
with open(PS1, "w", encoding="utf-8-sig") as fh:
    fh.write("\n".join(lines) + "\n")

print("--- synthesising speech with SAPI ---")
for w in [WAV] + per_phrase_wavs:
    if os.path.exists(w):
        os.remove(w)
res = subprocess.run([PS, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", PS1],
                     capture_output=True, encoding="utf-8", errors="replace")
out = ((res.stdout or "") + (res.stderr or "")).strip()
print(out[-600:])

if not all(os.path.exists(w) for w in per_phrase_wavs):
    print("SAPI produced no WAV; aborting")
    sys.exit(1)


def _load_mono(path: str) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=-1)
    if sr != RATE:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=RATE).astype(np.float32)
    return a


parts = []
for i, (text, pause) in enumerate(SCRIPT):
    parts.append(_load_mono(per_phrase_wavs[i]))
    parts.append(np.zeros(int(pause * RATE), dtype=np.float32))  # real silence
audio = np.concatenate(parts)
print(f"--- stitched {len(audio) / RATE:.2f}s @ {RATE}Hz "
      f"({len(SCRIPT)} phrases + inserted pauses) ---")

# For captured speech the floor is measured while nobody speaks; emulate that
# by priming the estimator with the file's own quietest stretch.
script_quiet = float(np.percentile([vi.dbfs(audio[i:i + 320])
                                    for i in range(0, audio.size, 320)], 10))

seg = vi.SilenceSegmenter(silence_db=-45.0, speech_db=-38.0, silence_hold=0.65,
                          min_speech=0.30, preroll=0.35, max_segment=30.0,
                          min_segment=0.35, adaptive=True)
seg.reset()
clock = [1000.0]
real_time = vi.time.time
vi.time.time = lambda: clock[0]

segs = []
step = 320
for i in range(0, audio.size, step):
    clock[0] += step / RATE
    out = seg.feed(audio[i:i + step])
    if out is not None:
        segs.append(out)
tail = seg.flush()
if tail is not None:
    segs.append(tail)
vi.time.time = real_time

print(f"--- noise floor settled at {seg.floor_db():.1f} dBFS, "
      f"gates {seg.thresholds()} ---")
print(f"--- {len(segs)} segments ---")
for s in segs:
    print(f"    #{s.index} {s.seconds:5.2f}s  speech {s.started:5.2f}s -> {s.ended:5.2f}s"
          f"  peak {s.peak_db:5.1f} dBFS  [{s.reason}]")

failures = []
if len(segs) != len(SCRIPT):
    failures.append(f"expected {len(SCRIPT)} segments, got {len(segs)}")

client = vi.ASRClient(timeout=300)
if not client.wait_ready(timeout=120):
    print("server not ready")
    sys.exit(1)

print("--- transcribing each segment through the live server ---")
for want, s in zip(SCRIPT, segs):
    text = client.transcribe(s.audio, model="qwen3-asr-0.6b", language="Chinese")
    got = "".join(ch for ch in (text or "") if ch.isalnum())
    exp = "".join(ch for ch in want[0] if ch.isalnum())
    ratio = difflib.SequenceMatcher(None, exp, got).ratio()
    print(f"    #{s.index} expect≈{want[0]!r} -> got {text!r}  (sim {ratio:.2f})")
    if not got:
        failures.append(f"segment {s.index}: empty transcription")
        continue
    if ratio < 0.4:
        failures.append(f"segment {s.index}: similarity only {ratio:.2f}")

print()
print("RESULT:", "ALL PASS" if not failures else f"FAILURES: {failures}")
sys.exit(1 if failures else 0)
