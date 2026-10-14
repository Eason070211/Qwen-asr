"""Tests for the voice_input safety net: retry, transcript log, failed audio,
console title and cancel-current-segment.

Run from the project directory:
    D:\\Anaconda\\envs\\yolo-asr\\python.exe test_voice_safety.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Scratch goes under the project rather than the system temp dir: that temp dir
# refuses subdirectory creation in some sandboxes, which has nothing to do with
# the code under test.
BUILD = os.path.join(HERE, ".build_tmp")
shutil.rmtree(BUILD, ignore_errors=True)
os.makedirs(BUILD, exist_ok=True)
import voice_input as vi  # noqa: E402

RATE = vi.RATE
rng = np.random.default_rng(7)
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def speech(seconds: float, level_db: float = -20.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    env = 0.5 * (1 + np.sin(2 * np.pi * 2.5 * t))
    sig = env * (np.sin(2 * np.pi * 140 * t) + 0.5 * np.sin(2 * np.pi * 280 * t))
    sig = sig / (np.sqrt(np.mean(sig ** 2)) + 1e-12) * 10 ** (level_db / 20)
    return sig.astype(np.float32)


def make_args(tmp: str, **over) -> types.SimpleNamespace:
    base = dict(
        model="qwen3-asr-0.6b", language=None, paste=False, restore_clipboard=False,
        max_seconds=300, vad=True, adaptive_noise=False, silence_db=-45.0,
        speech_db=-38.0, silence_hold=0.65, min_speech=0.30, preroll=0.35,
        max_segment=30.0, min_segment=0.35, queue_workers=1, queue_max=64,
        backlog_warn=5, retry=2, log_transcripts=True, status_title=True,
        cancel_hotkey="",
        transcripts_dir=os.path.join(tmp, "transcripts"),
        failed_audio_dir=os.path.join(tmp, "failed_audio"),
        hotkey="right ctrl",
    )
    base.update(over)
    return types.SimpleNamespace(**base)


tmp = os.path.join(BUILD, "safety")
os.makedirs(tmp, exist_ok=True)
print("=" * 70)
print("TEST A - ResultStore: appends text, rotates by size")
print("=" * 70)
store = vi.ResultStore(os.path.join(tmp, "transcripts"), enabled=True, max_bytes=400, keep=2)
paths = set()
for i in range(12):
    p = store.append(f"第{i}段识别文字" * 4, index=i, seconds=1.5 + i, reason="silence")
    if p:
        paths.add(p)
files = sorted(os.listdir(os.path.join(tmp, "transcripts")))
print("  files:", files)
check("store wrote something", store.written == 12, str(store.written))
check("log rotated rather than growing without bound",
      any(f.endswith(".old") for f in files), str(files))
check("rotated copies are capped (keep=2)", len(files) <= 3, str(files))
day_file = [f for f in files if not f.endswith(".old")][0]
content = open(os.path.join(tmp, "transcripts", day_file), encoding="utf-8").read()
check("entries carry timestamp, index and duration",
      "[20" in content and "#11" in content and "s silence" in content,
      content.splitlines()[0] if content else "(empty)")
check("disabled store writes nothing",
      vi.ResultStore(os.path.join(tmp, "off"), enabled=False).append("x", index=0,
                                                                    seconds=1, reason="t") is None)

print()
print("=" * 70)
print("TEST B - retry: transient failures are retried, success wins")
print("=" * 70)


class FlakyClient:
    """Fails the first `fail_times` calls, then succeeds."""
    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0
        self.lock = threading.Lock()

    def transcribe(self, audio, model=None, language=None):
        with self.lock:
            self.calls += 1
            n = self.calls
        if n <= self.fail_times:
            raise ConnectionError(f"模拟服务器断开 #{n}")
        return "重试后成功"


args = make_args(tmp, retry=2)
client = FlakyClient(fail_times=2)
rec = vi.Recorder(args, client)
rec.queue.start()
rec.enqueue(vi.Segment(blocks=[speech(1.5)], index=0, reason="silence",
                       started=0.0, ended=1.5, peak_db=-20.0, frames=int(1.5 * RATE)))
check("queue drained", rec.queue.wait_idle(timeout=30))
check("3 attempts made (1 + 2 retries)", client.calls == 3, str(client.calls))
check("result accepted after retries", rec.stats["done"] == 1, str(rec.stats))
check("retry counted", rec.stats["retried"] == 2, str(rec.stats))
check("no failure recorded", rec.stats["failed"] == 0, str(rec.stats))
rec.queue.shutdown()

print()
print("=" * 70)
print("TEST C - permanent failure keeps the audio for later")
print("=" * 70)


class DeadClient:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, model=None, language=None):
        self.calls += 1
        raise ConnectionRefusedError("服务器没开")


args_c = make_args(tmp, retry=1)
# Fresh, isolated directories: reusing another test's would make the assertions
# inspect leftovers from a previous run.
args_c.transcripts_dir = os.path.join(tmp, "transcripts_c")
args_c.failed_audio_dir = os.path.join(tmp, "failed_audio_c")
dead = DeadClient()
rec_c = vi.Recorder(args_c, dead)
rec_c.queue.start()
payload = speech(2.0)
rec_c.enqueue(vi.Segment(blocks=[payload], index=7, reason="silence",
                         started=0.0, ended=2.0, peak_db=-20.0, frames=payload.size))
check("queue drained", rec_c.queue.wait_idle(timeout=30))
check("attempts = 1 + 1 retry", dead.calls == 2, str(dead.calls))
check("failure counted", rec_c.stats["failed"] == 1, str(rec_c.stats))
fa_dir = args_c.failed_audio_dir
wavs = [f for f in os.listdir(fa_dir)] if os.path.isdir(fa_dir) else []
print("  failed_audio:", wavs)
check("audio saved as .wav", any(f.endswith(".wav") for f in wavs), str(wavs))
check("companion .txt explains the failure",
      any(f.endswith(".txt") for f in wavs) and
      "服务器没开" in open(os.path.join(fa_dir, [f for f in wavs if f.endswith('.txt')][0]),
                        encoding="utf-8").read(),
      str(wavs))
if any(f.endswith(".wav") for f in wavs):
    import soundfile as sf
    w = os.path.join(fa_dir, [f for f in wavs if f.endswith(".wav")][0])
    a, sr = sf.read(w, dtype="float32")
    check("saved audio is intact (2.0s @16k)",
          abs(len(a) / sr - 2.0) < 0.05 and sr == RATE, f"{len(a) / sr:.2f}s @ {sr}Hz")
check("nothing was written to the transcript log on failure",
      not os.path.isdir(args_c.transcripts_dir) or
      not os.listdir(args_c.transcripts_dir),
      str(os.listdir(args_c.transcripts_dir)) if os.path.isdir(args_c.transcripts_dir) else "no dir")
rec_c.queue.shutdown()

print()
print("=" * 70)
print("TEST D - console title reflects state")
print("=" * 70)
args_d = make_args(tmp)
rec_d = vi.Recorder(args_d, vi.ASRClient())
rec_d.gate = False
t_idle = rec_d.title_text()
rec_d.gate = True
t_rec = rec_d.title_text()
rec_d.stats.update(done=3, failed=1, cancelled=2)
rec_d.gate = False
t_mixed = rec_d.title_text()
print("  idle  :", t_idle)
print("  rec   :", t_rec)
print("  mixed :", t_mixed)
check("idle title says 待机", "待机" in t_idle, t_idle)
check("recording title says 录音中", "录音中" in t_rec, t_rec)
check("title shows the hotkey", "right ctrl" in t_idle, t_idle)
check("counters appear when non-zero",
      "完成 3" in t_mixed and "失败 1" in t_mixed and "取消 2" in t_mixed, t_mixed)
check("title disabled writes nothing (no crash)",
      vi.Recorder(make_args(tmp, status_title=False), vi.ASRClient())._refresh_title() is None)

print()
print("=" * 70)
print("TEST E - cancel drops the current chunk, keeps the queue")
print("=" * 70)
args_e = make_args(tmp)
rec_e = vi.Recorder(args_e, vi.ASRClient())
seg = vi.SilenceSegmenter(silence_db=-45.0, speech_db=-38.0, silence_hold=0.65,
                          min_speech=0.30, preroll=0.35, max_segment=30.0,
                          min_segment=0.35, adaptive=False)
seg.reset()
rec_e.seg = seg
rec_e.gate = True
# record 1.2s of speech, then cancel it
for i in range(0, int(1.2 * RATE), 320):
    seg.feed(speech(0.02)[:320])
buffered_before = seg._buf_frames / RATE
check("something is buffered before cancel", buffered_before > 1.0, f"{buffered_before:.2f}s")
check("cancel reports what it dropped", rec_e.cancel_current())
check("buffer is empty after cancel", seg._buf_frames == 0, str(seg._buf_frames))
check("cancel counted", rec_e.stats["cancelled"] == 1, str(rec_e.stats))
check("no job was queued by cancelling", rec_e.queue.pending() == 0,
      str(rec_e.queue.pending()))
check("cancel with nothing recording returns False",
      vi.Recorder(make_args(tmp), vi.ASRClient()).cancel_current() is False)
# the queue must be untouched: cancel only affects the live chunk
rec_e.enqueue(vi.Segment(blocks=[speech(1.0)], index=0, reason="silence",
                         started=0.0, ended=1.0, peak_db=-20.0, frames=RATE))
before = rec_e.queue.pending()
rec_e.cancel_current()
check("cancel does not remove already-queued jobs",
      rec_e.queue.pending() == before, f"{before} -> {rec_e.queue.pending()}")

print()
print("=" * 70)
print("TEST F - hotkey wiring for the cancel key")
print("=" * 70)
seen = {"cancel": 0, "stop": 0}
rec_f = vi.Recorder(make_args(tmp, cancel_hotkey="right alt"), vi.ASRClient())
rec_f.cancel_current = lambda: seen.__setitem__("cancel", seen["cancel"] + 1) or True
rec_f.stop = lambda auto=False: seen.__setitem__("stop", seen["stop"] + 1) or True
handler = vi.build_hotkey_handler("right ctrl", [], rec_f, "right alt")


class Ev:
    def __init__(self, name, event_type):
        self.name = name
        self.event_type = event_type


handler(Ev("right alt", "down"))
handler(Ev("right alt", "down"))          # auto-repeat must not double-fire
handler(Ev("right alt", "up"))
check("cancel fires once per press", seen["cancel"] == 1, str(seen))
handler(Ev("right ctrl", "down"))
handler(Ev("right ctrl", "up"))
check("main hotkey still works", seen["stop"] == 1, str(seen))
check("both keys are independent", seen == {"cancel": 1, "stop": 1}, str(seen))

print()
print("=" * 70)
print("TEST G - level normalisation (input devices with automatic gain control)")
print("=" * 70)
import voice_input  # noqa: E402  (already imported as vi; kept explicit)


def dbfs(x):
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    return 20 * np.log10(max(r, 1e-9))


def tone(seconds=1.0, db=-45.0):
    t = np.arange(int(seconds * RATE)) / RATE
    s = np.sin(2 * np.pi * 300 * t)
    return (s / np.sqrt(np.mean(s ** 2)) * 10 ** (db / 20)).astype(np.float32)


# This machine's AMD array drifts over ~33 dB, and the model returns "嗯。" or
# nothing for the quiet end, so quiet chunks must be lifted.
quiet = tone(1.0, -48.0)
norm, gain = vi.normalize_audio(quiet, -22.0)
print(f"  quiet  {-48.0:6.1f} -> {dbfs(norm):6.1f} dBFS  (+{gain:.1f} dB)")
check("quiet audio is lifted to the target",
      abs(dbfs(norm) - (-22.0)) < 1.0, f"{dbfs(norm):.1f} dBFS")
check("gain is positive for quiet audio", gain > 20, f"+{gain:.1f} dB")

# Loud audio is left alone: boosting it would only add distortion.
loud = tone(1.0, -18.0)
norm_l, gain_l = vi.normalize_audio(loud, -22.0)
check("already-loud audio is NOT boosted", gain_l == 0.0, f"+{gain_l:.1f} dB")

# A very quiet noisy chunk must not be pushed into clipping.
noise = (rng.normal(0, 1, RATE) * 10 ** (-46 / 20)).astype(np.float32)
norm_n, gain_n = vi.normalize_audio(noise, -22.0)
peak_n = float(np.max(np.abs(norm_n)))
print(f"  noise  {-46.0:6.1f} -> {dbfs(norm_n):6.1f} dBFS  (+{gain_n:.1f} dB) "
      f"peak={peak_n:.3f}")
check("normalisation never clips", peak_n <= 1.0, f"peak={peak_n:.3f}")

# Silence must be left untouched (no gain on nothing).
sil, gain_s = vi.normalize_audio(np.zeros(RATE, np.float32), -22.0)
check("silence is not amplified", gain_s == 0.0 and float(np.abs(sil).max()) == 0.0)

# An existing loud peak must cap the gain: the RMS wants a big boost, but the
# peak only leaves a little headroom, so the applied gain is limited by the peak
# (a sine-based signal alone has ~3 dB crest factor and cannot show this).
loudish = tone(1.0, -40.0).copy()
loudish[0] = 0.30                     # a transient ~9 dB below full scale
norm_c, gain_c = vi.normalize_audio(loudish, -22.0)
peak_norm = float(np.max(np.abs(norm_c)))
print(f"  peaky  {dbfs(loudish):6.1f} -> {dbfs(norm_c):6.1f} dBFS  "
      f"(+{gain_c:.1f} dB) peak={peak_norm:.3f}")
check("gain is capped so a loud peak cannot clip", peak_norm <= 1.0,
      f"peak={peak_norm:.3f}")
check("the peak cap allows only the available headroom",
      1.0 < gain_c <= 20 * np.log10(1.0 / 0.30) + 0.1, f"+{gain_c:.1f} dB")
check("the signal is still lifted as far as the peak allows",
      dbfs(norm_c) > dbfs(loudish) + 5, f"{dbfs(loudish):.1f} -> {dbfs(norm_c):.1f}")

print()
print("=" * 70)
print("TEST H - language fallback when auto-detect returns nothing")
print("=" * 70)


class RecordingClient(vi.ASRClient):
    """Feeds canned answers and records what language each request used."""
    def __init__(self, answers):
        super().__init__(auto_gain=False, fallback_language="Chinese")
        self.answers = list(answers)
        self.seen = []

    def _post(self, audio, model, language):
        self.seen.append(language)
        return self.answers.pop(0) if self.answers else ""


c = RecordingClient(["", "现在几点"])
out = c.transcribe(tone(1.0, -20.0), model="m", language=None)
check("empty auto-detect result triggers a retry", c.seen == [None, "Chinese"], str(c.seen))
check("the retry's text is returned", out == "现在几点", repr(out))
check("fallback counted", c.fallbacks == 1, str(c.fallbacks))

c2 = RecordingClient(["正常文字"])
out2 = c2.transcribe(tone(1.0, -20.0), model="m", language=None)
check("a non-empty result is NOT retried", c2.seen == [None], str(c2.seen))
check("text passes through", out2 == "正常文字", repr(out2))

c3 = RecordingClient(["", ""])
c3.transcribe(tone(1.0, -20.0), model="m", language="Chinese")
check("no retry when the language was already explicit",
      c3.seen == ["Chinese"], str(c3.seen))

c4 = RecordingClient(["", "你好"])
c4.fallback_language = None
c4.transcribe(tone(1.0, -20.0), model="m", language=None)
check("fallback can be disabled", c4.seen == [None], str(c4.seen))

print()
print("=" * 70)
print("RESULT:", "ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILURE(S): {FAILURES}")
print("=" * 70)
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(1 if FAILURES else 0)
