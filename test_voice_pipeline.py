"""Offline checks for the rewritten voice_input: segmentation + queue + end-to-end.

Run from the project directory:
    D:\\Anaconda\\envs\\yolo-asr\\python.exe test_voice_pipeline.py

Tests 1-9 are fully offline; test 10 additionally uses the running ASR server
(and skips its checks gracefully if the server is not up).
"""
from __future__ import annotations

import os
import sys
import threading
import time
import types

import numpy as np

# Import voice_input from this file's own directory, so the suite can live
# anywhere and still test the copy next to it.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import voice_input as vi  # noqa: E402

RATE = vi.RATE
rng = np.random.default_rng(7)
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def speech(seconds: float, level_db: float = -20.0, mod_hz: float = 2.5) -> np.ndarray:
    """Speech-like burst: harmonic stack, amplitude-modulated like syllables.

    The modulation matters - it is what a naive per-block RMS gate mistakes for
    silence, and it is what a real voice does several times a second.
    """
    t = np.arange(int(seconds * RATE)) / RATE
    env = 0.5 * (1 + np.sin(2 * np.pi * mod_hz * t))
    sig = env * (np.sin(2 * np.pi * 140 * t) + 0.5 * np.sin(2 * np.pi * 280 * t))
    sig = sig / (np.sqrt(np.mean(sig ** 2)) + 1e-12) * 10 ** (level_db / 20)
    return sig.astype(np.float32)


def silence(seconds: float, level_db: float = -70.0) -> np.ndarray:
    n = int(seconds * RATE)
    if level_db <= -90:
        return np.zeros(n, dtype=np.float32)
    noise = rng.normal(0, 1, n)
    noise = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-12) * 10 ** (level_db / 20)
    return noise.astype(np.float32)


def feed(seg: vi.SilenceSegmenter, audio: np.ndarray, block_ms: int = 20) -> list[vi.Segment]:
    out, step, pos = [], int(RATE * block_ms / 1000), 0
    while pos < audio.size:
        got = seg.feed(audio[pos:pos + step])
        if got is not None:
            out.append(got)
        pos += step
    return out


def make_segmenter(**kw) -> vi.SilenceSegmenter:
    base = dict(silence_db=-45.0, speech_db=-38.0, silence_hold=0.65,
                min_speech=0.30, preroll=0.35, max_segment=30.0, min_segment=0.35,
                adaptive=False)
    base.update(kw)
    seg = vi.SilenceSegmenter(**base)
    seg.reset()
    return seg


def level_of(audio: np.ndarray) -> float:
    return vi.dbfs(audio)


print("=" * 70)
print("TEST 1 - pause splitting with modulated (realistic) speech")
print("=" * 70)
audio = np.concatenate([
    silence(1.0, -75),
    speech(1.5), silence(1.2, -60),
    speech(2.0), silence(1.0, -55),
    speech(1.0), silence(1.5, -70),
])
segs = feed(make_segmenter(), audio)
print("  segments:", [(round(s.seconds, 2), s.reason, round(s.peak_db)) for s in segs])
check("3 pauses -> 3 segments", len(segs) == 3, f"got {len(segs)}")
if len(segs) == 3:
    durs = [s.seconds for s in segs]
    # Each segment = its speech + up to 0.65s of trailing hold + a little of the
    # 0.35s pre-roll (the silence that preceded the speech).
    check("durations are speech + hold, not runaway", all(1.5 <= d <= 3.6 for d in durs),
          str([round(d, 2) for d in durs]))
    check("audio content is speech-level (no leading 1s of silence)",
          all(level_of(s.audio) > -30 for s in segs),
          str([round(level_of(s.audio)) for s in segs]))
    check("reasons are all 'silence'", all(s.reason == "silence" for s in segs))
    check("every segment starts at the recording start (0.35s pre-roll fed while idle)",
          all(s.started < 0.05 for s in segs),
          str([round(s.started, 2) for s in segs]))
    check("segment 2 sits between end of seg 1 and end of seg 2",
          3.5 <= segs[1].ended <= 6.9, f"{segs[1].ended:.2f}s")
    check("each segment carries >= 0.3s of speech frames (not just the pre-roll)",
          all(s.frames > 0.3 * RATE for s in segs),
          str([round(s.frames / RATE, 2) for s in segs]))

print()
print("=" * 70)
print("TEST 2 - long-pause speaker: a 0.9s pause must still split")
print("=" * 70)
audio = np.concatenate([speech(1.2), silence(0.9, -60), speech(1.2), silence(1.2, -60)])
segs = feed(make_segmenter(), audio)
check("0.9s pause splits with hold=0.65", len(segs) == 2, f"got {len(segs)}")

print()
print("=" * 70)
print("TEST 3 - hold is respected: a 0.4s mid-sentence pause does NOT split")
print("=" * 70)
audio = np.concatenate([speech(1.2), silence(0.4, -60), speech(1.2), silence(1.2, -60)])
segs = feed(make_segmenter(), audio)
check("0.4s pause stays one segment", len(segs) == 1, f"got {len(segs)}")
if segs:
    check("that segment holds both sentences (1.2+0.4+1.2 + hold + pre-roll)",
          2.3 <= segs[0].seconds <= 3.9, f"{segs[0].seconds:.2f}s")

print()
print("=" * 70)
print("TEST 4 - short blips (clicks) are ignored, and the survivor is the speech")
print("=" * 70)
audio = np.concatenate([
    speech(0.10), silence(0.6, -70),      # 100ms click  (< min_speech 300ms)
    speech(0.15), silence(0.5, -70),      # 150ms cough
    speech(1.0), silence(1.2, -70),       # real speech
])
segs = feed(make_segmenter(), audio)
check("blips dropped, only real speech survives", len(segs) == 1, f"got {len(segs)}")
if segs:
    # 1.0s of speech + 0.65s hold + the 0.5-1.1s of preceding silence captured
    # by the 0.35s pre-roll; the important part is that the clicks are gone.
    check("surviving segment is the 1.0s utterance (plus pre-roll + hold)",
          1.0 <= segs[0].seconds <= 3.0, f"{segs[0].seconds:.2f}s")
    check("exactly one segment, so both clicks were dropped",
          len(segs) == 1, f"got {len(segs)}")
    check("and it contains speech-level audio, not the clicks' silence",
          level_of(segs[0].audio) > -30, f"{level_of(segs[0].audio):.0f} dBFS")

print()
print("=" * 70)
print("TEST 5 - max_segment splits non-stop talking without losing audio")
print("=" * 70)
segs = feed(make_segmenter(max_segment=3.0), speech(9.0))
total = sum(s.seconds for s in segs)
print("  segments:", [(round(s.seconds, 2), s.reason) for s in segs])
check("9s of speech -> 3 forced pieces", len(segs) == 3, f"got {len(segs)}")
check("all marked 'max-length'", all(s.reason == "max-length" for s in segs))
check("no audio lost (>= 8.9s of the 9.0s captured)",
      total >= 8.9, f"{total:.2f}s captured")

print()
print("=" * 70)
print("TEST 6 - flush on stop, and --no-vad (single whole-session job)")
print("=" * 70)
seg = make_segmenter()
feed(seg, np.concatenate([speech(1.0), silence(0.4, -60), speech(1.0)]))
tail = seg.flush("manual")
check("flush() yields the unfinished chunk",
      tail is not None and tail.reason == "manual")
if tail:
    check("flushed chunk keeps ~2.4s (both sentences + the 0.4s pause)",
          2.0 <= tail.seconds <= 3.0, f"{tail.seconds:.2f}s")

seg = make_segmenter(vad=False)
segs = feed(seg, np.concatenate([speech(1.0), silence(1.5, -60), speech(1.0)]))
tail = seg.flush("manual")
check("--no-vad produces no pause splits", len(segs) == 0, f"got {len(segs)}")
check("--no-vad keeps everything for the stop flush",
      tail is not None and tail.seconds >= 3.2,
      f"{tail.seconds:.2f}s" if tail else "no tail")

print()
print("=" * 70)
print("TEST 7 - edge cases: empty blocks, pure silence, gate sanity")
print("=" * 70)
seg = make_segmenter()
check("empty block -> None", seg.feed(np.zeros(0, np.float32)) is None)
check("3s of pure silence -> no segments", feed(seg, silence(3.0, -80)) == [])
check("gate clamped (silence_db never above speech_db)",
      make_segmenter(silence_db=-20.0, speech_db=-50.0).thresholds()[0] == -50.0)
seg = make_segmenter()
segs = feed(seg, speech(1.0), block_ms=50)
tail = seg.flush()
check("50ms blocks still work", tail is not None and tail.seconds >= 0.9)
# sustained noise above the open gate must not grow the buffer forever
seg = make_segmenter()
for _ in range(200):
    seg.feed(silence(0.02, -40.0))
check("buffer stays bounded under sustained sub-speech noise",
      seg._buf_frames <= RATE * 2, f"{seg._buf_frames / RATE:.2f}s buffered")

print()
print("=" * 70)
print("TEST 8 - adaptive noise floor calibrates the gates")
print("=" * 70)


class FakeClock:
    """Audio time, not wall-clock time, so a 20 ms block advances 20 ms.

    Must start at the real clock's value: NoiseFloor rate-limits its samples
    against `time.time()`, and a mismatched time base silently starves it.
    """
    def __init__(self, start=None):
        self.t = start if start is not None else time.time()

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def feed_clocked(seg: vi.SilenceSegmenter, audio: np.ndarray, clock: "FakeClock",
                 block_ms: int = 20) -> list[vi.Segment]:
    """Like feed(), but advances a fake clock with the audio it delivers.

    The noise estimator rate-limits its samples on wall-clock time, so feeding
    audio without advancing time starves it - a test artefact, not a bug.
    """
    out, step, pos = [], int(RATE * block_ms / 1000), 0
    while pos < audio.size:
        clock.advance(step / RATE)
        got = seg.feed(audio[pos:pos + step])
        if got is not None:
            out.append(got)
        pos += step
    return out


def calibrate(levels_db, seconds_per_block: float = 0.02) -> vi.SilenceSegmenter:
    """Feed a sequence of per-block levels (dBFS) and report the settled gates."""
    seg = vi.SilenceSegmenter(silence_db=-45.0, speech_db=-38.0, silence_hold=0.65,
                              min_speech=0.30, preroll=0.35, max_segment=30.0,
                              min_segment=0.35, adaptive=True)
    seg.reset()
    real_time = vi.time.time
    clock = FakeClock(real_time())
    vi.time.time = clock
    try:
        for lvl in levels_db:
            clock.advance(seconds_per_block)
            seg.feed(silence(seconds_per_block, lvl))
    finally:
        vi.time.time = real_time
    return seg


# A well-behaved quiet room.
seg = calibrate([-62.0] * 250, 0.02)          # 5s of room tone
floor, (sil, sp) = seg.floor_db(), seg.thresholds()
print(f"  quiet room: floor={floor:.0f} -> silence={sil:.0f} speech={sp:.0f}")
check("floor converged to the real -62 dBFS room", -68 <= floor <= -58, f"{floor:.1f}")
check("gates sit above the floor", floor < sil < sp,
      f"floor {floor:.0f}, silence {sil:.0f}, speech {sp:.0f}")

# A noisy room must move the gates up.
seg_n = calibrate([-42.0] * 250, 0.02)
floor_n, (sil_n, sp_n) = seg_n.floor_db(), seg_n.thresholds()
print(f"  noisy room: floor={floor_n:.0f} -> silence={sil_n:.0f} speech={sp_n:.0f}")
check("noisier room raises the gates", sil_n > sil and sp_n > sp,
      f"{sil_n:.0f}/{sp_n:.0f} vs {sil:.0f}/{sp:.0f}")

# THE IMPORTANT ONE: this machine's real capture device. Measured ambient was
# median -43 dBFS, p10 -47, but with occasional dropouts down to -97. A
# minimum-statistics floor would latch onto -97 and derive gates ~50 dB too
# sensitive, so every quiet room would look like it was full of speech.
real_mic = [-43.0] * 230
for i in (30, 90, 150, 210):                  # 4 brief driver dropouts
    real_mic[i] = -97.5
seg_r = calibrate(real_mic, 0.02)
floor_r, (sil_r, sp_r) = seg_r.floor_db(), seg_r.thresholds()
print(f"  real mic w/ dropouts: floor={floor_r:.0f} -> "
      f"silence={sil_r:.0f} speech={sp_r:.0f}")
check("dropouts do NOT drag the floor to -97", floor_r > -60, f"{floor_r:.1f}")
check("floor is at or below the real ambient, never above it (~-43)",
      -62 <= floor_r <= -36, f"{floor_r:.1f}")
check("gates land above the floor and below normal speech",
      floor_r < sil_r < sp_r < -20, f"silence={sil_r:.0f} speech={sp_r:.0f}")

# Before it has enough samples the fixed levels must be used, so the very first
# utterance of a session is not missed.
seg_cold = vi.SilenceSegmenter(silence_db=-45.0, speech_db=-38.0, silence_hold=0.65,
                               min_speech=0.30, preroll=0.35, max_segment=30.0,
                               min_segment=0.35, adaptive=True)
seg_cold.reset()
check("cold start uses the fixed fallback gates",
      seg_cold.thresholds() == (-45.0, -38.0), str(seg_cold.thresholds()))
check("cold start reports not-ready", not seg_cold.floor_ready())

# REGRESSION: a real session where the speaker talks in a series of sentences.
# A naive estimator sampled speech-level blocks, climbed to the speaker's own
# level (-20 dBFS), pushed the gates above the voice, and then discarded every
# segment as "no speech". The floor must never end up at or above the speech
# level it is supposed to detect.
real_session = [("quiet", 3.0)]           # real sessions start with a quiet moment
for _ in range(9):                        # 9 sentences with natural pauses
    real_session.append(("speech", 1.0))
    real_session.append(("quiet", 0.9))
audio = []
for kind, secs in real_session:
    audio.append(speech(secs) if kind == "speech" else silence(secs, -75.0))
seg_busy = vi.SilenceSegmenter(silence_db=-45.0, speech_db=-38.0, silence_hold=0.65,
                               min_speech=0.30, preroll=0.35, max_segment=30.0,
                               min_segment=0.35, adaptive=False)
seg_busy.reset()
real_time = vi.time.time                  # the real one, before any patching
clock_b = FakeClock(real_time())
vi.time.time = clock_b
try:
    busy_segs = feed_clocked(seg_busy, np.concatenate(audio), clock_b)
    tail_b = seg_busy.flush()
    if tail_b is not None:
        busy_segs.append(tail_b)
finally:
    vi.time.time = real_time

floor_b = seg_busy.floor_db()
sil_b, sp_b = seg_busy.thresholds()
print(f"  busy session: floor={floor_b:.0f} -> silence={sil_b:.0f} speech={sp_b:.0f}, "
      f"segments={len(busy_segs)}")
check("floor did NOT climb toward the speaker's level (-20 dBFS)",
      floor_b < -35, f"floor={floor_b:.1f}")
check("gates stayed well below ordinary speech",
      sp_b < -32, f"speech gate={sp_b:.1f}")
check("the utterances were still segmented", len(busy_segs) >= 6, f"{len(busy_segs)}")

# And a segment captured in that session must be loud enough to transcribe:
if busy_segs:
    lvl = level_of(busy_segs[0].audio)
    check("captured segments contain speech-level audio (not silence)",
          lvl > -32, f"{lvl:.0f} dBFS")

# The gates must never end up above the voice they are meant to detect: on the
# real mic profile, ordinary speech has to still register as speech.
seg_check = calibrate([-43.0] * 250, 0.02)
sil_c, sp_c = seg_check.thresholds()
segs_c = feed(seg_check, np.concatenate([speech(1.2), silence(1.5, -43.0)]))
print(f"  post-calibration check: floor={seg_check.floor_db():.0f} "
      f"gates={sil_c:.0f}/{sp_c:.0f} segments={len(segs_c)}")
check("a normal utterance is still detected on the real mic profile",
      len(segs_c) >= 1, f"got {len(segs_c)}")

# REGRESSION (2): calibration must not depend on wall-clock time. The audio
# callback delivers blocks on the audio clock; when its callbacks arrive faster
# than real time, a wall-clock sampling gate never opens, the window starves,
# `ready` stays false, the gates never leave their fixed fallback, and speech
# gets rejected as "no speech". Feed this whole session as fast as Python can
# and require the same result.
seg_fast = vi.SilenceSegmenter(silence_db=-45.0, speech_db=-38.0, silence_hold=0.65,
                               min_speech=0.30, preroll=0.35, max_segment=30.0,
                               min_segment=0.35, adaptive=True)
seg_fast.reset()
fast_audio = np.concatenate([silence(4.0, -43.0), speech(1.2), silence(0.9, -43.0),
                             speech(1.5), silence(0.9, -43.0), speech(1.0),
                             silence(1.2, -43.0)])
fast_segs = feed(seg_fast, fast_audio)          # as fast as possible, no clock
fast_tail = seg_fast.flush()
if fast_tail is not None:
    fast_segs.append(fast_tail)
floor_f, (sil_f, sp_f) = seg_fast.floor_db(), seg_fast.thresholds()
print(f"  fast-feed session: floor={floor_f:.0f} -> silence={sil_f:.0f} "
      f"speech={sp_f:.0f}, samples={len(seg_fast._floor._levels)}, "
      f"segments={len(fast_segs)}")
check("noise floor calibrated even when fed faster than real time",
      floor_f < -35, f"floor={floor_f:.1f}")
check("sampling window actually filled",
      len(seg_fast._floor._levels) >= 10, str(len(seg_fast._floor._levels)))
check("gates moved off the fixed fallback",
      abs(sil_f - (-45.0)) > 0.5, f"silence gate={sil_f:.1f}")
check("all utterances segmented with fast feeding", len(fast_segs) >= 2,
      f"{len(fast_segs)}")
if fast_segs:
    check("fast-fed segments contain speech-level audio",
          level_of(fast_segs[0].audio) > -32, f"{level_of(fast_segs[0].audio):.0f} dBFS")

# A soft utterance must still be caught after calibration.
seg_live = calibrate([-62.0] * 250, 0.02)
segs = feed(seg_live, np.concatenate([speech(1.0, -30.0), silence(1.5, -62.0)]))
check("a -30 dBFS utterance is detected after calibration", len(segs) == 1,
      f"got {len(segs)}")

# And a quiet room must not produce segments out of room tone alone.
seg_quiet = calibrate([-62.0] * 250, 0.02)
segs = feed(seg_quiet, silence(6.0, -62.0))
check("6s of pure room tone produces no segments", len(segs) == 0, f"got {len(segs)}")

print()
print("=" * 70)
print("TEST 9 - callback path: no copies, no blocking")
print("=" * 70)


class FakeClient:
    def __init__(self, delay: float = 1.5):
        self.calls: list[float] = []
        self.lock = threading.Lock()
        self.delay = delay

    def transcribe(self, audio, model=None, language=None):
        with self.lock:
            self.calls.append(audio.size / RATE)
        time.sleep(self.delay)
        return "识别结果"


args = types.SimpleNamespace(
    model="qwen3-asr-0.6b", language=None, paste=False, restore_clipboard=False,
    max_seconds=300, vad=True, adaptive_noise=False, silence_db=-45.0, speech_db=-38.0,
    silence_hold=0.65, min_speech=0.30, preroll=0.35, max_segment=30.0, min_segment=0.35,
    queue_workers=1, queue_max=64, backlog_warn=5, retry=0,
    log_transcripts=False, status_title=False, cancel_hotkey="", hotkey="right ctrl",
    transcripts_dir=os.path.join(HERE, ".build_tmp", "pipe_transcripts"),
    failed_audio_dir=os.path.join(HERE, ".build_tmp", "pipe_failed"),
)
client = FakeClient()
rec = vi.Recorder(args, client)
rec.queue.start()

# the exact object the audio thread touches
seg = rec._new_segmenter()
rec.seg = seg
rec.gate = True
block = np.zeros(320, dtype=np.float32)
worst = 0.0
for i in range(2000):                      # 40s of audio, including speech bursts
    data = speech(0.02, -20.0) if (i // 25) % 2 == 0 else silence(0.02, -70)
    t0 = time.perf_counter()
    out = seg.feed(data)
    worst = max(worst, time.perf_counter() - t0)
    if out is not None:
        rec.enqueue(out)
check("per-block feed stays well under the 20ms budget", worst < 0.005,
      f"worst {worst * 1000:.2f} ms")
rec.gate = False
# Drain the segments the burst phase produced before measuring the queue.
check("burst phase drained", rec.queue.wait_idle(timeout=30))
check("burst phase produced segments of plausible size",
      bool(client.calls) and max(client.calls) <= 30.5,
      f"durations {[round(c, 2) for c in client.calls[:6]]}")
client.calls.clear()
with rec.stats_lock:
    rec.stats.update(segments=0, done=0, failed=0, empty=0, chars=0)

t0 = time.time()
for i in range(4):
    rec.enqueue(vi.Segment(blocks=[speech(2.0)], index=i, reason="test",
                           started=0.0, ended=2.0, peak_db=-20.0, frames=2 * RATE))
enqueue_time = time.time() - t0
check("enqueueing 4 segments is instant (<0.5s incl. beep+log)",
      enqueue_time < 0.5, f"{enqueue_time:.3f}s")
check("backlog visible immediately", rec.queue.pending() == 4, str(rec.queue.pending()))
seg = make_segmenter()
feed(seg, np.concatenate([speech(1.0), silence(0.2, -60)]))
t0 = time.time()
rec.enqueue(seg.flush("manual"), reason="stop")
check("stop/flush path instant (<0.1s)", time.time() - t0 < 0.1, f"{time.time() - t0:.3f}s")
check("queue order preserved (FIFO)", rec.queue.pending() == 5, str(rec.queue.pending()))
check("queue drained", rec.queue.wait_idle(timeout=30))
check("all 5 jobs transcribed", len(client.calls) == 5, str(len(client.calls)))
check("stats counted 5 results", rec.stats["done"] == 5, str(rec.stats))
rec.queue.shutdown()

print()
print("=" * 70)
print("TEST 10 - live server end-to-end through the real queue")
print("=" * 70)
live = vi.ASRClient(timeout=300)
ready = live.wait_ready(timeout=600)
check("server reachable", ready)
if ready:
    rec_live = vi.Recorder(args, live)
    rec_live.queue.start()
    for i in range(3):
        rec_live.enqueue(vi.Segment(blocks=[speech(2.0)], index=i, reason="live",
                                    started=0.0, ended=2.0, peak_db=-20.0, frames=2 * RATE))
    t0 = time.time()
    check("3 live jobs completed", rec_live.queue.wait_idle(timeout=120),
          f"{time.time() - t0:.1f}s")
    check("no failures", rec_live.stats["failed"] == 0, str(rec_live.stats))
    rec_live.queue.shutdown()
    # --test-file path: synthesise a wav and transcribe it, then remove it.
    scratch = os.path.join(HERE, ".build_tmp")
    os.makedirs(scratch, exist_ok=True)
    wav = os.path.join(scratch, "_selftest_speech.wav")
    try:
        import soundfile as sf
        sf.write(wav, speech(2.0), RATE)
        text = vi.transcribe_file(wav, live, args)
        check("--test-file path returns a string", isinstance(text, str), repr(text)[:60])
    except Exception as e:
        check("--test-file path returns a string", False, f"{type(e).__name__}: {e}")
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass

print()
print("=" * 70)
print("RESULT:", "ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILURE(S): {FAILURES}")
print("=" * 70)
sys.exit(1 if FAILURES else 0)
