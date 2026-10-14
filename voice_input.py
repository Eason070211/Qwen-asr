"""
Qwen3-ASR local voice input tool (global hotkey, toggle mode).

Press the hotkey once to START recording, press it again to STOP.

Two things make this different from a naive record-then-transcribe tool:

1. **Transcription never blocks recording.**
   Finished audio chunks are pushed onto a queue and transcribed by background
   worker threads, while you keep talking. Pressing the hotkey only closes the
   current chunk - it does not wait for inference - so the next chunk can start
   immediately.

2. **Silence splitting (VAD).**
   While you speak, the input level is watched continuously. Whenever the
   input stays quiet for `--silence-hold` seconds (a natural pause between
   sentences), the speech so far is closed off and queued for transcription.
   So a long dictation turns into a stream of sentences that are transcribed
   (and pasted) while you are still recording the next one.

   The two gate levels are calibrated automatically from the measured noise
   floor (see --adaptive-noise), so a noisy mic or a quiet one both work
   without hand-tuning. Manual --silence-db / --speech-db override that.

Nothing is lost if something goes wrong: every result is appended to a
transcript log, and a segment whose transcription fails is retried and then
written to disk as a .wav (see --transcripts-dir / --failed-audio-dir).

Requires the Qwen3-ASR server to be running (see start_voice_input.bat).

Usage:
    python voice_input.py                            # default hotkey: right ctrl
    python voice_input.py --hotkey "ctrl+alt+f1"     # any combo or single key
    python voice_input.py --cancel-hotkey "right alt"  # discard the current chunk
    python voice_input.py --language Chinese         # force language (default: auto)
    python voice_input.py --no-paste                 # copy to clipboard only
    python voice_input.py --silence-hold 0.9         # tolerate longer pauses
    python voice_input.py --no-vad                   # never split, one job on stop
    python voice_input.py --dump-config              # show effective settings
    python voice_input.py --list-devices             # pick an input device
    python voice_input.py --test-file speech.wav     # offline test, no hotkey

All defaults live in config.py (`voice_*` settings) and can also be set through
QWEN_ASR_VOICE_* environment variables; command-line flags win.
"""

from __future__ import annotations

import argparse
import collections
import io
import math
import os
import re
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

SERVER_BASE = "http://127.0.0.1:8910"
TRANSCRIBE_URL = SERVER_BASE + "/v1/audio/transcriptions"
HEALTH_URL = SERVER_BASE + "/health"
RATE = 16000

# Text that carries no dictation (empty, or punctuation/symbols only).
_MEANINGFUL = re.compile(r"[\w\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", re.UNICODE)


# ============================================================ config bridge

def _cfg(*names, default):
    """Read a `voice_*` setting from config.py, tolerating its absence."""
    try:
        import config  # local module, next to this file
    except Exception:
        return default
    for name in names:
        value = getattr(config.settings, name, None)
        if value is not None:
            return value
    return default


# ============================================================ helpers

def beep(freq: int, dur: int) -> None:
    try:
        import winsound
        winsound.Beep(freq, dur)
    except Exception:
        pass


def dbfs(block: np.ndarray) -> float:
    """Level of a block in dBFS (0 = full scale, -120 = silence)."""
    if block.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-9))


def _finite(value, fallback: float) -> float:
    """Coerce a CLI/env value to a finite float (never raises)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def log(tag: str, message: str) -> None:
    print(f"{tag} {message}", flush=True)


def set_console_title(text: str) -> None:
    """Best-effort: show state in the window title so the log can stay scrolled."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW(text)
    except Exception:
        pass


# ============================================================ noise floor

class NoiseFloor:
    """Estimates the background noise level, so the speech gates can adapt.

    Uses a low percentile over a rolling window rather than a running minimum.
    A minimum-statistics estimator looks tempting, but real capture devices
    produce occasional near-silent blocks (driver dropouts, buffer gaps) while
    the room tone sits much higher - here it would latch onto those outliers and
    derive gates tens of dB too sensitive, so every pause would look like a
    pause in a silent room and the tool would fire on room tone.

    Sampling is rate-limited by the CALLER in audio frames, never by wall-clock
    time. An earlier version throttled on `time.time()` and silently starved:
    the audio callback delivers blocks on the audio clock, so when its callbacks
    arrived faster than real time, the "0.25 s since last sample" test was never
    satisfied, the window stayed at one entry, `ready` never became true, and
    the gates stayed at their fixed fallback forever - speech then read as
    "no speech" and every segment was discarded.

    The window is fed only while nobody is speaking, so speech cannot drag the
    estimate up. `ready` stays False until enough observations exist; callers
    should fall back to fixed levels until then.
    """

    def __init__(self, *, percentile: float = 10.0, window_s: float = 10.0,
                 sample_s: float = 0.25, min_samples: int = 10,
                 min_db: float = -95.0, max_db: float = -20.0):
        self.percentile = percentile
        self.sample_s = sample_s
        self.min_samples = min_samples
        self.min_db = min_db
        self.max_db = max_db
        self._cap = max(min_samples, int(round(window_s / sample_s)))
        self._levels: collections.deque = collections.deque(maxlen=self._cap)
        self.value = max_db

    @property
    def ready(self) -> bool:
        """Enough of the window is filled to trust the estimate."""
        return len(self._levels) >= min(self.min_samples, max(1, self._cap // 2))

    def window_fill(self) -> float:
        """How full the rolling window is, 0..1 (for diagnostics)."""
        return len(self._levels) / self._cap

    def observe(self, block_db: float) -> None:
        """Offer one quiet-block level. The caller decides the sampling rate."""
        self._levels.append(block_db)
        if self.ready:
            self.value = max(self.min_db,
                             min(self.max_db, float(np.percentile(self._levels,
                                                                  self.percentile))))

    def reset(self) -> None:
        self._levels.clear()
        self.value = self.max_db


# ============================================================ segmentation

@dataclass
class Segment:
    """One chunk of speech, ready to be transcribed.

    Holds the raw audio blocks; `audio` concatenates them lazily. Nothing here
    copies audio on the capture path, so the sounddevice callback stays cheap.
    """
    blocks: list[np.ndarray]
    index: int
    reason: str                  # "silence" | "max-length" | "manual"
    started: float               # seconds since recording start
    ended: float
    peak_db: float = -120.0
    frames: int = 0
    _audio: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def audio(self) -> np.ndarray:
        if self._audio is None:
            self._audio = (np.concatenate(self.blocks) if self.blocks
                           else np.zeros(0, np.float32))
        return self._audio

    @property
    def seconds(self) -> float:
        return (self.frames or sum(b.size for b in self.blocks)) / RATE


class SilenceSegmenter:
    """Splits a continuous stream of audio blocks into speech segments.

    Feed audio blocks in; it yields a `Segment` whenever a chunk is complete.
    A chunk is closed by the first of:

      * `silence_hold` seconds of input quieter than `silence_db`
        (a natural pause - this is the normal case),
      * `max_segment` seconds of speech (safety valve for non-stop talking),
      * an explicit `flush()` (the user pressed stop).

    Short blips (keyboard clicks, coughs) are discarded: speech must stay above
    `speech_db` for `min_speech` seconds before a chunk is taken seriously.
    The `preroll` seconds of audio *before* that point are kept, so the first
    syllable is never clipped.

    Levels are compared through a fast-attack / slow-release envelope follower.
    Comparing raw per-block RMS instead would split mid-word, because the
    amplitude of ordinary speech dips to near zero several times a second.
    """

    RELEASE = 0.08        # display decay per 20 ms block (~250 ms time constant)
    SILENCE_ABOVE_FLOOR = 10.0
    SPEECH_ABOVE_FLOOR = 18.0

    def __init__(self, *, silence_db: float, speech_db: float, silence_hold: float,
                 min_speech: float, preroll: float, max_segment: float,
                 min_segment: float, vad: bool = True, sample_rate: int = RATE,
                 adaptive: bool = True):
        self.sample_rate = sample_rate
        self.hold_seconds = silence_hold
        self.min_speech_seconds = min_speech
        self.min_frames = max(1, int(round(min_segment * sample_rate)))
        self.preroll_frames = max(0, int(round(preroll * sample_rate)))
        self.max_frames = max(1, int(round(max_segment * sample_rate)))
        # Only sample the noise floor after this much confirmed-quiet audio, so
        # a single quiet block inside a word cannot feed the estimator.
        self.observe_after = max(1, int(round(0.25 * sample_rate)))
        self.vad = vad
        self.adaptive = adaptive
        self.manual_silence_db = silence_db
        self.manual_speech_db = speech_db
        self._floor = NoiseFloor()
        self._floor_valid = False

        self._buf: list[np.ndarray] = []
        self._buf_frames = 0
        self._pre: collections.deque = collections.deque()
        self._pre_frames = 0
        self._active = False        # speech confirmed, chunk is being collected
        self._attempt = -1          # -1 = idle, else frames buffered since onset
        self._speech_frames = 0     # frames above the speech gate in this attempt
        self._last_above = -1       # `_attempt` when speech was last seen
        self._gap = 0               # consecutive frames below the silence gate
        self._disp = -120.0         # envelope-followed level
        self._peak = -120.0
        self._pos = 0               # total frames seen since reset()
        self._observe_cooldown = 0  # confirmed-quiet frames since last sample
        self._index = 0
        self._speech_start = 0
        self.reset()

    # -------------------------------------------------------- gate levels
    def thresholds(self) -> tuple[float, float]:
        """(silence_db, speech_db) currently in force.

        Until the noise floor has enough quiet observations, the fixed values
        are used, so the very first utterance is never missed.
        """
        if self.adaptive and self._floor.ready:
            silence = self._floor.value + self.SILENCE_ABOVE_FLOOR
            speech = self._floor.value + self.SPEECH_ABOVE_FLOOR
        else:
            silence, speech = self.manual_silence_db, self.manual_speech_db
        # The closing gate must never sit below the opening gate, otherwise a
        # chunk could never be closed and every one would run to max_segment.
        return min(silence, speech), speech

    def floor_db(self) -> float:
        """The measured background noise level in dBFS."""
        return self._floor.value

    def floor_ready(self) -> bool:
        return self._floor.ready

    # -------------------------------------------------------- internals
    def _prune_pre(self) -> None:
        limit = self.preroll_frames
        frames = 0
        keep: collections.deque = collections.deque()
        for block in reversed(self._pre):
            if frames >= limit:
                break
            keep.appendleft(block)
            frames += block.size
        self._pre = keep
        self._pre_frames = frames

    def _make(self, reason: str) -> Segment:
        pre = list(self._pre)
        blocks = (pre + self._buf) if pre else list(self._buf)
        frames = sum(b.size for b in blocks)
        seg = Segment(
            blocks=blocks,
            index=self._index,
            reason=reason,
            started=max(0.0, (self._speech_start - self._pre_frames) / self.sample_rate),
            ended=self._pos / self.sample_rate,
            peak_db=self._peak,
            frames=frames,
        )
        self._index += 1
        return seg

    def _discard(self) -> None:
        self._buf = []
        self._buf_frames = 0
        self._pre.clear()
        self._pre_frames = 0
        self._active = False
        self._attempt = -1
        self._speech_frames = 0
        self._last_above = -1
        self._gap = 0
        self._peak = -120.0

    def _emit(self, reason: str, speech_frames: int) -> Optional[Segment]:
        """Close the current chunk. Returns None if it was not real speech."""
        if not self._buf and not self._pre:
            return None
        if speech_frames < self._min_speech_frames():
            return None                         # only a click / cough
        if self._buf_frames + self._pre_frames < self.min_frames:
            return None
        return self._make(reason)

    def _min_speech_frames(self) -> int:
        return max(1, int(round(self.min_speech_seconds * self.sample_rate)))

    def _hold_frames(self) -> int:
        return max(1, int(round(self.hold_seconds * self.sample_rate)))

    # -------------------------------------------------------- public
    def feed(self, block: np.ndarray) -> Optional[Segment]:
        """Consume one audio block; return a finished Segment, if any.

        Runs on the audio callback thread: no I/O, no copies, no locks.
        """
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return None

        level = dbfs(block)
        # Fast attack, slow release: bridges the amplitude dips inside a word.
        self._disp = level if level > self._disp else self._disp - self.RELEASE * (
            self._disp - level)
        silence_db, speech_db = self.thresholds()

        quiet = self._disp < silence_db
        above = self._disp >= speech_db
        # Feed the noise estimator only from audio that is not speech. Two
        # guards matter here, and getting them wrong is how the estimator once
        # ended up at the speaker's own level (-20 dBFS), after which the gates
        # rose above the voice and every segment was discarded as "no speech":
        #   * `not self._active` - never sample a confirmed chunk;
        #   * a broad not-speech test (raw level below the speech gate) rather
        #     than "confirmed quiet", so the estimate can still be pulled *down*
        #     while the gates are still too high from a previous session.
        # Upward movement is bounded by the current estimate (a quieter-or-equal
        # observation can only lower it); downward is always allowed.
        if not self._active and not above:
            self._observe_cooldown += block.size
            if self._observe_cooldown >= self.observe_after:
                self._observe_cooldown = 0
                self._floor.observe(min(level, self._disp, self._floor.value))

        # 1) Classify this block. Once a chunk is confirmed, only a genuine
        #    quiet stretch ends it; until then we are deciding whether the
        #    sound is speech at all.
        #
        #    Speech evidence is taken from the RAW block level, not the
        #    envelope-followed one. The follower's slow release dips below the
        #    speech gate during ordinary speech (and dips twice per cycle on a
        #    modulated voice), so counting only `_disp >= speech_db` frames
        #    could never reach `min_speech`: chunks were then rejected as "no
        #    speech" and the user got nothing at all.
        if self._active:
            if above:
                self._gap = 0
            elif quiet:
                self._gap += block.size
        elif not quiet:
            if self._attempt < 0:               # start (or restart) a candidate
                self._attempt = 0
                self._speech_frames = 0
                self._last_above = 0
            if level >= speech_db:
                self._speech_frames += block.size
                self._last_above = self._attempt
                self._peak = max(self._peak, level)
        else:
            self._attempt = -1                  # a blip: keys, cough, door
            self._speech_frames = 0

        # 2) Every block that belongs to a candidate or a confirmed chunk is
        #    kept - this is the audio that will actually be transcribed.
        if self._active:
            self._buf.append(block)
            self._buf_frames += block.size
        elif self._attempt >= 0:
            self._buf.append(block)
            self._buf_frames += block.size
            self._attempt += block.size
            if self._speech_frames >= self._min_speech_frames():
                self._active = True             # speech confirmed
                self._speech_start = self._attempt - self._buf_frames
                self._gap = max(0, (self._last_above + block.size) - self._attempt)
            elif self._attempt > self.sample_rate:
                self._discard()                 # a second of noise, not speech
        elif self.preroll_frames > 0:           # keep the run-up to speech
            self._pre.append(block)
            self._pre_frames += block.size
            if self._pre_frames > self.preroll_frames:
                self._prune_pre()

        # The block counts as captured from here on, so segment timings that
        # read `_pos` (Segment.ended) include it.
        self._pos += block.size

        # 3) Close the chunk when a pause - or the length cap - says so.
        seg: Optional[Segment] = None
        if self._active:
            if self.vad and self._gap >= self._hold_frames():
                seg = self._emit("silence", self._speech_frames)
                self._discard()
            elif self._buf_frames >= self.max_frames:
                seg = self._emit("max-length", self._speech_frames)
                self._discard()                 # keep going as a fresh chunk

        return seg

    def flush(self, reason: str = "manual") -> Optional[Segment]:
        """Close the current chunk now (the user pressed stop)."""
        speech = self._speech_frames if self._attempt >= 0 else 0
        seg = self._emit(reason, speech)
        self._discard()
        return seg

    def cancel(self) -> float:
        """Throw away the chunk in progress; returns its length in seconds.

        Cancelling must not disturb the noise-floor estimate (the discarded
        audio may well have been speech), so only the capture state is reset.
        """
        seconds = self._buf_frames / self.sample_rate
        self._buf = []
        self._buf_frames = 0
        self._active = False
        self._attempt = -1
        self._speech_frames = 0
        self._last_above = -1
        self._gap = 0
        self._peak = -120.0
        return seconds

    def reset(self) -> None:
        self._discard()
        self._disp = -120.0
        self._index = 0
        self._pos = 0
        self._speech_start = 0
        self._observe_cooldown = 0
        self._floor.reset()


# ============================================================ result store

class ResultStore:
    """Append-only safety net for transcription results.

    Text is written as it arrives, so a missed paste, a closed window or a
    crash never loses a dictation. The log is rotated by size so a long-running
    tool cannot fill the disk.
    """

    def __init__(self, directory: str, enabled: bool = True,
                 max_bytes: int = 2_000_000, keep: int = 5):
        self.directory = directory
        self.enabled = enabled
        self.max_bytes = max_bytes
        self.keep = keep
        self._lock = threading.Lock()
        self.written = 0

    def _path_for(self, when: datetime) -> str:
        return os.path.join(self.directory, f"transcripts-{when:%Y-%m-%d}.txt")

    def _rotate(self, path: str) -> None:
        try:
            if os.path.getsize(path) < self.max_bytes:
                return
            stamp = datetime.now().strftime("%H%M%S")
            os.replace(path, f"{path}.{stamp}.old")
            # keep only the newest few rotated files
            base = os.path.basename(path)
            olds = sorted(f for f in os.listdir(self.directory) if f.startswith(base) and f.endswith(".old"))
            for stale in olds[:-self.keep]:
                try:
                    os.remove(os.path.join(self.directory, stale))
                except OSError:
                    pass
        except OSError:
            pass

    def append(self, text: str, *, index: int, seconds: float, reason: str) -> Optional[str]:
        """Append one result. Returns the path written, or None if disabled."""
        if not self.enabled:
            return None
        when = datetime.now()
        try:
            with self._lock:
                os.makedirs(self.directory, exist_ok=True)
                path = self._path_for(when)
                self._rotate(path)
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(f"[{when:%Y-%m-%d %H:%M:%S}] #{index} "
                             f"{seconds:.2f}s {reason}\n{text}\n\n")
                self.written += 1
            return path
        except OSError as e:
            log("[WARN]", f"写入转写记录失败: {e}")
            return None


class FailedAudioStore:
    """Keeps the audio of segments that could not be transcribed at all."""

    def __init__(self, directory: str, enabled: bool = True):
        self.directory = directory
        self.enabled = enabled
        self._lock = threading.Lock()
        self.saved = 0

    def save(self, audio: np.ndarray, *, index: int, seconds: float, error: str) -> Optional[str]:
        if not self.enabled or audio.size == 0:
            return None
        when = datetime.now()
        try:
            with self._lock:
                os.makedirs(self.directory, exist_ok=True)
                path = os.path.join(
                    self.directory,
                    f"failed-{when:%Y%m%d-%H%M%S}-{index:03d}.wav")
                sf.write(path, audio, RATE, subtype="PCM_16")
                with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as fh:
                    fh.write(f"time: {when:%Y-%m-%d %H:%M:%S}\n"
                             f"index: {index}\nduration: {seconds:.2f}s\n"
                             f"error: {error}\n")
                self.saved += 1
            return path
        except Exception as e:
            log("[WARN]", f"保存失败音频出错: {e}")
            return None


# ============================================================ transcription queue

@dataclass
class Job:
    """A queued transcription request."""
    audio: np.ndarray
    index: int
    seconds: float
    reason: str
    peak_db: float = -120.0
    done: threading.Event = field(default_factory=threading.Event)
    text: Optional[str] = None          # None = failed
    error: str = ""
    attempts: int = 0
    queued_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0


class TranscribeQueue:
    """FIFO backlog of transcription jobs, drained by background workers.

    `enqueue` never blocks: recording threads hand audio over and continue.
    Workers report each finished job through `on_result` / `on_error`, which
    run *outside* the queue lock so slow work (HTTP, clipboard) cannot stall
    the producer.
    """

    def __init__(self, *, workers: int, maxsize: int, transcribe: Callable[[np.ndarray], str],
                 on_result: Callable[[Job], None], on_error: Callable[[Job], None],
                 retries: int = 0, on_retry: Optional[Callable[[Job], None]] = None):
        self._cv = threading.Condition()
        self._items: collections.deque[Job] = collections.deque()
        self._inflight = 0
        self._dropped = 0
        self._stop = False
        self._workers = max(1, int(workers))
        self._maxsize = max(1, int(maxsize))
        self._transcribe = transcribe
        self._on_result = on_result
        self._on_error = on_error
        self._on_retry = on_retry
        self._retries = max(0, int(retries))
        self._threads: list[threading.Thread] = []

    # -------------------------------------------------------------- public
    def start(self) -> None:
        for i in range(self._workers):
            t = threading.Thread(target=self._run, name=f"asr-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def enqueue(self, job: Job) -> int:
        """Queue a job. Returns the pending count; never blocks."""
        dropped = None
        with self._cv:
            if len(self._items) >= self._maxsize:
                dropped = self._items.popleft()
                self._dropped += 1
            self._items.append(job)
            pending = len(self._items)
            self._cv.notify()
        if dropped is not None:
            log("[WARN]", f"队列已满（{self._maxsize}），丢弃最旧的一段 #{dropped.index}")
            dropped.done.set()
        return pending

    def pending(self) -> int:
        """Jobs waiting plus jobs currently being transcribed."""
        with self._cv:
            return len(self._items) + self._inflight

    def dropped(self) -> int:
        with self._cv:
            return self._dropped

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        """Block until the backlog is empty. True if it drained in time."""
        deadline = None if timeout is None else time.time() + timeout
        with self._cv:
            while self._items or self._inflight:
                remaining = None if deadline is None else deadline - time.time()
                if remaining is not None and remaining <= 0:
                    return False
                self._cv.wait(0.25 if remaining is None else min(0.25, remaining))
            return True

    def shutdown(self) -> None:
        """Drain the backlog, then stop the workers."""
        self.wait_idle()
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    # -------------------------------------------------------------- worker
    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._items and not self._stop:
                    self._cv.wait(0.5)
                if self._stop and not self._items:
                    return
                job = self._items.popleft()
                self._inflight += 1
                job.started_at = time.time()
            # A job stays "pending" until its result has been handled, not just
            # until inference returned. Otherwise wait_idle() (and the shutdown
            # drain) can report the queue empty while the transcript log write,
            # the paste and the failed-audio save are still running - the last
            # dictation of a session would then be silently incomplete.
            try:
                self._transcribe_job(job)
                job.finished_at = time.time()
                job.done.set()
                if job.text is None:
                    self._on_error(job)
                else:
                    self._on_result(job)
            except Exception as e:                      # never kill the worker
                log("[ERR]", f"结果处理异常: {e}")
            finally:
                with self._cv:
                    self._inflight -= 1
                    self._cv.notify_all()

    def _transcribe_job(self, job: Job) -> None:
        """Run one job, retrying transient server failures with backoff."""
        last_error = ""
        for attempt in range(self._retries + 1):
            job.attempts = attempt + 1
            try:
                job.text = self._transcribe(job.audio)
                job.error = ""
                return
            except Exception as e:
                last_error = str(e)
                if attempt < self._retries:
                    if self._on_retry is not None:
                        try:
                            self._on_retry(job)
                        except Exception:
                            pass
                    time.sleep(min(6.0, 1.0 * (2 ** attempt)))
        job.text = None
        job.error = last_error


def normalize_audio(audio: np.ndarray, target_db: float = -22.0,
                    peak_ceiling_db: float = -1.0) -> tuple[np.ndarray, float]:
    """Gain-up a quiet chunk so the model sees a usable level.

    Windows input devices with automatic gain control (this machine's AMD mic
    array drifts over a 33 dB range) hand us segments anywhere from -50 to
    -15 dBFS. The model's output tracks that level: at -48 dBFS it answers
    "嗯。" or nothing, at -16 dBFS it transcribes properly. So normalise to a
    target RMS, but never let the peak clip - raising a noisy segment to full
    scale would just give the model louder noise.

    Returns (audio, applied_gain_db).
    """
    if audio.size == 0:
        return audio, 0.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))
    if rms <= 1e-9 or peak <= 1e-9:
        return audio, 0.0
    target = 10 ** (target_db / 20.0)
    ceiling = 10 ** (peak_ceiling_db / 20.0)
    gain = min(target / rms, ceiling / peak)
    if gain <= 1.0:
        return audio, 0.0
    gain = min(gain, 100.0)                      # +40 dB is already extreme
    return (audio * gain).astype(np.float32), 20.0 * math.log10(gain)


# ============================================================ ASR client

class ASRClient:
    """Thin HTTP client for the transcription endpoint.

    Also prepares each chunk for the model: level normalisation (see
    `normalize_audio`) and a language fallback, both of which this deployment
    needs because of its microphone and its language auto-detection.
    """

    def __init__(self, base: str = SERVER_BASE, timeout: int = 300, *,
                 auto_gain: bool = True, gain_target_db: float = -22.0,
                 fallback_language: Optional[str] = "Chinese"):
        self.session = requests.Session()
        try:
            from requests.adapters import HTTPAdapter
            adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
        except Exception:
            pass
        self.transcribe_url = base + "/v1/audio/transcriptions"
        self.health_url = base + "/health"
        self.timeout = timeout
        self.auto_gain = auto_gain
        self.gain_target_db = gain_target_db
        self.fallback_language = fallback_language
        self.gains: collections.deque = collections.deque(maxlen=200)
        self.fallbacks = 0
        self._lock = threading.Lock()

    def wait_ready(self, timeout: int = 360) -> bool:
        """Block until the server reports status=ok (model loaded)."""
        log("[server]", "等待服务器就绪（首次需加载模型，约 1-2 分钟）")
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = self.session.get(self.health_url, timeout=3)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    log("[server]", "就绪: " + SERVER_BASE)
                    return True
            except requests.RequestException:
                pass
            print(".", end="", flush=True)
            time.sleep(5)
        log("[server]", "等待超时：请确认 server.py 已启动且模型加载成功")
        return False

    def _post(self, audio: np.ndarray, model: str,
              language: Optional[str]) -> str:
        buf = io.BytesIO()
        sf.write(buf, audio, RATE, format="WAV", subtype="PCM_16")
        buf.seek(0)
        data = {"model": model}
        if language:
            data["language"] = language
        with self._lock:                    # requests.Session is not thread-safe
            r = self.session.post(
                self.transcribe_url,
                files={"file": ("speech.wav", buf, "audio/wav")},
                data=data,
                timeout=self.timeout,
            )
        r.raise_for_status()
        return r.json().get("text", "") or ""

    def transcribe(self, audio: np.ndarray, *, model: str,
                   language: Optional[str]) -> str:
        """POST one chunk; returns the text. Raises on failure."""
        if audio.size == 0:
            return ""
        if self.auto_gain:
            audio, gain = normalize_audio(audio, self.gain_target_db)
            if gain:
                self.gains.append(gain)
                if gain >= 3.0:
                    log("[GAIN]", f"这段偏轻，已提升 {gain:.1f} dB 再送模型")
        text = self._post(audio, model, language)
        # Language auto-detection returns empty for Chinese on this model, so an
        # empty result is retried once with an explicit language.
        if not _MEANINGFUL.search(text or "") and self.fallback_language \
                and not language:
            self.fallbacks += 1
            log("[LANG]", f"自动识别没出文字，改用 {self.fallback_language} 重试一次")
            text = self._post(audio, model, self.fallback_language)
        return text

    def gain_stats(self) -> tuple[int, float, float]:
        """(count, min dB, max dB) of applied normalisation gains."""
        if not self.gains:
            return 0, 0.0, 0.0
        return len(self.gains), min(self.gains), max(self.gains)


def resolve_device(spec):
    """Turn a --device value into something sounddevice accepts.

    Accepts an index (0, 3, ...) or a name fragment ('bose', 'Microphone Array').
    Returns None for "use the system default".
    """
    if spec is None or spec == "":
        return None
    try:
        return int(spec)
    except (TypeError, ValueError):
        pass
    needle = str(spec).lower()
    try:
        devices = sd.query_devices()
    except Exception as e:
        log("[WARN]", f"无法枚举音频设备（{e}），使用系统默认输入")
        return None
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0 and needle in dev.get("name", "").lower():
            log("[VAD]", f"使用输入设备 [{idx}] {dev.get('name')}")
            return idx
    log("[WARN]", f"未找到匹配 {spec!r} 的输入设备，使用系统默认输入")
    return None


def list_devices() -> None:
    print("可用的输入设备（把编号或名称片段传给 --device）:")
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        mark = "  <-- 默认" if idx == default_in else ""
        print(f"  [{idx:>3}] {dev['name']}  channels={dev['max_input_channels']}{mark}")


def _target_foreground() -> int:
    """Best-effort: the native window that currently has focus (0 = unknown)."""
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        root = ctypes.windll.user32.GetAncestor(hwnd, 2)   # GA_ROOT
        return int(root or hwnd or 0)
    except Exception:
        return 0


def paste_text(text: str, paste: bool, restore_clipboard: bool, paste_lock: threading.Lock) -> None:
    """Put `text` on the clipboard and (optionally) paste it at the cursor."""
    try:
        import pyperclip
    except Exception as e:
        log("[WARN]", f"未安装 pyperclip，仅打印文本: {e}")
        return

    with paste_lock:                        # keep auto-pastes from interleaving
        old = None
        if restore_clipboard and paste:
            try:
                old = pyperclip.paste()
            except Exception:
                old = None
        try:
            pyperclip.copy(text)
        except Exception as e:
            log("[WARN]", f"写入剪贴板失败: {e}")
            return
        if not paste:
            return
        target = _target_foreground()
        try:
            import keyboard as kb
            kb.press_and_release("ctrl+v")
        except Exception as e:
            log("[WARN]", f"自动粘贴失败（请手动 Ctrl+V）: {e}")
            return
    # Outside the lock: give the focused application time to consume the paste
    # before the old clipboard content goes back, otherwise a slow app pastes
    # stale text. Holding the paste lock here would delay the next segment.
    if old is None:
        return
    for _ in range(20):
        time.sleep(0.05)
        if _target_foreground() != target:
            break
    try:
        pyperclip.copy(old)
    except Exception:
        pass


# ============================================================ recorder

class Recorder:
    """Owns the microphone stream, the segmenter and the transcription queue.

    The hotkey only ever calls `start()` / `stop()`; both return quickly.
    `stop()` closes the current chunk and hands it to the queue - it never
    waits for inference.
    """

    def __init__(self, args, client):
        self.args = args
        self.client = client
        self.lock = threading.Lock()
        self.paste_lock = threading.Lock()
        self.seg: Optional[SilenceSegmenter] = None
        self.stream = None
        self.gate = False                   # audio callback is recording
        self.watchdogs = 0
        self.stats_lock = threading.Lock()
        self.stats = {"segments": 0, "done": 0, "failed": 0, "empty": 0,
                      "chars": 0, "cancelled": 0, "retried": 0}
        self.store = ResultStore(args.transcripts_dir, enabled=args.log_transcripts)
        self.failed = FailedAudioStore(args.failed_audio_dir, enabled=True)
        self.queue = TranscribeQueue(
            workers=args.queue_workers,
            maxsize=args.queue_max,
            transcribe=self._transcribe,
            on_result=self._on_result,
            on_error=self._on_error,
            retries=args.retry,
            on_retry=self._on_retry,
        )

    # -------------------------------------------------------------- plumbing
    def _new_segmenter(self) -> SilenceSegmenter:
        a = self.args
        return SilenceSegmenter(
            silence_db=a.silence_db,
            speech_db=a.speech_db,
            silence_hold=a.silence_hold,
            min_speech=a.min_speech,
            preroll=a.preroll,
            max_segment=a.max_segment,
            min_segment=a.min_segment,
            vad=a.vad,
            adaptive=a.adaptive_noise,
        )

    def _transcribe(self, audio: np.ndarray) -> str:
        return self.client.transcribe(audio, model=self.args.model,
                                      language=self.args.language or None)

    # -------------------------------------------------------------- callbacks
    def enqueue(self, seg: Optional[Segment], reason: str = "") -> None:
        if seg is None:
            return
        audio = seg.audio
        if audio.size == 0:
            return
        reason = reason or seg.reason
        job = Job(audio=audio, index=seg.index, seconds=audio.size / RATE,
                  reason=reason, peak_db=seg.peak_db)
        pending = self.queue.enqueue(job)
        with self.stats_lock:
            self.stats["segments"] += 1
        log("[SEG]", f"#{seg.index} 切出一段 {job.seconds:.2f}s "
                     f"(峰值 {seg.peak_db:.0f} dBFS, {reason}) → 已排队，"
                     f"待处理 {pending}")
        beep(900, 60)
        self._refresh_title()
        if pending >= self.args.backlog_warn:
            log("[WARN]", f"转写积压 {pending} 段：说话可以继续，文本会稍后依次输出")

    def _on_retry(self, job: Job) -> None:
        with self.stats_lock:
            self.stats["retried"] += 1
        log("[RETRY]", f"#{job.index} 转写失败，第 {job.attempts} 次重试：{job.error}")

    def _on_result(self, job: Job) -> None:
        text = (job.text or "").strip()
        latency = job.finished_at - job.queued_at
        if not _MEANINGFUL.search(text):
            with self.stats_lock:
                self.stats["empty"] += 1
                self.stats["done"] += 1
            log("[SKIP]", f"#{job.index} {job.seconds:.2f}s 无有效语音（{latency:.1f}s）")
            self._refresh_title()
            return
        with self.stats_lock:
            self.stats["done"] += 1
            self.stats["chars"] += len(text)
        path = self.store.append(text, index=job.index, seconds=job.seconds,
                                 reason=job.reason)
        log("[TEXT]", f"#{job.index} ({job.seconds:.2f}s, {latency:.1f}s) {text}")
        paste_text(text, self.args.paste, self.args.restore_clipboard, self.paste_lock)
        if self.args.paste:
            log("[PASTE]", f"#{job.index} 已粘贴到光标处")
        if path:
            log("[SAVE]", f"#{job.index} 已追加到 {path}")
        beep(1500, 60)
        self._refresh_title()

    def _on_error(self, job: Job) -> None:
        with self.stats_lock:
            self.stats["failed"] += 1
        log("[FAIL]", f"#{job.index} 转写失败（重试 {job.attempts} 次）: {job.error}")
        saved = self.failed.save(job.audio, index=job.index, seconds=job.seconds,
                                 error=job.error)
        if saved:
            log("[KEEP]", f"#{job.index} 音频已保存，可稍后重试转写: {saved}")
        self._refresh_title()

    def _callback(self, indata, frames, time_info, status) -> None:
        """Audio thread. Must stay fast: no I/O, no waiting on inference."""
        with self.lock:
            if not self.gate or self.seg is None:
                return
            seg = self.seg
        try:
            out = seg.feed(indata[:, 0])
        except Exception as e:                     # never kill the audio stream
            log("[ERR]", f"切段异常: {e}")
            with self.lock:
                self.gate = False
            return
        if out is not None:
            self.enqueue(out)

    # -------------------------------------------------------------- status
    def title_text(self) -> str:
        with self.stats_lock:
            s = dict(self.stats)
        pending = self.queue.pending()
        if self.gate:
            head = "● 录音中"
        elif pending:
            head = "… 转写中"
        else:
            head = "○ 待机"
        parts = [f"{head} | 待处理 {pending}"]
        if s["done"]:
            parts.append(f"完成 {s['done']}")
        if s["failed"]:
            parts.append(f"失败 {s['failed']}")
        if s["cancelled"]:
            parts.append(f"取消 {s['cancelled']}")
        return f"Qwen3-ASR 语音输入 [{self.args.hotkey}] " + " ".join(parts)

    def _refresh_title(self) -> None:
        if not self.args.status_title:
            return
        set_console_title(self.title_text())

    # -------------------------------------------------------------- control
    def start(self) -> None:
        with self.lock:
            if self.gate:
                return
            self._close_stream_locked()
            self.seg = self._new_segmenter()
            self.seg.reset()
            try:
                device = resolve_device(self.args.device)
                self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                                             device=device, callback=self._callback)
                self.stream.start()
            except Exception as e:
                log("[ERR]", f"无法打开麦克风: {e}")
                self.stream = None
                self.seg = None
                return
            self.gate = True
        beep(1200, 120)
        if self.args.adaptive_noise:
            local, levels = self.describe_gates()
            log("[REC]", f"录音中…（停顿 {self.args.silence_hold:.2f}s 自动切段；"
                         f"{local}）")
            log("[VAD]", levels)
        else:
            log("[REC]", "录音中…（再按一次停止；停顿会自动切段并排队转写）")
        self._refresh_title()
        if self.args.cancel_hotkey:
            log("[TIP]", f"说错了按 [{self.args.cancel_hotkey}] 丢弃当前这段")

    def cancel_current(self) -> bool:
        """Discard the chunk being recorded (already-queued ones are untouched)."""
        with self.lock:
            if not self.gate or self.seg is None:
                return False
            seg = self.seg
        seconds = seg.cancel()
        with self.stats_lock:
            self.stats["cancelled"] += 1
        beep(400, 150)
        log("[CANCEL]", f"已丢弃当前 {seconds:.2f}s（排队中的不受影响）")
        self._refresh_title()
        return True

    def describe_gates(self) -> tuple[str, str]:
        """Human-readable view of the current gates (for the console/log)."""
        seg = self.seg
        if seg is None:
            return "门限未初始化", ""
        silence, speech = seg.thresholds()
        local = f"静音门限 {silence:.0f} dBFS / 说话门限 {speech:.0f} dBFS"
        if not self.args.adaptive_noise:
            levels = (f"固定门限：静音 {silence:.0f} / 说话 {speech:.0f} dBFS"
                      f"（--no-adaptive-noise）")
        elif not seg.floor_ready():
            levels = (f"自适应校准中（当前用 {silence:.0f} / {speech:.0f} dBFS，"
                      f"测到环境噪声后自动调整）")
        else:
            levels = (f"自适应噪声底 {seg.floor_db():.0f} dBFS → "
                      f"静音 {silence:.0f} / 说话 {speech:.0f} dBFS"
                      f"（噪声底会随环境自动调整）")
        return local, levels

    def refresh_floor(self) -> float:
        """The noise floor the gates are currently derived from."""
        seg = self.seg
        return seg.floor_db() if seg is not None else -95.0

    def stop(self, auto: bool = False) -> bool:
        """Close the current chunk and stop recording. Returns True if it was on."""
        with self.lock:
            if not self.gate:
                return False
            self.gate = False
            seg = self.seg
            self.seg = None
            if seg is not None and self.args.adaptive_noise:
                floor_note = f"（自适应噪声底 {seg.floor_db():.0f} dBFS）"
            else:
                floor_note = ""
            tail = seg.flush("manual") if seg is not None else None
        beep(700, 120)
        if tail is not None:
            self.enqueue(tail, reason="stop")
        pending = self.queue.pending()
        if auto:
            log("[WARN]", f"录音超过 {self.args.max_seconds}s，已自动停止")
        if pending:
            log("[REC]", f"录音已停止{floor_note}，{pending} 段仍在后台转写"
                         f"（可以接着按热键开始下一段）")
        else:
            log("[REC]", f"录音已停止{floor_note}")

        def _close():
            with self.lock:
                self._close_stream_locked()
        threading.Thread(target=_close, name="stream-close", daemon=True).start()
        self._refresh_title()
        return True

    def _close_stream_locked(self) -> None:
        stream = self.stream
        self.stream = None
        if stream is None:
            return
        for fn in (stream.stop, stream.close):
            try:
                fn()
            except Exception:
                pass

    def _watchdog(self, token: int) -> None:
        time.sleep(self.args.max_seconds)
        with self.lock:
            if token != self.watchdogs or not self.gate:
                return
        self.stop(auto=True)

    def arm_watchdog(self) -> None:
        with self.lock:
            self.watchdogs += 1
            token = self.watchdogs
        threading.Thread(target=self._watchdog, args=(token,), daemon=True).start()

    def status_line(self) -> str:
        with self.stats_lock:
            s = dict(self.stats)
        bits = [f"已转写 {s['done']} 段（{s['chars']} 字）",
                f"跳过 {s['empty']}", f"失败 {s['failed']}"]
        if s["retried"]:
            bits.append(f"重试 {s['retried']}")
        if s["cancelled"]:
            bits.append(f"取消 {s['cancelled']}")
        if self.failed.saved:
            bits.append(f"保留音频 {self.failed.saved}")
        bits.append(f"待处理 {self.queue.pending()}")
        if self.queue.dropped():
            bits.append(f"丢弃 {self.queue.dropped()}")
        if self.store.written:
            bits.append(f"已记录到 {self.args.transcripts_dir}")
        return "，".join(bits)

    def drain(self, label: str) -> None:
        pending = self.queue.pending()
        if not pending:
            return
        # On this CPU one chunk takes roughly a tenth of its own duration, so
        # allow a generous multiple and report honestly if it is still not done.
        budget = max(60.0, 20.0 * pending)
        log("[QUEUE]", f"{label}：等待 {pending} 段转写完成（最多 {budget:.0f}s）…")
        ok = self.queue.wait_idle(timeout=budget)
        if not ok:
            log("[WARN]", f"{label}：仍有 {self.queue.pending()} 段未完成，已放弃等待")
        else:
            log("[QUEUE]", f"{label}：队列已清空。{self.status_line()}")


# ============================================================ hotkey + main

def _parse_hotkey(spec: str) -> tuple[str, list[str]]:
    parts = [p.strip().lower() for p in (spec or "").split("+") if p.strip()]
    if not parts:
        return "", []
    return parts[-1], parts[:-1]


def build_hotkey_handler(trigger: str, mods: list[str], recorder: Recorder,
                         cancel_trigger: str = ""):
    """One toggle per physical press-release cycle (auto-repeat is ignored)."""
    state = {"down": False, "cancel_down": False}

    def handler(e):
        if cancel_trigger and e.name == cancel_trigger:
            if e.event_type == "down":
                state["cancel_down"] = True
            elif e.event_type == "up" and state["cancel_down"]:
                state["cancel_down"] = False
                try:
                    recorder.cancel_current()
                except Exception as ex:
                    log("[ERR]", f"取消异常: {ex}")
            return
        if e.name != trigger:
            return
        if e.event_type == "down":
            state["down"] = True            # auto-repeat 'down' keeps this True
        elif e.event_type == "up" and state["down"]:
            state["down"] = False
            try:
                if not recorder.stop():
                    recorder.start()
                    recorder.arm_watchdog()
            except Exception as e:
                log("[ERR]", f"快捷键处理异常（已复位）: {e}")
                with recorder.lock:
                    recorder.gate = False

    return handler


def transcribe_file(path: str, client: ASRClient, args) -> Optional[str]:
    """Offline: transcribe one local file and print the text."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != RATE:
        log("[WARN]", f"输入采样率 {sr}Hz，将重采样到 {RATE}Hz")
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=RATE)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    audio = np.asarray(audio, dtype=np.float32)
    log("[TEST]", f"文件 {path}，时长 {len(audio) / RATE:.1f}s，转写中…")
    try:
        return client.transcribe(audio, model=args.model, language=args.language or None)
    except Exception as e:
        log("[ERR]", f"转写请求失败: {e}")
        return None


def build_parser() -> argparse.ArgumentParser:
    """CLI defaults come from config.py (which itself reads QWEN_ASR_VOICE_*)."""
    p = argparse.ArgumentParser(
        description="Qwen3-ASR local voice input (global hotkey, queue + silence splitting)")
    p.add_argument("--hotkey", default=str(_cfg("voice_hotkey", default="right ctrl")),
                   help="key toggling recording, e.g. 'right ctrl' (default) or 'ctrl+alt+f1'")
    p.add_argument("--cancel-hotkey", default=str(_cfg("voice_cancel_hotkey", default="")),
                   help="key discarding the chunk being recorded, e.g. 'right alt' "
                        "(default: disabled)")
    p.add_argument("--language", default=str(_cfg("voice_language", default="")) or None,
                   help="force language, e.g. Chinese (default: auto-detect)")
    p.add_argument("--model", default="qwen3-asr-0.6b", help="model id reported to the server")
    p.add_argument("--paste", action=argparse.BooleanOptionalAction,
                   default=bool(_cfg("voice_paste", default=True)),
                   help="auto-paste (Ctrl+V) at the cursor after transcribing (default: on)")
    p.add_argument("--restore-clipboard", action=argparse.BooleanOptionalAction,
                   default=bool(_cfg("voice_restore_clipboard", default=True)),
                   help="put the previous clipboard content back after pasting (default: on)")
    p.add_argument("--max-seconds", type=int,
                   default=int(_cfg("voice_max_seconds", default=300)),
                   help="auto-stop the whole session after this many seconds (default: 300)")
    p.add_argument("--device", default=str(_cfg("voice_device", default="")) or None,
                   help="input device index or name fragment, e.g. --device 1 or "
                        "--device bose (default: system default input)")

    g = p.add_argument_group("silence splitting (VAD)")
    g.add_argument("--vad", action=argparse.BooleanOptionalAction,
                   default=bool(_cfg("voice_vad", default=True)),
                   help="split on pauses and transcribe while recording (default: on)")
    g.add_argument("--adaptive-noise", action=argparse.BooleanOptionalAction,
                   default=bool(_cfg("voice_adaptive_noise", default=True)),
                   help="measure the noise floor and set the gates from it "
                        "(default: on; --no-adaptive-noise uses the fixed dB values)")
    g.add_argument("--silence-db", type=float,
                   default=_cfg("voice_silence_db", default=-45.0),
                   help="fixed mode: level below this counts as a pause "
                        "(-45 dBFS; raise towards -40 on a noisy mic)")
    g.add_argument("--speech-db", type=float,
                   default=_cfg("voice_speech_db", default=-38.0),
                   help="fixed mode: level above this counts as speech (-38 dBFS)")
    g.add_argument("--silence-hold", type=float,
                   default=_cfg("voice_silence_hold", default=0.65),
                   help="pause length that closes a segment (default: 0.65s; "
                        "raise to 0.8-1.0 if you pause mid-sentence)")
    g.add_argument("--min-speech", type=float,
                   default=_cfg("voice_min_speech", default=0.30),
                   help="ignore speech shorter than this - clicks/coughs (default: 0.30s)")
    g.add_argument("--preroll", type=float,
                   default=_cfg("voice_preroll", default=0.35),
                   help="audio kept from before speech is detected (default: 0.35s)")
    g.add_argument("--max-segment", type=float,
                   default=_cfg("voice_max_segment", default=30.0),
                   help="force-split non-stop speech at this length (default: 30s)")
    g.add_argument("--min-segment", type=float,
                   default=_cfg("voice_min_segment", default=0.35),
                   help="drop segments shorter than this (default: 0.35s)")

    q = p.add_argument_group("queue")
    q.add_argument("--queue-workers", type=int,
                   default=int(_cfg("voice_queue_workers", default=1)),
                   help="background transcription threads (default: 1; the server "
                        "serialises CPU inference, so more rarely helps)")
    q.add_argument("--queue-max", type=int,
                   default=int(_cfg("voice_queue_max", default=64)),
                   help="max queued segments before the oldest is dropped (default: 64)")
    q.add_argument("--backlog-warn", type=int,
                   default=int(_cfg("voice_backlog_warn", default=5)),
                   help="warn when this many segments are pending (default: 5)")
    q.add_argument("--retry", type=int,
                   default=int(_cfg("voice_retry", default=2)),
                   help="retries per segment when the server fails (default: 2)")

    r = p.add_argument_group("result safety net")
    r.add_argument("--log-transcripts", action=argparse.BooleanOptionalAction,
                   default=bool(_cfg("voice_log_transcripts", default=True)),
                   help="append every result to a daily transcript log (default: on)")
    r.add_argument("--transcripts-dir",
                   default=str(_cfg("voice_transcripts_dir",
                                    default=os.path.join(os.path.dirname(
                                        os.path.abspath(__file__)), "transcripts"))),
                   help="where the transcript log is written")
    r.add_argument("--failed-audio-dir",
                   default=str(_cfg("voice_failed_audio_dir",
                                    default=os.path.join(os.path.dirname(
                                        os.path.abspath(__file__)), "failed_audio"))),
                   help="where audio of failed segments is saved as .wav")
    r.add_argument("--status-title", action=argparse.BooleanOptionalAction, default=True,
                   help="show recording state and backlog in the window title (default: on)")
    r.add_argument("--auto-gain", action=argparse.BooleanOptionalAction,
                   default=bool(_cfg("voice_auto_gain", default=True)),
                   help="normalise each segment's level before sending it "
                        "(default: on; needed when the device has auto gain control)")
    r.add_argument("--gain-target-db", type=float,
                   default=_cfg("voice_gain_target_db", default=-22.0),
                   help="target RMS level for normalisation (default: -22 dBFS)")
    r.add_argument("--fallback-language",
                   default=str(_cfg("voice_fallback_language", default="Chinese")),
                   help="if auto-detect returns nothing, retry with this language "
                        "(default: Chinese; empty disables the retry)")

    p.add_argument("--test-file", type=str, default=None,
                   help="transcribe a local audio file and exit (no hotkey)")
    p.add_argument("--list-devices", action="store_true",
                   help="list input devices and exit")
    p.add_argument("--dump-config", action="store_true",
                   help="print the effective settings and exit")
    return p


def sanitize_args(args) -> None:
    """Keep CLI/env values usable even when a human typos one."""
    args.max_seconds = max(5, int(_finite(args.max_seconds, 300)))
    args.queue_workers = max(1, min(4, int(_finite(args.queue_workers, 1))))
    args.queue_max = max(1, int(_finite(args.queue_max, 64)))
    args.backlog_warn = max(1, int(_finite(args.backlog_warn, 5)))
    args.retry = max(0, min(5, int(_finite(args.retry, 2))))
    args.silence_hold = max(0.10, _finite(args.silence_hold, 0.65))
    args.min_speech = max(0.05, _finite(args.min_speech, 0.30))
    args.preroll = max(0.0, min(2.0, _finite(args.preroll, 0.35)))
    args.max_segment = max(2.0, _finite(args.max_segment, 30.0))
    args.min_segment = max(0.10, _finite(args.min_segment, 0.35))
    args.silence_db = _finite(args.silence_db, -45.0)
    args.speech_db = _finite(args.speech_db, -38.0)
    args.gain_target_db = max(-40.0, min(-6.0, _finite(args.gain_target_db, -22.0)))
    if args.silence_db > args.speech_db:
        args.silence_db = args.speech_db      # keep the gate consistent


CONFIG_KEYS = ("hotkey", "cancel_hotkey", "language", "model", "paste",
               "restore_clipboard", "max_seconds", "device", "vad",
               "adaptive_noise", "silence_db", "speech_db", "silence_hold",
               "min_speech", "preroll", "max_segment", "min_segment",
               "queue_workers", "queue_max", "backlog_warn", "retry",
               "log_transcripts", "transcripts_dir", "failed_audio_dir",
               "status_title", "auto_gain", "gain_target_db",
               "fallback_language")


def dump_config(args) -> None:
    print("有效配置（命令行 > 环境变量 > config.py 默认值）:")
    for key in CONFIG_KEYS:
        print(f"  {key:18} = {getattr(args, key)!r}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sanitize_args(args)

    if args.dump_config:
        dump_config(args)
        return

    if args.list_devices:
        list_devices()
        return

    client = ASRClient(timeout=300,
                       auto_gain=bool(args.auto_gain),
                       gain_target_db=float(args.gain_target_db),
                       fallback_language=args.fallback_language or None)

    # offline test mode
    if args.test_file:
        text = transcribe_file(args.test_file, client, args)
        print(f"[RESULT] {text}" if text is not None else "[RESULT] 失败", flush=True)
        return

    if not client.wait_ready():
        import sys
        sys.exit(1)

    # microphone sanity check
    try:
        devs = sd.query_devices()
        if not [d for d in devs if d.get("max_input_channels", 0) > 0]:
            log("[WARN]", "未检测到麦克风输入设备")
    except Exception:
        pass

    recorder = Recorder(args, client)
    recorder.queue.start()

    trigger, mods = _parse_hotkey(args.hotkey)
    cancel_trigger, _ = _parse_hotkey(args.cancel_hotkey)

    import keyboard as kb
    kb.hook(build_hotkey_handler(trigger, mods, recorder, cancel_trigger))

    print("=" * 60)
    print(f"  按 [{args.hotkey}] 开始录音，再按一次停止")
    print("  （在松开按键时触发：长按不会误触发、不会卡住）")
    if cancel_trigger:
        print(f"  说错了按 [{args.cancel_hotkey}] 丢弃当前这段（已排队的不受影响）")
    if args.vad:
        print(f"  停顿 {args.silence_hold:.2f}s 自动切段并排队转写")
        print("  转写在后台进行：录音不会被转写阻塞，可以连续说、连续切")
    else:
        print("  静音切段已关闭：停止时才提交整段录音")
    print(f"  每段转写完成后自动"
          f"{'粘贴到光标处' if args.paste else '复制到剪贴板（未开启自动粘贴）'}"
          f"{'，并还原原剪贴板' if (args.paste and args.restore_clipboard) else ''}")
    if args.log_transcripts:
        print(f"  结果同时追加到 {args.transcripts_dir}（转写失败会保留音频）")
    if args.auto_gain:
        print(f"  音量偏轻会自动提升到 {args.gain_target_db:.0f} dBFS 再送模型"
              f"（麦克风自动增益环境下必需）")
    if args.fallback_language:
        print(f"  自动识别没出文字时，会用 {args.fallback_language} 再试一次")
    print(f"  录音超过 {args.max_seconds}s 自动停止")
    print("  退出：Ctrl+C（会先把队列转写完）")
    print("=" * 60, flush=True)
    recorder._refresh_title()

    stop_event = threading.Event()

    def _on_signal(signum, frame):
        log("[signal]", "收到退出信号，正在收尾…")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _on_signal)
        except Exception:
            pass

    try:
        while not stop_event.wait(0.5):
            pass
    finally:
        recorder.stop()
        recorder.drain("退出")
        recorder.queue.shutdown()
        print("[DONE] " + recorder.status_line(), flush=True)


if __name__ == "__main__":
    main()
