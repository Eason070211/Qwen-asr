"""Stress test: is the ASR server safe under concurrent requests?

Fires N requests at once and reports success/failure and timing. A server that
serialises inference should answer all of them, with wall time roughly equal to
N x single-request time. Any connection reset / 500 means concurrency is unsafe.
"""
from __future__ import annotations

import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import soundfile as sf

BASE = "http://127.0.0.1:8910"
RATE = 16000
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SECONDS = 2.0

rng = np.random.default_rng(3)
t = np.arange(int(SECONDS * RATE)) / RATE
env = 0.5 * (1 + np.sin(2 * np.pi * 2.5 * t))
sig = env * (np.sin(2 * np.pi * 140 * t) + 0.5 * np.sin(2 * np.pi * 280 * t))
sig = (sig / (np.sqrt(np.mean(sig ** 2))) * 10 ** (-20 / 20)).astype(np.float32)
wav = io.BytesIO()
sf.write(wav, sig, RATE, format="WAV", subtype="PCM_16")
payload = wav.getvalue()

print(f"health: ", end="", flush=True)
try:
    print(requests.get(BASE + "/health", timeout=10).json().get("status"))
except Exception as e:
    print("DOWN", e)
    sys.exit(2)


def one(i: int):
    t0 = time.time()
    try:
        r = requests.post(
            BASE + "/v1/audio/transcriptions",
            files={"file": (f"s{i}.wav", payload, "audio/wav")},
            data={"model": "qwen3-asr-0.6b"},
            timeout=120,
        )
        return i, r.status_code, time.time() - t0, (r.text or "")[:40]
    except Exception as e:
        return i, "EXC", time.time() - t0, f"{type(e).__name__}: {e}"


print(f"--- {N} concurrent requests ({SECONDS}s audio each) ---")
wall0 = time.time()
with ThreadPoolExecutor(max_workers=N) as ex:
    results = list(ex.map(one, range(N)))
wall = time.time() - wall0

ok = sum(1 for r in results if r[1] == 200)
print(f"  succeeded: {ok}/{N}   wall={wall:.2f}s")
for i, status, dt, extra in sorted(results, key=lambda r: r[0]):
    print(f"    #{i} status={status} {dt:5.2f}s  {extra}")

# is the server still alive afterwards?
time.sleep(0.5)
try:
    st = requests.get(BASE + "/health", timeout=10).json().get("status")
    print(f"  server after burst: {st}")
except Exception as e:
    print(f"  server after burst: DEAD ({type(e).__name__})")
    sys.exit(3)

sys.exit(0 if ok == N else 1)
