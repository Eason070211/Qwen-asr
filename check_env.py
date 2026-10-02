"""
Environment smoke test for Qwen3-ASR server.

Checks:
  1. Python version
  2. Critical imports (torch, transformers, qwen_asr, fastapi, etc.)
  3. Model weights presence
  4. Audio codec availability
  5. Quick inference test (optional — pass --quick-test)

Usage:
    python check_env.py
    python check_env.py --quick-test   # loads model + runs one sample
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

OK = "\033[32mOK\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


def check(label: str, ok: bool, detail: str = ""):
    status = OK if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return ok


def check_imports() -> bool:
    print("\n== imports ==")
    all_ok = True
    critical = {
        "torch": "Core ML framework",
        "transformers": "HuggingFace transformers",
        "accelerate": "Device management",
        "qwen_asr": "Qwen3-ASR package",
        "soundfile": "Audio I/O",
        "fastapi": "API server",
        "uvicorn": "ASGI server",
        "modelscope": "Model downloader",
    }
    optional = {
        "librosa": "Audio resampling (fallback available)",
        "websockets": "WebSocket streaming",
        "requests": "HTTP client",
    }
    for pkg, desc in critical.items():
        try:
            m = importlib.import_module(pkg)
            ver = getattr(m, "__version__", "?")
            check(f"{pkg:20s} {ver:10s}  {desc}", True)
        except ImportError as e:
            check(f"{pkg:20s} {'(missing)':10s}  {desc}", False, str(e))
            all_ok = False
    for pkg, desc in optional.items():
        try:
            m = importlib.import_module(pkg)
            ver = getattr(m, "__version__", "?")
            check(f"{pkg:20s} {ver:10s}  {desc}", True)
        except ImportError:
            check(f"{pkg:20s} {'(missing)':10s}  {desc}", False, "optional, skipping")
    return all_ok


def check_model() -> bool:
    print("\n== model weights ==")
    models_dir = Path(__file__).resolve().parent / "models"
    if not models_dir.exists():
        check("models/ directory", False, "not found — run download_model.py")
        return False

    found = False
    for name in ["Qwen3-ASR-0.6B", "Qwen3-ASR-0.6B-hf", "Qwen3-ASR-1.7B",
                  "Qwen3-ForcedAligner-0.6B"]:
        d = models_dir / name
        if d.exists():
            safetensors = list(d.glob("*.safetensors"))
            config = d / "config.json"
            has_weights = len(safetensors) > 0
            has_config = config.exists()
            size_gb = sum(f.stat().st_size for f in safetensors) / (1024**3)
            ok = has_weights and has_config
            detail = f"{size_gb:.2f} GB, config={'yes' if has_config else 'no'}"
            check(f"{name:35s}", ok, detail)
            if ok:
                found = True

    if not found:
        print("\n  No valid model found.")
        print("  Run:  .venv\\Scripts\\python download_model.py --model qwen3-asr-0.6b")
    return found


def check_audio() -> bool:
    print("\n== audio codecs (soundfile) ==")
    try:
        import soundfile as sf
        formats = sorted(sf.available_formats().keys())
        key_fmts = [f for f in ["WAV", "FLAC", "MP3", "OGG"] if f in formats]
        check(f"soundfile {sf.__version__}", True, f"formats: {', '.join(key_fmts)}")
        return True
    except Exception as e:
        check("soundfile", False, str(e))
        return False


def check_python() -> bool:
    print("\n== python ==")
    v = sys.version
    major, minor = sys.version_info[:2]
    ok = minor >= 9  # qwen-asr requires >=3.9
    check(f"Python {v}", ok)
    return ok


def quick_test():
    print("\n== quick test (load model + transcribe sample) ==")
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
        import numpy as np

        models_dir = Path(__file__).resolve().parent / "models"
        # prefer -hf for direct transformers compat
        candidates = ["Qwen3-ASR-0.6B-hf", "Qwen3-ASR-0.6B"]
        model_path = None
        for c in candidates:
            if (models_dir / c).exists():
                model_path = models_dir / c
                break

        if not model_path:
            check("model loading", False, "no model weights found")
            return

        print(f"  Loading {model_path} ...")
        model = Qwen3ASRModel.from_pretrained(
            str(model_path),
            dtype=torch.float32,
            device_map="cpu",
            max_new_tokens=128,
        )
        check("model loading", True)

        # synthesize 1 second of silence
        silence = np.zeros(16000, dtype=np.float32)
        print("  Running transcribe on 1s silence ...")
        result = model.transcribe(audio=(silence, 16000), language=None)
        check("transcription", True, f"text='{result.text[:60]}'")

    except Exception as e:
        check("quick test", False, str(e))


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR environment check")
    parser.add_argument("--quick-test", action="store_true",
                        help="Load model and run a sample (takes ~30-60s)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Qwen3-ASR environment check")
    print("=" * 60)

    py_ok = check_python()
    imp_ok = check_imports()
    mod_ok = check_model()
    aud_ok = check_audio()

    print("\n== summary ==")
    if py_ok and imp_ok and mod_ok and aud_ok:
        print("  Environment looks good!")
    else:
        print("  Some checks failed. See above for details.")

    if args.quick_test:
        if not mod_ok:
            print("\n  Skipping quick test — no model weights.")
        else:
            quick_test()

    print()


if __name__ == "__main__":
    main()
