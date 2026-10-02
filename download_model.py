"""
Download Qwen3-ASR weights.

Default source is Hugging Face (works from anywhere with open network).
ModelScope is kept as a fallback for machines inside mainland China where
hf.co may be slow/blocked.

Usage:
    python download_model.py                          # default: 0.6B-hf
    python download_model.py --model qwen3-asr-0.6b-hf
    python download_model.py --all                    # download all
    python download_model.py --list                   # show available
    python download_model.py --source modelscope      # force ModelScope

Models are saved under ./models/<local_dir>/  so that config.py and
check_env.py can find them without changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODELS = {
    "qwen3-asr-0.6b": {
        "hf_id": "Qwen/Qwen3-ASR-0.6B",
        "scope_id": "Qwen/Qwen3-ASR-0.6B",
        "local_dir": "Qwen3-ASR-0.6B",
        "description": "Qwen3-ASR 0.6B, native format (DEFAULT; matches qwen-asr 0.0.6 + transformers 4.x, ~1.88 GB)",
    },
    "qwen3-asr-0.6b-hf": {
        "hf_id": "Qwen/Qwen3-ASR-0.6B-hf",
        "scope_id": "Qwen/Qwen3-ASR-0.6B-hf",
        "local_dir": "Qwen3-ASR-0.6B-hf",
        "description": "Qwen3-ASR 0.6B, HF format (requires transformers 5.x; NOT compatible with qwen-asr 0.0.6, ~1.58 GB)",
    },
    "qwen3-asr-1.7b": {
        "hf_id": "Qwen/Qwen3-ASR-1.7B",
        "scope_id": "Qwen/Qwen3-ASR-1.7B",
        "local_dir": "Qwen3-ASR-1.7B",
        "description": "Qwen3-ASR 1.7B, native format (~4.70 GB, slow on CPU)",
    },
    "qwen3-forced-aligner-0.6b": {
        "hf_id": "Qwen/Qwen3-ForcedAligner-0.6B",
        "scope_id": "Qwen/Qwen3-ForcedAligner-0.6B",
        "local_dir": "Qwen3-ForcedAligner-0.6B",
        "description": "Forced aligner for timestamps, native format (~1.84 GB, optional)",
    },
}

DEFAULT_MODEL = "qwen3-asr-0.6b"


def list_models():
    print("Available models:\n")
    for name, info in MODELS.items():
        default = "  <-- default" if name == DEFAULT_MODEL else ""
        print(f"  {name:32s} {info['description']}{default}")
        print(f"  {'':32s} HF:        {info['hf_id']}")
        print(f"  {'':32s} ModelScope: {info['scope_id']}")
        print()
    print("Usage:")
    print("  python download_model.py")
    print("  python download_model.py --model qwen3-asr-0.6b-hf")
    print("  python download_model.py --all")
    print("  python download_model.py --source modelscope   # fallback mirror")
    print()
    print("NOTE: default source is Hugging Face (huggingface.co).")
    print("      Use `--source modelscope` when hf.co is unreachable (e.g. mainland China).")


def download(name: str, output_dir: Path, source: str):
    if name not in MODELS:
        print(f"Unknown model: {name}", file=sys.stderr)
        print(f"Valid names: {', '.join(MODELS.keys())}", file=sys.stderr)
        sys.exit(1)

    info = MODELS[name]
    dest = output_dir / info["local_dir"]

    if dest.exists() and any(dest.iterdir()):
        print(f"Model already exists: {dest}")
        print(f"  Delete it first if you want to re-download.")
        return

    print(f"Downloading {name} ...")
    print(f"  Source:  {'Hugging Face' if source == 'hf' else 'ModelScope'} ({info[source + '_id']})")
    print(f"  Target:  {dest}")
    print()

    try:
        if source == "hf":
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=info["hf_id"],
                local_dir=str(dest),
            )
        else:
            from modelscope.hub.snapshot_download import snapshot_download
            snapshot_download(
                model_id=info["scope_id"],
                local_dir=str(dest),
            )
        print(f"\nDone. Model saved to: {dest}")
    except Exception as e:
        print(f"\nDownload failed: {e}", file=sys.stderr)
        if source == "hf":
            print("If huggingface.co is unreachable from your network, retry with:", file=sys.stderr)
            print("  python download_model.py --model %s --source modelscope" % name, file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download Qwen3-ASR weights")
    parser.add_argument("--model", choices=list(MODELS.keys()), default=DEFAULT_MODEL,
                        help=f"Which model to download (default: {DEFAULT_MODEL})")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--source", choices=["hf", "modelscope"], default="hf",
                        help="Download source (default: hf = huggingface.co)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ./models)")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        for name in MODELS:
            download(name, output_dir, args.source)
    else:
        download(args.model, output_dir, args.source)


if __name__ == "__main__":
    main()
