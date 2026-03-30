"""Simple CLI image captioning tool.

Usage:
    python caption_cli.py /path/to/image.jpg
    python caption_cli.py /path/to/image.jpg --model nlpconnect/vit-gpt2-image-captioning --num-beams 5

The tool is intended for quick semantic adherence checks of generated images.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import torch
from PIL import Image
from transformers import pipeline
from dotenv import load_dotenv

# Load environment from project root (.env) so HF caches respect HF_HOME / TRANSFORMERS_CACHE
ENV_ROOT = Path(__file__).resolve().parents[2] / ".env"
if ENV_ROOT.exists():
    load_dotenv(ENV_ROOT)

# Default captioning model can be overridden via .env (CAPTION_MODEL_ID)
DEFAULT_MODEL_ID = os.getenv("CAPTION_MODEL_ID", "Salesforce/blip2-flan-t5-xl")


def load_pipe(model_id: str, device: str | None) -> "pipeline":
    device_arg = 0 if device == "cuda" else -1
    return pipeline(
        "image-to-text",
        model=model_id,
        device=device_arg,
    )


def generate_caption(pipe, image_path: Path, max_new_tokens: int | None = None) -> str:
    image = Image.open(image_path).convert("RGB")
    if max_new_tokens is not None:
        outputs: List[dict] = pipe(image, generate_kwargs={"max_new_tokens": max_new_tokens})
    else:
        outputs: List[dict] = pipe(image)
    # pipeline returns list of {"generated_text": "..."}
    return outputs[0]["generated_text"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple CLI image captioning tool")
    parser.add_argument("image", type=Path, help="Path to an image file")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id (defaults to CAPTION_MODEL_ID env, fallback to Salesforce/blip2-flan-t5-xl)",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                        help="Force device; default picks CUDA if available")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Limit newly generated tokens (optional)")
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_pipe(args.model, device if device == "cuda" else None)
    caption = generate_caption(pipe, args.image, max_new_tokens=args.max_new_tokens)

    print(f"\nImage: {args.image}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"\nCaption:\n{caption}\n")


if __name__ == "__main__":
    main()
