#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image
from transformers import CLIPTokenizer
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _select_device(allow_cuda: bool, allow_mps: bool) -> str:
    device = "cpu"
    if torch.cuda.is_available() and allow_cuda:
        device = "cuda"
    elif (torch.has_mps or torch.backends.mps.is_available()) and allow_mps:
        device = "mps"
    return device


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate images with the local SD v1.5 diffusion model.")
    parser.add_argument("--prompt", required=True, help="Text prompt.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt (unconditional prompt).")
    parser.add_argument("--outdir", default=None, help="Output directory (default: data-generation-outputs/diffusion).")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed (each image increments by 1). If omitted, uses a random seed.",
    )
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps.")
    parser.add_argument("--sampler", default="ddpm", help="Sampler name (only 'ddpm' is supported by this repo).")
    parser.add_argument("--cfg-scale", type=float, default=8.0, help="Classifier-free guidance scale.")
    parser.add_argument("--no-cfg", action="store_true", help="Disable classifier-free guidance.")
    parser.add_argument("--strength", type=float, default=0.9, help="img2img strength (0-1). Ignored for txt2img.")
    parser.add_argument("--init-image", default=None, help="Optional path to an input image for img2img.")
    parser.add_argument("--allow-cuda", action="store_true", help="Allow CUDA if available.")
    parser.add_argument("--no-mps", action="store_true", help="Disable Apple MPS even if available.")
    args = parser.parse_args()

    repo_root = _repo_root()
    sd_dir = repo_root / "diffusion-model" / "sd"
    data_dir = repo_root / "diffusion-model" / "data"

    vocab_path = data_dir / "vocab.json"
    merges_path = data_dir / "merges.txt"
    weights_path = data_dir / "v1-5-pruned-emaonly.ckpt"

    missing = [p for p in (vocab_path, merges_path, weights_path) if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing diffusion resources: {missing_str}. "
            "Run ./download_sd15_resources.sh to download them."
        )

    if str(sd_dir) not in sys.path:
        sys.path.insert(0, str(sd_dir))

    import model_loader  # type: ignore
    import pipeline  # type: ignore

    device = _select_device(allow_cuda=args.allow_cuda, allow_mps=not args.no_mps)
    print(f"Using device: {device}")

    tokenizer = CLIPTokenizer(str(vocab_path), merges_file=str(merges_path))
    models = model_loader.preload_models_from_standard_weights(str(weights_path), device)

    outdir = Path(args.outdir) if args.outdir else (repo_root / "data-generation-outputs" / "diffusion")
    outdir.mkdir(parents=True, exist_ok=True)

    input_image = Image.open(args.init_image) if args.init_image else None
    do_cfg = not args.no_cfg

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(args.num_images):
        seed = (args.seed + i) if args.seed is not None else None
        output_array = pipeline.generate(
            prompt=args.prompt,
            uncond_prompt=args.negative_prompt,
            input_image=input_image,
            strength=args.strength,
            do_cfg=do_cfg,
            cfg_scale=args.cfg_scale,
            sampler_name=args.sampler,
            n_inference_steps=args.steps,
            seed=seed,
            models=models,
            device=device,
            idle_device="cpu",
            tokenizer=tokenizer,
        )
        image = Image.fromarray(output_array)
        seed_suffix = f"seed{seed}" if seed is not None else "seed_random"
        outpath = outdir / f"sd15_{timestamp}_{seed_suffix}.png"
        image.save(outpath)
        print(f"Wrote {outpath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
