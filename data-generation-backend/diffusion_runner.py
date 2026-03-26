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
    parser.add_argument("--outdir", default=None, help="Output directory (default: data-generation-outputs).")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed (each image increments by 1). If omitted, uses a random base seed per run.",
    )
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps.")
    parser.add_argument("--sampler", default="ddpm", help="Sampler name (only 'ddpm' is supported by this repo).")
    parser.add_argument("--cfg-scale", type=float, default=8.0, help="Classifier-free guidance scale.")
    parser.add_argument(
        "--sweep-cfg",
        action="store_true",
        help="Sweep integer cfg scales 5–13 (inclusive) across images (requires at least 9 images; remainder is distributed unevenly).",
    )
    parser.add_argument(
        "--sweep-num-steps",
        action="store_true",
        help="Sweep inference steps 10–50 in increments of 5 across images (requires at least 9 images; remainder is distributed unevenly). When combined with --sweep-cfg, performs a full cfg/steps grid (9×9).",
    )
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
            "Run ./get_resources.sh to download them."
        )

    if str(sd_dir) not in sys.path:
        sys.path.insert(0, str(sd_dir))

    import model_loader  # type: ignore
    import pipeline  # type: ignore

    device = _select_device(allow_cuda=args.allow_cuda, allow_mps=not args.no_mps)
    print(f"Using device: {device}")

    tokenizer = CLIPTokenizer(str(vocab_path), merges_file=str(merges_path))
    models = model_loader.preload_models_from_standard_weights(str(weights_path), device)

    input_image = Image.open(args.init_image) if args.init_image else None
    do_cfg = not args.no_cfg

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else (repo_root / "data-generation-outputs")
    run_outdir = (outdir / timestamp) if args.num_images > 1 else outdir
    run_outdir.mkdir(parents=True, exist_ok=True)

    seed_base = args.seed
    if seed_base is None:
        seed_base = int(torch.randint(0, 2**31 - 1, (1,)).item())

    cfg_levels = [float(cfg) for cfg in range(5, 14)]          # 5…13
    step_levels = list(range(10, 51, 5))                       # 10,15,…,50

    if args.sweep_cfg and args.sweep_num_steps:
        grid = [(c, s) for c in cfg_levels for s in step_levels]  # cartesian grid
        grid_size = len(grid)
        if args.num_images < grid_size:
            raise ValueError(
                f"--sweep-cfg and --sweep-num-steps together require at least {grid_size} images to cover all cfg/step pairs once."
            )
        repeats = (args.num_images + grid_size - 1) // grid_size
        schedule = (grid * repeats)[: args.num_images]
        cfg_schedule = [c for c, _ in schedule]
        steps_schedule = [s for _, s in schedule]
        if args.no_cfg:
            raise ValueError("--sweep-cfg cannot be combined with --no-cfg.")
    elif args.sweep_cfg:
        sweep_steps = len(cfg_levels)
        if args.num_images < sweep_steps:
            raise ValueError(f"--sweep-cfg requires at least {sweep_steps} images to cover integer cfg 5–13 evenly.")
        if args.no_cfg:
            raise ValueError("--sweep-cfg cannot be combined with --no-cfg.")
        repeats = (args.num_images + sweep_steps - 1) // sweep_steps
        cfg_schedule = (cfg_levels * repeats)[: args.num_images]
        steps_schedule = [args.steps] * args.num_images
    elif args.sweep_num_steps:
        step_count = len(step_levels)
        if args.num_images < step_count:
            raise ValueError(f"--sweep-num-steps requires at least {step_count} images to cover step counts 10–50.")
        repeats_steps = (args.num_images + step_count - 1) // step_count
        steps_schedule = (step_levels * repeats_steps)[: args.num_images]
        cfg_schedule = [args.cfg_scale] * args.num_images
    else:
        cfg_schedule = [args.cfg_scale] * args.num_images
        steps_schedule = [args.steps] * args.num_images

    for i, (cfg_scale, n_steps) in enumerate(zip(cfg_schedule, steps_schedule)):
        seed = seed_base + i
        output_array = pipeline.generate(
            prompt=args.prompt,
            uncond_prompt=args.negative_prompt,
            input_image=input_image,
            strength=args.strength,
            do_cfg=do_cfg,
            cfg_scale=cfg_scale,
            sampler_name=args.sampler,
            n_inference_steps=n_steps,
            seed=seed,
            models=models,
            device=device,
            idle_device="cpu",
            tokenizer=tokenizer,
        )
        image = Image.fromarray(output_array)
        seed_suffix = f"seed{seed}"
        cfg_suffix = f"cfg{cfg_scale:.1f}"
        steps_suffix = f"steps{n_steps}"
        outpath = run_outdir / f"{timestamp}_{seed_suffix}_{cfg_suffix}_{steps_suffix}.png"
        image.save(outpath)
        print(f"Wrote {outpath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
