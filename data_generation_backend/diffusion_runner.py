#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image
from dotenv import load_dotenv
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


def _env_flag(var_name: str) -> bool:
    """Return True when the environment variable looks truthy."""

    val = os.getenv(var_name, "").strip().lower()
    return val == 'true'


def _build_schedules(args: argparse.Namespace) -> tuple[list[float], list[int]]:
    cfg_levels = [float(cfg) for cfg in range(5, 14)]  # 5…13
    step_levels = list(range(10, 51, 5))  # 10,15,…,50

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

    return cfg_schedule, steps_schedule


def _resolve_diffusers_scheduler(config, sampler_name: str):
    try:
        from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
    except ImportError as exc:  # pragma: no cover - handled earlier
        raise ImportError("The 'diffusers' package is required but not installed.") from exc

    name = sampler_name.lower()
    if name == "ddpm":
        return DDPMScheduler.from_config(config)
    if name == "ddim":
        return DDIMScheduler.from_config(config)
    if name in {"dpm", "dpmsolver", "dpm++", "dpmpp"}:
        return DPMSolverMultistepScheduler.from_config(config)
    return None


def _prepare_diffusers_pipelines(device: str, sampler_name: str):
    try:
        from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
    except ImportError as exc:
        raise RuntimeError(
            "USE_HG_DIFFUSERS is set but the 'diffusers' package is not installed. "
            "Install it with `pip install diffusers` or unset USE_HG_DIFFUSERS."
        ) from exc

    model_id = os.getenv("HF_DIFFUSERS_MODEL_ID", "runwayml/stable-diffusion-v1-5")
    # MPS can produce black outputs with float16; keep fp16 only on CUDA, fp32 elsewhere.
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    safety_enabled = _env_flag("ENABLE_SAFETY_CHECKER")

    pipeline_kwargs = {"torch_dtype": torch_dtype}
    if not safety_enabled:
        pipeline_kwargs["safety_checker"] = None
        pipeline_kwargs["feature_extractor"] = None

    text2img = StableDiffusionPipeline.from_pretrained(model_id, **pipeline_kwargs)
    scheduler = _resolve_diffusers_scheduler(text2img.scheduler.config, sampler_name)
    if scheduler is not None:
        text2img.scheduler = scheduler
    text2img = text2img.to(device)
    text2img.enable_attention_slicing()
    if hasattr(text2img, "enable_vae_slicing"):
        text2img.enable_vae_slicing()
    text2img.set_progress_bar_config(disable=True)

    img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
    scheduler_img = _resolve_diffusers_scheduler(img2img.scheduler.config, sampler_name)
    if scheduler_img is not None:
        img2img.scheduler = scheduler_img
    img2img = img2img.to(device)
    img2img.enable_attention_slicing()
    if hasattr(img2img, "enable_vae_slicing"):
        img2img.enable_vae_slicing()
    img2img.set_progress_bar_config(disable=True)

    return text2img, img2img


def _generate_with_diffusers(
    args: argparse.Namespace,
    device: str,
    cfg_schedule: list[float],
    steps_schedule: list[int],
    seed_base: int,
    run_outdir: Path,
    timestamp: str,
    input_image: Image.Image | None,
):
    text2img, img2img = _prepare_diffusers_pipelines(device=device, sampler_name=args.sampler)

    for i, (cfg_scale, n_steps) in enumerate(zip(cfg_schedule, steps_schedule)):
        seed = seed_base + i
        generator_device = device if device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        guidance_scale = 1.0 if args.no_cfg else cfg_scale

        if input_image is not None:
            result = img2img(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=input_image,
                strength=args.strength,
                guidance_scale=guidance_scale,
                num_inference_steps=n_steps,
                generator=generator,
            )
        else:
            result = text2img(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=n_steps,
                generator=generator,
            )

        image = result.images[0]
        seed_suffix = f"seed{seed}"
        cfg_suffix = f"cfg{cfg_scale:.1f}"
        steps_suffix = f"steps{n_steps}"
        outpath = run_outdir / f"{timestamp}_{seed_suffix}_{cfg_suffix}_{steps_suffix}.png"
        image.save(outpath)
        print(f"Wrote {outpath}")


def _generate_with_local_sd(
    args: argparse.Namespace,
    repo_root: Path,
    device: str,
    cfg_schedule: list[float],
    steps_schedule: list[int],
    seed_base: int,
    run_outdir: Path,
    timestamp: str,
    input_image: Image.Image | None,
):
    sd_dir = repo_root / "diffusion_model" / "sd"
    data_dir = repo_root / "diffusion_model" / "data"

    vocab_path = data_dir / "vocab.json"
    merges_path = data_dir / "merges.txt"
    weights_path = data_dir / "v1-5-pruned-emaonly.ckpt"

    missing = [p for p in (vocab_path, merges_path, weights_path) if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing diffusion resources: {missing_str}. Run ./scripts/get_resources.sh to download them."
        )

    if str(sd_dir) not in sys.path:
        sys.path.insert(0, str(sd_dir))

    import model_loader  # type: ignore
    import pipeline  # type: ignore

    tokenizer = CLIPTokenizer(str(vocab_path), merges_file=str(merges_path))
    models = model_loader.preload_models_from_standard_weights(str(weights_path), device)
    do_cfg = not args.no_cfg

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


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion v1.5.")
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

    use_hg_diffusers = _env_flag("USE_HG_DIFFUSERS")
    repo_root = _repo_root()

    device = _select_device(allow_cuda=args.allow_cuda, allow_mps=not args.no_mps)
    print(f"Using device: {device}")

    input_image = Image.open(args.init_image).convert("RGB") if args.init_image else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else (repo_root / "data-generation-outputs")
    run_outdir = (outdir / timestamp) if args.num_images > 1 else outdir
    run_outdir.mkdir(parents=True, exist_ok=True)

    seed_base = args.seed
    if seed_base is None:
        seed_base = int(torch.randint(0, 2**31 - 1, (1,)).item())

    cfg_schedule, steps_schedule = _build_schedules(args)

    if use_hg_diffusers:
        print("USE_HG_DIFFUSERS=true -> using Hugging Face diffusers backend (runwayml/stable-diffusion-v1-5).")
        _generate_with_diffusers(
            args=args,
            device=device,
            cfg_schedule=cfg_schedule,
            steps_schedule=steps_schedule,
            seed_base=seed_base,
            run_outdir=run_outdir,
            timestamp=timestamp,
            input_image=input_image,
        )
    else:
        print("USE_HG_DIFFUSERS not set/false -> using local Stable Diffusion backend (diffusion_model).")
        _generate_with_local_sd(
            args=args,
            repo_root=repo_root,
            device=device,
            cfg_schedule=cfg_schedule,
            steps_schedule=steps_schedule,
            seed_base=seed_base,
            run_outdir=run_outdir,
            timestamp=timestamp,
            input_image=input_image,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
