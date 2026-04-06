#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    except ImportError as exc: 
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
        ) from exc
    
    from compel import Compel

    model_id_env = (os.getenv("HF_DIFFUSERS_MODEL_ID", "") or "").strip()
    model_id = model_id_env or "runwayml/stable-diffusion-v1-5"
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

    compel = Compel(
        tokenizer=text2img.tokenizer,
        text_encoder=text2img.text_encoder,
    )


    return text2img, img2img, compel


def _generate_with_diffusers(
    args: argparse.Namespace,
    device: str,
    prompts: list[str],
    negative_prompts: list[str],
    cfg_schedule: list[float],
    steps_schedule: list[int],
    seeds: list[int],
    run_outdir: Path,
    timestamp: str,
    input_image: Image.Image | None,
    input_images: list[str] | None = None,
):
    text2img, img2img, compel = _prepare_diffusers_pipelines(device=device, sampler_name=args.sampler)
    img_paths = input_images if input_images is not None else [""] * len(prompts)

    for prompt, neg_prompt, cfg_scale, n_steps, seed, img_path in zip(
        prompts, negative_prompts, cfg_schedule, steps_schedule, seeds, img_paths
    ):
        active_image = Image.open(img_path).convert("RGB") if img_path else input_image

        generator_device = device if device in {"cuda", "mps"} else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        guidance_scale = 9.0 if args.no_cfg else cfg_scale

        prompt_embeds = compel(prompt)
        neg_prompt_embeds = compel(neg_prompt)
        prompt_embeds, neg_prompt_embeds = compel.pad_conditioning_tensors_to_same_length([prompt_embeds, neg_prompt_embeds])

        if active_image is not None:
            result = img2img(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_prompt_embeds,
                image=active_image,
                strength=args.strength,
                guidance_scale=guidance_scale,
                num_inference_steps=n_steps,
                generator=generator,
            )
        else:
            result = text2img(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_prompt_embeds,
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
    prompts: list[str],
    negative_prompts: list[str],
    cfg_schedule: list[float],
    steps_schedule: list[int],
    seeds: list[int],
    run_outdir: Path,
    timestamp: str,
    input_image: Image.Image | None,
):
    sd_dir = repo_root / "data_generation_backend" / "diffusion_model" / "sd"
    data_dir = repo_root / "data_generation_backend" / "diffusion_model" / "data"

    vocab_path = data_dir / "vocab.json"
    merges_path = data_dir / "merges.txt"
    weights_path = "/Users/susheelutagi/Documents/ComfyUI/models/checkpoints/v1-5-pruned-emaonly.ckpt"

    # missing = [p for p in (vocab_path, merges_path, weights_path) if not p.exists()]
    # if missing:
    #     missing_str = ", ".join(str(p) for p in missing)
    #     raise FileNotFoundError(
    #         f"Missing diffusion resources: {missing_str}. Run ./scripts/get_resources.sh to download them."
    #     )

    if str(sd_dir) not in sys.path:
        sys.path.insert(0, str(sd_dir))

    import model_loader  # type: ignore
    import pipeline      # type: ignore

    tokenizer = CLIPTokenizer(str(vocab_path), merges_file=str(merges_path))
    models = model_loader.preload_models_from_standard_weights(str(weights_path), device)
    do_cfg = not args.no_cfg

    for prompt, neg_prompt, cfg_scale, n_steps, seed in zip(
        prompts, negative_prompts, cfg_schedule, steps_schedule, seeds
    ):
        output_array = pipeline.generate(
            prompt=prompt,
            uncond_prompt=neg_prompt,
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
    parser.add_argument("--prompt", help="Text prompt (ignored when --from-json is used).", required=False)
    parser.add_argument("--negative-prompt", default="", help="Negative prompt (unconditional prompt).")
    parser.add_argument("--from-json",default=None,help="Path to a prompts JSON file; generates one image per sample in the file.",)
    parser.add_argument("--outdir", default=None, help="Output directory (default: data_generation_outputs).")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate.")
    parser.add_argument("--seed",type=int,default=None,help="Base seed (each image increments by 1). If omitted, uses a random base seed per run.",)
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps.")
    parser.add_argument("--sampler", default="ddim", help="Sampler name: choose ddpm or ddim.")
    parser.add_argument("--cfg-scale", type=float, default=8.0, help="Classifier-free guidance scale.")
    parser.add_argument("--sweep-cfg",action="store_true",help="Sweep integer cfg scales 5–13 (inclusive) across images (requires at least 9 images; remainder is distributed unevenly).")
    parser.add_argument("--sweep-num-steps",action="store_true",help="Sweep inference steps 10–50 in increments of 5 across images (requires at least 9 images",)
    parser.add_argument("--no-cfg", action="store_true", help="Disable classifier-free guidance.")
    parser.add_argument("--strength", type=float, default=0.7, help="img2img strength (0-1). Ignored for txt2img.")
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
    outdir = Path(args.outdir) if args.outdir else (repo_root / "data_generation_outputs")

    init_images: list[str] | None = None

    if args.from_json:
        if args.sweep_cfg or args.sweep_num_steps:
            raise ValueError("--from-json cannot be combined with sweep flags.")
        json_path = Path(args.from_json)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        coco_class = data.get("coco_class")
        samples = data.get("samples")


        if not coco_class or not isinstance(coco_class, str): raise ValueError("JSON must include string field 'coco_class'.")
        if not isinstance(samples, list) or not samples: raise ValueError("JSON must include non-empty list field 'samples'.")


        prompts: list[str] = []
        negative_prompts: list[str] = []
        cfg_schedule: list[float] = []
        steps_schedule: list[int] = []
        init_images: list[str] = []

        for idx, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(f"Sample at index {idx} is not an object.")
            try:
                prompts.append(sample["prompt"])
            except KeyError as exc:
                raise ValueError(f"Sample {idx} missing required key 'prompt'.") from exc
            negative_prompts.append(sample.get("negative_prompt", args.negative_prompt))
            cfg_schedule.append(float(sample.get("cfg_scale", args.cfg_scale)))
            steps_schedule.append(int(sample.get("steps", args.steps)))
            init_images.append(sample.get("init_image", ""))

        num_images = len(prompts)
        outdir = outdir / coco_class
        run_outdir = outdir
        run_outdir.mkdir(parents=True, exist_ok=True)
    else:
        if args.prompt is None:
            raise ValueError("--prompt is required when not using --from-json.")
        num_images = args.num_images
        cfg_schedule, steps_schedule = _build_schedules(args)
        prompts = [args.prompt] * num_images
        negative_prompts = [args.negative_prompt] * num_images
        run_outdir = (outdir / timestamp) if num_images > 1 else outdir
        run_outdir.mkdir(parents=True, exist_ok=True)

    seed_base = args.seed
    if seed_base is None:
        seed_base = int(torch.randint(0, 2**31 - 1, (1,)).item())
    seeds = [seed_base + i for i in range(num_images)]



    if use_hg_diffusers:
        print("USE_HG_DIFFUSERS=true -> using Hugging Face diffusers backend (runwayml/stable-diffusion-v1-5).")
        _generate_with_diffusers(
            args=args,
            device=device,
            prompts=prompts,
            negative_prompts=negative_prompts,
            cfg_schedule=cfg_schedule,
            steps_schedule=steps_schedule,
            seeds=seeds,
            run_outdir=run_outdir,
            timestamp=timestamp,
            input_image=input_image,
            input_images=init_images,
        )
    else:
        print("USE_HG_DIFFUSERS not set/false -> using local Stable Diffusion backend (diffusion_model).")
        _generate_with_local_sd(
            args=args,
            repo_root=repo_root,
            device=device,
            prompts=prompts,
            negative_prompts=negative_prompts,
            cfg_schedule=cfg_schedule,
            steps_schedule=steps_schedule,
            seeds=seeds,
            run_outdir=run_outdir,
            timestamp=timestamp,
            input_image=input_image,
            input_images=init_images,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
