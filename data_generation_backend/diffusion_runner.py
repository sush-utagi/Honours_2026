#!/usr/bin/env python3
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import CLIPTokenizer
import torch

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def polygon_to_mask(segmentation: list[list[float]], image_height: int = 512, image_width: int = 512) -> Image.Image:
    """
    Convert pre-scaled COCO polygon segmentations into a binary mask image for ControlNet.
    segmentation: list of flat lists of [x1, y1, x2, y2, ...] coordinates natively in 512x512 space.
    """
    mask = Image.new('L', (image_width, image_height), 0)
    for poly in segmentation:
        if not poly or len(poly) < 4:
            continue
        polygon = list(zip(poly[0::2], poly[1::2]))
        ImageDraw.Draw(mask).polygon(polygon, outline=255, fill=255)
    
    # Needs to be 3-channel RGB for diffusers pipelines
    return mask.convert('RGB')


def is_padded(
    image: Image.Image,
    padding_value: int = 127,
    tolerance: int = 2
) -> bool:
    arr = np.array(image.convert("RGB"))
    corners = [
        arr[0, 0],    # top-left
        arr[0, -1],   # top-right
        arr[-1, 0],   # bottom-left
        arr[-1, -1],  # bottom-right
    ]
    return any(
        all(abs(int(channel) - padding_value) < tolerance for channel in corner)
        for corner in corners
    )


def apply_padding_mask(
    synthetic_image: Image.Image,
    source_image: Image.Image,
    padding_value: int = 127,
    tolerance: int = 2
) -> Image.Image:
    source_array = np.array(source_image.convert("RGB"))
    synthetic_array = np.array(synthetic_image.convert("RGB"))
    padding_mask = np.all(
        np.abs(source_array.astype(int) - padding_value) < tolerance,
        axis=-1  # shape: (H, W)
    )
    result = synthetic_array.copy()
    result[padding_mask] = [padding_value, padding_value, padding_value]

    return Image.fromarray(result)


@lru_cache(maxsize=64)
def _load_image_cached(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

@lru_cache(maxsize=64)
def _get_canny_image_cached(path: str) -> Image.Image:
    image_array = np.array(_load_image_cached(path))
    edges = cv2.Canny(image_array, 100, 200)
    return Image.fromarray(np.stack([edges, edges, edges], axis=-1))

@lru_cache(maxsize=64)
def _is_padded_cached(path: str, padding_value: int = 127, tolerance: int = 2) -> bool:
    return is_padded(_load_image_cached(path), padding_value, tolerance)

@lru_cache(maxsize=64)
def _get_padding_mask_cached(path: str, padding_value: int = 127, tolerance: int = 2) -> np.ndarray:
    source_array = np.array(_load_image_cached(path))
    return np.all(
        np.abs(source_array.astype(int) - padding_value) < tolerance,
        axis=-1
    )

def _apply_padding_mask_cached(
    synthetic_image: Image.Image,
    padding_mask: np.ndarray,
    padding_value: int = 127,
) -> Image.Image:
    synthetic_array = np.array(synthetic_image.convert("RGB"))
    result = synthetic_array.copy()
    result[padding_mask] = [padding_value, padding_value, padding_value]
    return Image.fromarray(result)


@dataclass
class EnvConfig:
    device: str
    use_hg_diffusers: bool
    model_id: str
    is_xl: bool
    is_local: bool
    safety_enabled: bool
    ip_repo_id: str
    ip_subfolder: str
    ip_weight_name: str
    ip_scale: float
    control_scale: float


def _build_schedules(args: argparse.Namespace) -> tuple[list[float], list[int], list[float], list[float]]:
    cfg_levels = [args.cfg_scale]
    if args.sweep_cfg:
        if args.no_cfg:
            raise ValueError("--sweep-cfg cannot be combined with --no-cfg.")
        cfg_levels = [float(cfg) for cfg in range(5, 14)]  # 5…13

    step_levels = [args.steps]
    if args.sweep_num_steps:
        step_levels = list(range(10, 51, 5))  # 10,15,…,50

    strength_levels = [args.strength]
    if args.sweep_strength:
        # Typical useful range for img2img strength; keeps steps meaningful
        strength_levels = [round(s, 1) for s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

    control_levels = [args.control_scale]
    if args.sweep_control_scale:
        if not (args.use_canny or args.use_segnet):
            raise ValueError("--sweep-control-scale requires --use-canny or --use-segnet.")
        control_levels = [round(s, 1) for s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]

    sweep_dims = [cfg_levels, step_levels, strength_levels, control_levels]
    sweep_counts = [len(dim) for dim in sweep_dims if len(dim) > 1]

    # No sweeps: just repeat single values
    if not sweep_counts:
        cfg_schedule = cfg_levels * args.num_images
        steps_schedule = step_levels * args.num_images
        strength_schedule = strength_levels * args.num_images
        control_schedule = control_levels * args.num_images
        return cfg_schedule, steps_schedule, strength_schedule, control_schedule

    grid = list(product(cfg_levels, step_levels, strength_levels, control_levels))
    grid_size = len(grid)

    if args.num_images < grid_size:
        sweep_flags = []
        if len(cfg_levels) > 1:
            sweep_flags.append("--sweep-cfg")
        if len(step_levels) > 1:
            sweep_flags.append("--sweep-num-steps")
        if len(strength_levels) > 1:
            sweep_flags.append("--sweep-strength")
        if len(control_levels) > 1:
            sweep_flags.append("--sweep-control-scale")
        joined = " and ".join(sweep_flags)
        raise ValueError(f"{joined} together require at least {grid_size} images to cover all combinations once.")

    repeats = (args.num_images + grid_size - 1) // grid_size
    schedule = (grid * repeats)[: args.num_images]
    cfg_schedule = [c for c, _, _, _ in schedule]
    steps_schedule = [s for _, s, _, _ in schedule]
    strength_schedule = [st for _, _, st, _ in schedule]
    control_schedule = [cs for _, _, _, cs in schedule]

    return cfg_schedule, steps_schedule, strength_schedule, control_schedule


def _resolve_diffusers_scheduler(config, sampler_name: str):
    try:
        from diffusers import (
            DDPMScheduler,
            DDIMScheduler,
            DPMSolverMultistepScheduler,
            DPMSolverSDEScheduler,
            EulerAncestralDiscreteScheduler,
        )
    except ImportError as exc: 
        raise ImportError("The 'diffusers' package is required but not installed.") from exc

    name = sampler_name.lower()
    if name == "ddpm":
        return DDPMScheduler.from_config(config)
    if name == "ddim":
        return DDIMScheduler.from_config(config)
    if name in {"euler-a"}:
        return EulerAncestralDiscreteScheduler.from_config(config)
    if name in {"dpm++"}:
        return DPMSolverMultistepScheduler.from_config(config)
    if name in {"dpm++ sde"}: # only noise outputs for some reason
        return DPMSolverSDEScheduler.from_config(config)
    return None


def _prepare_diffusers_pipelines(
    env_config: EnvConfig,
    sampler_name: str,
    use_canny: bool = False,
    use_segnet: bool = False,
    embeddings: list[str] | None = None,
):
    try:
        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionXLPipeline,
            StableDiffusionXLImg2ImgPipeline,
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            StableDiffusionXLControlNetPipeline,
        )
    except ImportError as exc:
        raise RuntimeError(
            "USE_HG_DIFFUSERS is set but the 'diffusers' package is not installed. "
        ) from exc

    from compel import Compel, ReturnedEmbeddingsType

    torch_dtype = torch.float16 if env_config.device == "cuda" else torch.float32

    pipeline_kwargs = {"torch_dtype": torch_dtype}
    if not env_config.is_xl and not env_config.safety_enabled:
        pipeline_kwargs["safety_checker"] = None
        pipeline_kwargs["feature_extractor"] = None

    controlnet = None

    # controlnet init
    if use_canny or use_segnet:
        if use_segnet:
            controlnet_model_id = (
                "diffusers/controlnet-canny-sdxl-1.0" if env_config.is_xl  # Fallback
                else "lllyasviel/sd-controlnet-seg"
            )
            print(f"[pipeline] loading ControlNet (Segmentation): {controlnet_model_id}")
        else:
            controlnet_model_id = (
                "diffusers/controlnet-canny-sdxl-1.0" if env_config.is_xl
                else "lllyasviel/sd-controlnet-canny"
            )
            print(f"[pipeline] loading ControlNet (Canny): {controlnet_model_id}")
        controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch_dtype)

    load_method = "from_single_file (local)" if env_config.is_local else "from_pretrained"
    print(f"[pipeline] sampler: {sampler_name}")
    print(f"[pipeline] loading {'SDXL' if env_config.is_xl else 'SD 1.x'} model: {env_config.model_id} ({load_method})")

    if (use_canny or use_segnet) and controlnet is not None:
        if env_config.is_local:
            if not Path(env_config.model_id).exists():
                raise FileNotFoundError(f"Local weights file not found: {env_config.model_id}")
            PipelineClass = StableDiffusionXLControlNetPipeline if env_config.is_xl else StableDiffusionControlNetPipeline
            text2img = PipelineClass.from_single_file(env_config.model_id, controlnet=controlnet, **pipeline_kwargs)
        elif env_config.is_xl:
            text2img = StableDiffusionXLControlNetPipeline.from_pretrained(
                env_config.model_id, controlnet=controlnet, **pipeline_kwargs
            )
        else:
            text2img = StableDiffusionControlNetPipeline.from_pretrained(
                env_config.model_id, controlnet=controlnet, **pipeline_kwargs
            )
    else:
        if env_config.is_local:
            if not Path(env_config.model_id).exists():
                raise FileNotFoundError(f"Local weights file not found: {env_config.model_id}")
            PipelineClass = StableDiffusionXLPipeline if env_config.is_xl else StableDiffusionPipeline
            text2img = PipelineClass.from_single_file(env_config.model_id, **pipeline_kwargs)
        elif env_config.is_xl:
            text2img = StableDiffusionXLPipeline.from_pretrained(env_config.model_id, **pipeline_kwargs)
        else:
            text2img = StableDiffusionPipeline.from_pretrained(env_config.model_id, **pipeline_kwargs)

    # Load textual inversion embeddings (if any). We do this after pipeline init so weights are available.
    for emb_path in embeddings or []:
        try:
            text2img.load_textual_inversion(emb_path)
            print(f"[pipeline] loaded textual inversion embedding: {emb_path}")
        except Exception as exc:  # pragma: no cover - defensive load
            print(f"[pipeline] warning: failed to load embedding {emb_path}: {exc}")

    scheduler = _resolve_diffusers_scheduler(text2img.scheduler.config, sampler_name)
    if scheduler is not None:
        text2img.scheduler = scheduler
    text2img = text2img.to(env_config.device)
    try:
        text2img.load_ip_adapter(env_config.ip_repo_id, subfolder=env_config.ip_subfolder, weight_name=env_config.ip_weight_name)
        text2img.set_ip_adapter_scale(env_config.ip_scale)
        print(f"[pipeline] loaded IP-Adapter ({env_config.ip_weight_name}) with scale {env_config.ip_scale}")
    except Exception as e:
        print(f"[pipeline] warning: failed to load IP-Adapter: {e}")

    # text2img.enable_attention_slicing()
    if hasattr(text2img, "vae") and hasattr(text2img.vae, "enable_slicing"):
        text2img.vae.enable_slicing()
    text2img.set_progress_bar_config(disable=True)

    img2img_components = {k: v for k, v in text2img.components.items() if k != "controlnet"}
    if env_config.is_xl:
        img2img = StableDiffusionXLImg2ImgPipeline(**img2img_components)
    else:
        img2img = StableDiffusionImg2ImgPipeline(**img2img_components)

    scheduler_img = _resolve_diffusers_scheduler(img2img.scheduler.config, sampler_name)
    if scheduler_img is not None:
        img2img.scheduler = scheduler_img
    img2img = img2img.to(env_config.device)
    # img2img.enable_attention_slicing()
    if hasattr(img2img, "vae") and hasattr(img2img.vae, "enable_slicing"):
        img2img.vae.enable_slicing()
    img2img.set_progress_bar_config(disable=True)

    # Compel setup differs between SD 1.x (single tokenizer) and SDXL (dual tokenizer)
    if env_config.is_xl:
        # Suppress "passing multiple tokenizers/text encoders is deprecated" –
        # the replacement (CompelForSDXL) isn't available until compel v3.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="passing multiple tokenizers", category=DeprecationWarning)
            compel = Compel(
                tokenizer=[text2img.tokenizer, text2img.tokenizer_2],
                text_encoder=[text2img.text_encoder, text2img.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
            )
    else:
        compel = Compel(
            tokenizer=text2img.tokenizer,
            text_encoder=text2img.text_encoder,
        )

    return text2img, img2img, compel


def _generate_with_diffusers(
    args: argparse.Namespace,
    env_config: EnvConfig,
    prompts: list[str],
    negative_prompts: list[str],
    cfg_schedule: list[float],
    steps_schedule: list[int],
    strength_schedule: list[float],
    control_schedule: list[float],
    seeds: list[int],
    run_outdir: Path,
    timestamp: str,
    input_image: Image.Image | None,
    input_images: list[str] | None = None,
    controlnet_images: list[str] | None = None,
    ip_images: list[str] | None = None,
    segmentations: list[list[list[float]]] | None = None,
    use_canny: bool = False,
    use_segnet: bool = False,
    control_scale: float = 1.0,
    embeddings: list[str] | None = None,
):
    text2img, img2img, compel = _prepare_diffusers_pipelines(
        env_config=env_config,
        sampler_name=args.sampler,
        use_canny=use_canny,
        use_segnet=use_segnet,
        embeddings=embeddings,
    )
    img_paths = input_images if input_images is not None else [""] * len(prompts)
    cn_paths = controlnet_images if controlnet_images is not None else [""] * len(prompts)
    ip_paths = ip_images if ip_images is not None else [""] * len(prompts)
    segs = segmentations if segmentations is not None else [[]] * len(prompts)

    for prompt, neg_prompt, cfg_scale, n_steps, strength, control_scale, seed, img_path, cn_path, ip_path, seg in zip(
        prompts, negative_prompts, cfg_schedule, steps_schedule, strength_schedule, control_schedule, seeds, img_paths, cn_paths, ip_paths, segs
    ):
        active_image = None if args.force_txt2img else (
            _load_image_cached(img_path) if img_path else input_image
        )
        # ControlNet conditioning image (from controlnet_image JSON field)
        cn_image = _load_image_cached(cn_path) if cn_path else None
        
        # IP Adapter reference image
        ip_image = _load_image_cached(ip_path) if ip_path else None

        generator_device = env_config.device if env_config.device in {"cuda", "mps"} else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        guidance_scale = 9.0 if args.no_cfg else cfg_scale

        # SDXL compel returns (embeds, pooled) but  1.x returns just embeds
        if env_config.is_xl:
            prompt_embeds, pooled_prompt = compel(prompt)
            neg_prompt_embeds, neg_pooled_prompt = compel(neg_prompt)
            prompt_embeds, neg_prompt_embeds = compel.pad_conditioning_tensors_to_same_length(
                [prompt_embeds, neg_prompt_embeds]
            )
        else:
            prompt_embeds = compel(prompt)
            neg_prompt_embeds = compel(neg_prompt)
            prompt_embeds, neg_prompt_embeds = compel.pad_conditioning_tensors_to_same_length(
                [prompt_embeds, neg_prompt_embeds]
            )
            pooled_prompt = None
            neg_pooled_prompt = None

        # Build common kwargs; add pooled embeds for SDXL
        common_kwargs = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": neg_prompt_embeds,
            "guidance_scale": guidance_scale,
            "num_inference_steps": n_steps,
            "generator": generator,
        }
        if env_config.is_xl:
            common_kwargs["pooled_prompt_embeds"] = pooled_prompt
            common_kwargs["negative_pooled_prompt_embeds"] = neg_pooled_prompt
            
        if ip_image is not None:
            common_kwargs["ip_adapter_image"] = ip_image

        if use_segnet and seg:
            mask_rgb = polygon_to_mask(seg, image_height=512, image_width=512) # Potentially cacheable but inexpensive
            result = text2img(
                image=mask_rgb,
                controlnet_conditioning_scale=control_scale,
                **common_kwargs,
            )
        elif use_canny and cn_image is not None:
            # Use the dedicated controlnet_image as the canny source
            if cn_path:
                canny_image = _get_canny_image_cached(cn_path)
            else:
                image_array = np.array(cn_image)
                edges = cv2.Canny(image_array, 100, 200)
                canny_image = Image.fromarray(np.stack([edges, edges, edges], axis=-1))
                
            result = text2img(
                image=canny_image,
                controlnet_conditioning_scale=control_scale,
                **common_kwargs,
            )
        elif use_canny and active_image is not None:
            # Fallback: use init_image as canny source (legacy / CLI mode)
            if img_path:
                canny_image = _get_canny_image_cached(img_path)
            else:
                image_array = np.array(active_image)
                edges = cv2.Canny(image_array, 100, 200)
                canny_image = Image.fromarray(np.stack([edges, edges, edges], axis=-1))
                
            result = text2img(
                image=canny_image,
                controlnet_conditioning_scale=control_scale,
                **common_kwargs,
            )
        elif active_image is not None:
            result = img2img(
                image=active_image,
                strength=strength,
                **common_kwargs,
            )
        else:
            result = text2img(**common_kwargs)

        image = result.images[0]
        if active_image is not None:
            if img_path and _is_padded_cached(img_path):
                pad_mask = _get_padding_mask_cached(img_path)
                image = _apply_padding_mask_cached(image, pad_mask)
            elif not img_path and is_padded(active_image):
                image = apply_padding_mask(image, active_image)

        seed_suffix = f"seed{seed}"
        cfg_suffix = f"cfg{cfg_scale:.1f}"
        steps_suffix = f"steps{n_steps}"
        strength_suffix = f"str{strength:.1f}"
        control_suffix = f"cs{control_scale:.1f}"
        outpath = run_outdir / f"{timestamp}_{seed_suffix}_{cfg_suffix}_{steps_suffix}_{strength_suffix}_{control_suffix}.png"
        image.save(outpath)
        print(f"Wrote {outpath}")


def main() -> int:
    model_id = (os.getenv("HF_DIFFUSERS_MODEL_ID", "") or "runwayml/stable-diffusion-v1-5").strip()
    is_xl = "sdxl" in model_id.lower() or "stable-diffusion-xl" in model_id.lower()
    
    try:
        ip_scale = float(os.getenv("HF_IP_ADAPTER_SCALE", "0.6"))
    except ValueError:
        ip_scale = 0.6
        
    try:
        default_control_scale = float(os.getenv("HF_CONTROLNET_SCALE", "1.0"))
    except ValueError:
        default_control_scale = 1.0

    env_config = EnvConfig(
        device=os.getenv("DEVICE", "cpu"),
        use_hg_diffusers=os.getenv("USE_HG_DIFFUSERS", "").strip().lower() == "true",
        model_id=model_id,
        is_xl=is_xl,
        is_local=model_id.endswith((".safetensors", ".ckpt")),
        safety_enabled=os.getenv("ENABLE_SAFETY_CHECKER", "").strip().lower() == "true",
        ip_repo_id=os.getenv("HF_IP_ADAPTER_REPO_ID", "h94/IP-Adapter"),
        ip_subfolder=os.getenv("HF_IP_ADAPTER_SUBFOLDER", "sdxl_models" if is_xl else "models"),
        ip_weight_name=os.getenv("HF_IP_ADAPTER_WEIGHT_NAME", "ip-adapter_sdxl_vit-h.bin" if is_xl else "ip-adapter_sd15.bin"),
        ip_scale=ip_scale,
        control_scale=default_control_scale,
    )
    
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion v1.5.")
    parser.add_argument("--prompt", help="Text prompt (ignored when --from-json is used).", required=False)
    parser.add_argument("--negative-prompt", default="", help="Negative prompt (unconditional prompt).")
    parser.add_argument("--from-json",default=None,help="Path to a prompts JSON file; generates one image per sample in the file.",)
    parser.add_argument("--outdir", default=None, help="Output directory (default: data_generation_outputs).")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate.")
    parser.add_argument("--seed",type=int,default=None,help="Base seed (each image increments by 1). If omitted, uses a random base seed per run.",)
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--sampler",default="euler-a",help="Sampler name: ddpm, ddim, euler-a (ancestral, default), dpm++ sde, or dpmsolver variants.",)
    parser.add_argument("--cfg-scale", type=float, default=9.0, help="Classifier-free guidance scale.")
    parser.add_argument("--sweep-cfg",action="store_true",help="Sweep integer cfg scales 5–13 (inclusive) across images (requires at least 9 images; remainder is distributed unevenly).")
    parser.add_argument("--sweep-num-steps",action="store_true",help="Sweep inference steps 10–50 in increments of 5 across images (requires at least 9 images",)
    parser.add_argument("--sweep-control-scale", action="store_true", help="Sweep ControlNet conditioning scale 0.1–1.0 in 0.1 steps (requires --use-canny or --use-segnet).")
    parser.add_argument("--sweep-strength", action="store_true", help="Sweep img2img strength 0.1–0.9 in 0.1 increments across images.")
    parser.add_argument("--no-cfg", action="store_true", help="Disable classifier-free guidance.")
    parser.add_argument("--strength", type=float, default=0.8, help="img2img strength (0-1). Ignored for txt2img.")
    parser.add_argument("--init-image", default=None, help="Optional path to an input image for img2img.")
    parser.add_argument("--allow-cuda", action="store_true", help="Allow CUDA if available.")
    parser.add_argument("--no-mps", action="store_true", help="Disable Apple MPS even if available.")
    parser.add_argument("--use-canny", action="store_true", help="Use Canny Edge ControlNet for generation.")
    parser.add_argument("--use-segnet", action="store_true", help="Use Segmentation ControlNet for generation.")
    parser.add_argument("--control-scale", type=float, default=env_config.control_scale, help="ControlNet conditioning scale (0.0–1.0).")
    parser.add_argument("--embeddings",nargs="*",default=[],help="Paths to textual inversion embedding folder.",)
    parser.add_argument("--force-txt2img", action="store_true",help="Force text-to-image even when init_image is provided in JSON (ignores all init images).",)

    args = parser.parse_args()

    repo_root = _repo_root()

    if not args.allow_cuda and not ("mps" if not args.no_mps else False):
        pass # Allow device override if explicit arguments restrict it, but generally rely on env_config.device.
    if args.no_mps and env_config.device == "mps":
        env_config.device = "cpu"
    elif args.allow_cuda and torch.cuda.is_available():
        env_config.device = "cuda"

    print(f"Using device: {env_config.device}")

    input_image = Image.open(args.init_image).convert("RGB") if args.init_image else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else (repo_root / "data_generation_outputs")

    init_images: list[str] | None = None
    json_embeddings: list[str] = []

    if args.from_json:
        if args.sweep_cfg or args.sweep_num_steps or args.sweep_strength or args.sweep_control_scale:
            raise ValueError("--from-json cannot be combined with sweep flags.")
        json_path = Path(args.from_json)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        coco_class = data.get("coco_class")
        samples = data.get("samples")
        embedding_path = data.get("embedding_path")
        generation_mode = data.get("generation_mode", "ti")

        if not coco_class or not isinstance(coco_class, str):
            raise ValueError("JSON must include string field 'coco_class'.")
        if not isinstance(samples, list) or not samples:
            raise ValueError("JSON must include non-empty list field 'samples'.")

        if generation_mode == "controlnet":
            print(f"[pipeline] JSON generation_mode='controlnet' — using Canny ControlNet.")
            args.use_canny = True
        elif generation_mode == "ti":
            print(f"[pipeline] JSON generation_mode='ti' — using Textual Inversion.")

        # Optional textual inversion embedding(s) supplied at top-level of JSON.
        # Skip embedding loading for controlnet mode (uses plain words, no TI).
        if embedding_path and generation_mode != "controlnet":
            raw_paths = [embedding_path] if isinstance(embedding_path, str) else embedding_path
            if not isinstance(raw_paths, list) or not all(isinstance(p, str) for p in raw_paths):
                raise ValueError("JSON field 'embedding_path' must be a string or list of strings.")
            for raw_path in raw_paths:
                path_obj = Path(raw_path)
                candidate_paths = [path_obj] if path_obj.is_absolute() else [repo_root / path_obj, json_path.parent / path_obj]
                for candidate in candidate_paths:
                    if candidate.exists():
                        json_embeddings.append(str(candidate.resolve()))
                        break
                else:
                    tried = ", ".join(str(c) for c in candidate_paths)
                    raise FileNotFoundError(f"Embedding path not found for '{raw_path}'. Tried: {tried}")


        prompts: list[str] = []
        negative_prompts: list[str] = []
        cfg_schedule: list[float] = []
        steps_schedule: list[int] = []
        strength_schedule: list[float] = []
        control_schedule: list[float] = []
        init_images: list[str] = []
        controlnet_images: list[str] = []
        ip_images: list[str] = []
        segmentations: list[list[list[float]]] = []

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
            strength_schedule.append(float(sample.get("strength", args.strength)))
            control_schedule.append(float(sample.get("control_scale", args.control_scale)))
            init_images.append(sample.get("init_image", ""))
            controlnet_images.append(sample.get("controlnet_image", ""))
            ip_images.append(sample.get("ip_image", ""))
            segmentations.append(sample.get("segmentation", []))

        num_images = len(prompts)
        outdir = outdir / coco_class
        run_outdir = outdir
        run_outdir.mkdir(parents=True, exist_ok=True)
    else:
        if args.prompt is None:
            raise ValueError("--prompt is required when not using --from-json.")
        num_images = args.num_images
        cfg_schedule, steps_schedule, strength_schedule, control_schedule = _build_schedules(args)
        prompts = [args.prompt] * num_images
        negative_prompts = [args.negative_prompt] * num_images
        run_outdir = (outdir / timestamp) if num_images > 1 else outdir
        run_outdir.mkdir(parents=True, exist_ok=True)

    seed_base = args.seed
    if seed_base is None:
        seed_base = int(torch.randint(0, 2**31 - 1, (1,)).item())
    seeds = [seed_base + i for i in range(num_images)]



    # Combine embeddings from CLI and JSON (JSON first so run-specific embeddings load before global ones)
    all_embeddings = json_embeddings + [str(p) for p in args.embeddings]

    if env_config.use_hg_diffusers:
        print(f"USE_HG_DIFFUSERS=true -> using Hugging Face diffusers backend ({env_config.model_id}).")
        _generate_with_diffusers(
            args=args,
            env_config=env_config,
            prompts=prompts,
            negative_prompts=negative_prompts,
            cfg_schedule=cfg_schedule,
            steps_schedule=steps_schedule,
            strength_schedule=strength_schedule,
            control_schedule=control_schedule,
            seeds=seeds,
            run_outdir=run_outdir,
            timestamp=timestamp,
            input_image=input_image,
            input_images=init_images,
            controlnet_images=controlnet_images if args.from_json else None,
            ip_images=ip_images if args.from_json else None,
            segmentations=segmentations if args.from_json else None,
            use_canny=args.use_canny,
            use_segnet=args.use_segnet,
            control_scale=args.control_scale,
            embeddings=all_embeddings,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
