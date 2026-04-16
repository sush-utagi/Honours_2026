
from __future__ import annotations

import math
import os
import warnings

from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

_MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-large-patch14")

_model: CLIPModel | None = None
_processor: CLIPProcessor | None = None
_device: str | None = None


def _ensure_model() -> tuple[CLIPModel, CLIPProcessor, str]:
    global _model, _processor, _device
    if _model is not None:
        assert _processor is not None and _device is not None
        return _model, _processor, _device

    _device = os.getenv("DEVICE", "cpu")
    print(f"[clip-grader] loading {_MODEL_ID} on {_device} …")
    _processor = CLIPProcessor.from_pretrained(_MODEL_ID, use_fast=True)
    _model = CLIPModel.from_pretrained(_MODEL_ID).to(_device).eval()
    print("[clip-grader] model ready.")
    return _model, _processor, _device


def _load_image(path: Path) -> Image.Image | None:
    try:
        img = Image.open(path).convert("RGB")
        img.load()
        return img
    except Exception as exc:
        warnings.warn(f"[clip-grader] skipping unreadable image {path}: {exc}")
        return None


@torch.no_grad()
def compute_scores(
    image_paths: Sequence[Union[str, Path]],
    target_text: str,
    batch_size: int = 32,
) -> List[float]:
    model, processor, device = _ensure_model()
    paths = [Path(p) for p in image_paths]
    scores: list[float | None] = [None] * len(paths)

    text_inputs = processor(text=[target_text], return_tensors="pt", padding=True).to(device)
    text_embeds = model.get_text_features(**text_inputs)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    n_batches = math.ceil(len(paths) / batch_size)
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, len(paths))
        batch_paths = paths[start:end]

        images: list[Image.Image] = []
        valid_indices: list[int] = []

        for i, p in enumerate(batch_paths, start=start):
            img = _load_image(p)
            if img is None:
                scores[i] = float("nan")
            else:
                images.append(img)
                valid_indices.append(i)

        if not images:
            continue

        pixel_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        image_embeds = model.get_image_features(**pixel_inputs)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        # CLIP score: 100 * max(cosine_similarity, 0)
        sims = (image_embeds @ text_embeds.T).squeeze(-1).clamp(min=0).mul(100).cpu().tolist()
        if isinstance(sims, float):
            sims = [sims]

        for idx, sim in zip(valid_indices, sims):
            scores[idx] = sim

        if (b + 1) % 10 == 0 or (b + 1) == n_batches:
            print(f"[clip-grader] batch {b + 1}/{n_batches} done")

    return [s if s is not None else float("nan") for s in scores]


@torch.no_grad()
def compute_scores_per_prompt(
    image_paths: Sequence[Union[str, Path]],
    prompts: Sequence[str],
    batch_size: int = 32,
) -> List[float]:
    if len(image_paths) != len(prompts):
        raise ValueError(
            f"image_paths ({len(image_paths)}) and prompts ({len(prompts)}) must have the same length"
        )

    model, processor, device = _ensure_model()
    paths = [Path(p) for p in image_paths]
    scores: list[float | None] = [None] * len(paths)

    n_batches = math.ceil(len(paths) / batch_size)
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, len(paths))
        batch_paths = paths[start:end]
        batch_prompts_raw = list(prompts[start:end])

        images: list[Image.Image] = []
        batch_texts: list[str] = []
        valid_indices: list[int] = []

        for i, (p, prompt) in enumerate(zip(batch_paths, batch_prompts_raw), start=start):
            img = _load_image(p)
            if img is None:
                scores[i] = float("nan")
            else:
                images.append(img)
                batch_texts.append(prompt)
                valid_indices.append(i)

        if not images:
            continue

        pixel_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        text_inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        image_embeds = model.get_image_features(**pixel_inputs)
        text_embeds = model.get_text_features(**text_inputs)

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # CLIP score: 100 * max(cosine_similarity, 0)
        sims = (image_embeds * text_embeds).sum(dim=-1).clamp(min=0).mul(100).cpu().tolist()
        if isinstance(sims, float):
            sims = [sims]

        for idx, sim in zip(valid_indices, sims):
            scores[idx] = sim

        if (b + 1) % 10 == 0 or (b + 1) == n_batches:
            print(f"[clip-grader] (per-prompt) batch {b + 1}/{n_batches} done")

    return [s if s is not None else float("nan") for s in scores]


@torch.no_grad()
def extract_clip_embeddings(
    image_paths: Sequence[Union[str, Path]],
    batch_size: int = 32,
) -> tuple[np.ndarray, list[int]]:
    """Extract L2-normalised CLIP image embeddings for downstream projection.
    Returns raw embedding vectors rather than scalar similarity scores, making them suitable for
    dimensionality reduction techniques such as UMAP.
    """
    model, processor, device = _ensure_model()
    paths = [Path(p) for p in image_paths]

    all_embeds: list[np.ndarray] = []
    valid_indices: list[int] = []

    n_batches = math.ceil(len(paths) / batch_size)
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, len(paths))
        batch_paths = paths[start:end]

        images: list[Image.Image] = []
        batch_valid: list[int] = []

        for i, p in enumerate(batch_paths, start=start):
            img = _load_image(p)
            if img is not None:
                images.append(img)
                batch_valid.append(i)

        if not images:
            continue

        pixel_inputs = processor(
            images=images, return_tensors="pt", padding=True,
        ).to(device)
        embeds = model.get_image_features(**pixel_inputs)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)

        all_embeds.append(embeds.cpu().numpy())
        valid_indices.extend(batch_valid)

        if (b + 1) % 10 == 0 or (b + 1) == n_batches:
            print(f"[clip-grader] embedding batch {b + 1}/{n_batches} done")

    if not all_embeds:
        return np.empty((0, 0), dtype=np.float32), []

    return np.concatenate(all_embeds, axis=0), valid_indices


@torch.no_grad()
def extract_clip_text_embeddings(
    texts: List[str],
) -> np.ndarray: # Shape ``(N, D)`` L2-normalised CLIP text embeddings.
    model, processor, device = _ensure_model()
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    embeds = model.get_text_features(**inputs)
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds.cpu().numpy()
