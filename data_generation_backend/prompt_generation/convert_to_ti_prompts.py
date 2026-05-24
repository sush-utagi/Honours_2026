#!/usr/bin/env python3
"""Convert an existing IP-Adapter or ControlNet prompt JSON into a
Textual Inversion prompt JSON.

Changes made:
  1. generation_mode -> "ti"
  2. Adds embedding_path (top-level) pointing to the learned_embeds.safetensors
  3. Replaces (class_name)1.2 with (<S*>)1.2 in every prompt
  4. Strips ip_image / controlnet_image from each sample (optional, kept for traceability)

Usage:
    python convert_to_ti_prompts.py \
        --input  toaster_ip_adapter_prompts.json \
        --output toaster_ti_prompts.json \
        --embedding-path ../embeddings/toaster/learned_embeds.safetensors \
        --placeholder "<S*>"

    python convert_to_ti_prompts.py \
        --input  hair_drier_ip_adapter_prompts.json \
        --output hair_drier_ti_prompts.json \
        --embedding-path ../embeddings/dryer/learned_embeds.safetensors \
        --placeholder "<S*>"
"""
import argparse
import json
import re
from pathlib import Path


def convert(
    input_path: Path,
    output_path: Path,
    embedding_path: str,
    placeholder: str = "<S*>",
    strip_images: bool = True,
) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    class_name = data["coco_class"]
    print(f"[convert] class: {class_name}")
    print(f"[convert] replacing '({class_name})1.2' -> '({placeholder})1.2' in all prompts")

    # Update top-level fields
    data["generation_mode"] = "ti"
    data["embedding_path"] = embedding_path

    # Pattern: (class_name)1.2  — handles "toaster" or "hair drier"
    # The Compel syntax uses literal parentheses
    escaped_class = re.escape(class_name)
    pattern = re.compile(rf"\({escaped_class}\)(\d+\.?\d*)")

    converted = 0
    for sample in data.get("samples", []):
        old_prompt = sample["prompt"]
        new_prompt = pattern.sub(rf"({placeholder})\1", old_prompt)

        if new_prompt != old_prompt:
            converted += 1
        sample["prompt"] = new_prompt

        # Optionally strip conditioning image fields (they're meaningless for TI)
        if strip_images:
            sample.pop("ip_image", None)
            sample.pop("controlnet_image", None)

    print(f"[convert] converted {converted}/{len(data.get('samples', []))} prompts")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"[convert] saved -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IP-Adapter/ControlNet prompt JSON to TI format.")
    parser.add_argument("--input", required=True, help="Path to source JSON (e.g. toaster_ip_adapter_prompts.json)")
    parser.add_argument("--output", required=True, help="Path to output TI JSON")
    parser.add_argument("--embedding-path", required=True, help="Relative or absolute path to learned_embeds.safetensors")
    parser.add_argument("--placeholder", default="<S*>", help="Placeholder token used during TI training (default: <S*>)")
    parser.add_argument("--keep-images", action="store_true", help="Keep ip_image/controlnet_image fields in output")
    args = parser.parse_args()

    convert(
        input_path=Path(args.input),
        output_path=Path(args.output),
        embedding_path=args.embedding_path,
        placeholder=args.placeholder,
        strip_images=not args.keep_images,
    )
