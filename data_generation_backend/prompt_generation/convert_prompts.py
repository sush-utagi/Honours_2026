#!/usr/bin/env python3
"""Convert textual inversion tokens in prompt JSONs to plain words.

Reads a prompt JSON file, replaces all TI placeholder tokens with their
plain-English equivalents, and writes the result to a new file.

Usage::

    python convert_prompts.py toaster_ti_prompts.json
    python convert_prompts.py hair_drier_ti_prompts.json

Produces ``toaster_ti_prompts_plain.json``, etc.
"""

import argparse
import json
from pathlib import Path

# Mapping of TI placeholder tokens → plain words.
TOKEN_MAP: dict[str, str] = {
    "<coco-toaster>": "toaster",
    "<coco-dryer>": "hair dryer",
}


def convert(input_path: Path) -> Path:
    with input_path.open("r", encoding="utf-8") as f:
        raw = f.read()

    for token, replacement in TOKEN_MAP.items():
        raw = raw.replace(token, replacement)

    out_path = input_path.with_name(input_path.stem + "_plain" + input_path.suffix)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(raw)

    # Quick sanity check: count samples
    data = json.loads(raw)
    n = len(data.get("samples", []))
    print(f"[done] {input_path.name} → {out_path.name}  ({n} samples converted)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace TI tokens with plain words in prompt JSONs.")
    parser.add_argument("files", nargs="+", help="One or more prompt JSON files to convert.")
    args = parser.parse_args()

    for f in args.files:
        convert(Path(f))
