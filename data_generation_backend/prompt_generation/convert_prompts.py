#!/usr/bin/env python3


"""converts from textual inversion prompts to controlnet prompts
"""

import argparse
import json
from pathlib import Path

TOKEN_MAP: dict[str, str] = {
    "<coco-toaster>": "toaster",
    "<coco-dryer>": "hair dryer",
}


def convert(input_path: Path) -> Path:
    with input_path.open("r", encoding="utf-8") as f:
        raw = f.read()

    for token, replacement in TOKEN_MAP.items():
        raw = raw.replace(token, replacement)

    data = json.loads(raw)
    data["generation_mode"] = "controlnet"

    out_path = input_path.with_name(input_path.stem + "_controlnet" + input_path.suffix)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    n = len(data.get("samples", []))
    print(f"[done] {input_path.name} → {out_path.name}  ({n} samples converted)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace TI tokens with plain words in prompt JSONs.")
    parser.add_argument("files", nargs="+", help="One or more prompt JSON files to convert.")
    args = parser.parse_args()

    for f in args.files:
        convert(Path(f))
