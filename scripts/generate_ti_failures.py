#!/usr/bin/env python3
import json
import os
import subprocess
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[1]
    prompt_dir = repo_root / "data_generation_backend" / "prompt_generation"
    out_dir = repo_root / "data_generation_outputs" / "ti_failures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Truncate toaster TI prompts to first 10
    # toaster_in = prompt_dir / "toaster_ti_prompts.json"
    # toaster_out = prompt_dir / "toaster_ti_prompts_first10.json"
    
    # print(f"Loading {toaster_in}...")
    # with open(toaster_in, "r", encoding="utf-8") as f:
    #     toaster_data = json.load(f)
    
    # toaster_first10 = {
    #     "coco_class": toaster_data["coco_class"],
    #     "generation_mode": toaster_data["generation_mode"],
    #     "embedding_path": toaster_data["embedding_path"],
    #     "samples": toaster_data["samples"][:10]
    # }
    
    # print(f"Writing first 10 toaster samples to {toaster_out}...")
    # with open(toaster_out, "w", encoding="utf-8") as f:
    #     json.dump(toaster_first10, f, indent=4, ensure_ascii=False)

    # 2. Truncate hair drier TI prompts to first 10
    dryer_in = prompt_dir / "hair_drier_ti_prompts.json"
    dryer_out = prompt_dir / "hair_drier_ti_prompts_first10.json"
    
    print(f"Loading {dryer_in}...")
    with open(dryer_in, "r", encoding="utf-8") as f:
        dryer_data = json.load(f)
        
    dryer_first10 = {
        "coco_class": dryer_data["coco_class"],
        "generation_mode": dryer_data["generation_mode"],
        "embedding_path": dryer_data["embedding_path"],
        "samples": dryer_data["samples"][:10]
    }
    
    print(f"Writing first 10 hair drier samples to {dryer_out}...")
    with open(dryer_out, "w", encoding="utf-8") as f:
        json.dump(dryer_first10, f, indent=4, ensure_ascii=False)

    # 3. Call diffusion runner for both JSON files
    runner_script = repo_root / "data_generation_backend" / "diffusion_runner.py"
    
    # Run toaster
    # print("\n--- Generating Toaster TI Failure Samples ---")
    # cmd_toaster = [
    #     "python", str(runner_script),
    #     "--from-json", str(toaster_out),
    #     "--outdir", str(out_dir)
    # ]
    # print(f"Executing: {' '.join(cmd_toaster)}")
    # subprocess.run(cmd_toaster, check=True)

    # Run hair drier
    print("\n--- Generating Hair Drier TI Failure Samples ---")
    cmd_dryer = [
        "python", str(runner_script),
        "--from-json", str(dryer_out),
        "--outdir", str(out_dir)
    ]
    print(f"Executing: {' '.join(cmd_dryer)}")
    subprocess.run(cmd_dryer, check=True)

    print("\nGeneration completed successfully! Images are located in data_generation_outputs/ti_failures/")

if __name__ == "__main__":
    main()
