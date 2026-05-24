#!/usr/bin/env python3
import os
import glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def main():
    # Resolve paths relative to the script location
    repo_root = Path(__file__).resolve().parents[1]
    ti_failures_dir = repo_root / "data_generation_outputs" / "ti_failures" / "hair_drier_10_ti"
    out_dir = repo_root / "data_generation_outputs" / "ti_failures"
    
    if not ti_failures_dir.exists():
        print(f"Error: Directory not found: {ti_failures_dir}")
        return 1
        
    # Find all PNG files in the folder
    png_files = sorted(list(ti_failures_dir.glob("*.png")))
    
    if len(png_files) < 10:
        print(f"Error: Found only {len(png_files)} PNG files in {ti_failures_dir}. Need at least 10.")
        return 1
        
    # Take the first 10 files
    selected_files = png_files[:10]
    print("Combining the following 10 images:")
    for f in selected_files:
        print(f" - {f.name}")
        
    # Load all images and check sizes
    images = [Image.open(f).convert("RGB") for f in selected_files]
    img_width, img_height = images[0].size
    
    # Grid size: 1x10 row
    total_width = img_width * 10
    banner_height = 160
    total_height = img_height + banner_height
    
    # Create the combined canvas
    combined_img = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    
    # Paste images side-by-side with zero padding, offset by the banner height
    for idx, img in enumerate(images):
        combined_img.paste(img, (idx * img_width, banner_height))
        
    # Draw the title banner at the top
    draw = ImageDraw.Draw(combined_img)
    caption_text = 'Textual Inversion Generations for "hair drier" Class'
    
    # Attempt to load a premium standard macOS bold font
    font = None
    font_paths = [
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
        ("/Library/Fonts/Arial Bold.ttf", 0),
        ("/System/Library/Fonts/Helvetica.ttc", 1),  # Index 1 is typically Bold
        ("/System/Library/Fonts/Helvetica.ttc", 0),
        ("/System/Library/Fonts/Supplemental/Arial.ttf", 0)
    ]
    
    for path, index in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size=80, index=index)
                break
            except IOError:
                continue
                
    if font is None:
        # Fallback to default
        font = ImageFont.load_default()
        print("Warning: Standard bold fonts not found. Using PIL default font.")
        
    # Calculate text bounding box to center it
    try:
        # For newer Pillow versions
        left, top, right, bottom = draw.textbbox((0, 0), caption_text, font=font)
        text_width = right - left
        text_height = bottom - top
    except AttributeError:
        # Fallback for older Pillow versions
        text_width, text_height = draw.textsize(caption_text, font=font)
        
    text_x = (total_width - text_width) // 2
    text_y = (banner_height - text_height) // 2 - 5
    
    # Draw text in clean charcoal/black color
    draw.text((text_x, text_y), caption_text, fill=(30, 30, 30), font=font)
    
    # Save the output file
    output_path = out_dir / "ti_failures_grid.png"
    combined_img.save(output_path, "PNG")
    print(f"\nSuccessfully created and saved the combined grid to:\n{output_path}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
