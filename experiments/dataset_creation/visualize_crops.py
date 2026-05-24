import os
import glob
import math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from PIL import Image
from pycocotools.coco import COCO

# ==============================================================================
# CONFIGURATION
# Set the target full-frame COCO image filename and the dataset split here.
# The script will automatically find the corresponding contextual crops.
# ==============================================================================
TARGET_IMAGE_NAME = "000000000532.jpg"  # Example: "000000315200.jpg" (has 12 crops)
SPLIT = "train"                         # e.g., "train", "val", "test"

# Paths based on preprocess_coco.py and the project structure
COCO_IMAGES_DIR = Path("coco_dataset/split/images")
CROPS_DIR = Path("coco_dataset/contextual_crops/images")
ORIGINAL_ANN_PATH = Path(f"coco_dataset/split/annotations/instances_{SPLIT}.json")
CROPS_ANN_PATH = Path(f"coco_dataset/contextual_crops/annotations/single_instances_{SPLIT}.json")
OUTPUT_FIG_PATH = Path("experiments/figures/visualize_crops.png")

def main():
    # 1. Parse the image ID
    base_name = TARGET_IMAGE_NAME.split(".")[0]
    try:
        img_id = int(base_name)
    except ValueError:
        print(f"Error: Could not parse integer image ID from {TARGET_IMAGE_NAME}")
        return
    
    # 2. Locate the full-frame image
    fullframe_path = COCO_IMAGES_DIR / SPLIT / TARGET_IMAGE_NAME
    if not fullframe_path.exists():
        print(f"Error: Full-frame image not found at {fullframe_path}")
        print("Please ensure TARGET_IMAGE_NAME and SPLIT are correct.")
        return
        
    # 3. Locate the associated contextual crops
    crops_split_dir = CROPS_DIR / SPLIT
    if not crops_split_dir.exists():
        print(f"Error: Crops directory not found at {crops_split_dir}")
        return
        
    crop_pattern = f"{img_id}_*.jpg"
    crop_paths = list(crops_split_dir.glob(crop_pattern))
    
    if not crop_paths:
        print(f"No contextual crops found for image ID {img_id} in {crops_split_dir}")
        return
        
    print(f"Found {len(crop_paths)} contextual crops for {TARGET_IMAGE_NAME}")
    
    # Load COCO annotations
    print("Loading original COCO annotations...")
    coco_orig = COCO(str(ORIGINAL_ANN_PATH))
    print("Loading contextual crops COCO annotations...")
    coco_crops = COCO(str(CROPS_ANN_PATH))
    
    # 4. Load the images
    full_img = Image.open(fullframe_path).convert("RGB")
    
    # Parse out crops data
    crops_data = []
    for p in crop_paths:
        crop_img = Image.open(p).convert("RGB")
        crops_data.append((crop_img, p))
    
    # 5. Create the visualization
    num_crops = len(crops_data)
    
    # Layout crops in a single row
    ncols = num_crops
    nrows = 1
    
    # Dynamically adjust figure size for a thin, long plot
    # The main image gets a width ratio of 1.5, space is 0.3, and crops are 1 each.
    fig_width = 3.5 * (1.5 + 0.3 + ncols) 
    fig_height = 5  # Fixed small height for a single row
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create gridspec: 
    # Col 0: main image
    # Col 1: empty space for the arrow
    # Col 2+: crop images
    width_ratios = [1.5, 0.3] + [1] * ncols
    height_ratios = [0.2] + [1] * nrows  # Extra row at the top for the title
    gs = fig.add_gridspec(nrows + 1, 2 + ncols, width_ratios=width_ratios, height_ratios=height_ratios)
    
    # ---------------------------------------------------------
    # Plot the original full-frame image on the left
    # ---------------------------------------------------------
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(full_img)
    ax_main.set_title("Original COCO Image", fontsize=16, fontweight='bold', pad=15)
    ax_main.axis("off")
    
    # Draw original bounding boxes in red
    orig_ann_ids = coco_orig.getAnnIds(imgIds=[img_id])
    orig_anns = coco_orig.loadAnns(orig_ann_ids)
    
    for ann in orig_anns:
        if ann.get('iscrowd', 0) == 1:
            continue
            
        cat = coco_orig.loadCats([ann['category_id']])[0]
        label_name = cat['name']
        x, y, w, h = ann['bbox']
        
        # Draw red bounding box (with transparency)
        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='red', facecolor='none', alpha=0.5)
        ax_main.add_patch(rect)
    
    # ---------------------------------------------------------
    # Draw the grey shaded squircle background for crops
    # ---------------------------------------------------------
    ax_bg = fig.add_subplot(gs[:, 2:])
    ax_bg.axis("off")
    
    # FancyBboxPatch creates the rounded rectangle (squircle)
    bg_rect = patches.FancyBboxPatch(
        (0, 0), 1, 1, 
        boxstyle="round,pad=0.05,rounding_size=0.05",
        transform=ax_bg.transAxes,
        edgecolor="#b0b0b0", facecolor="#f0f0f0", 
        linewidth=2,
        zorder=0,
        clip_on=False
    )
    ax_bg.add_patch(bg_rect)
    # Place title inside the grey box, near the top
    ax_bg.text(0.5, 0.96, "Contextual crops", transform=ax_bg.transAxes, 
               ha="center", va="top", fontsize=18, fontweight='bold', color="#333333")
    
    # ---------------------------------------------------------
    # Draw the single connecting arrow
    # ---------------------------------------------------------
    # Arrow from the right-middle of main image to the left-middle of the shaded area
    con = ConnectionPatch(
        xyA=(1.02, 0.5), coordsA=ax_main.transAxes,
        xyB=(-0.05, 0.5), coordsB=ax_bg.transAxes,
        arrowstyle="-|>", shrinkA=0, shrinkB=0,
        mutation_scale=25, color="black", linewidth=3,
        zorder=10 
    )
    fig.add_artist(con)
    
    # ---------------------------------------------------------
    # Plot crops inside the grid
    # ---------------------------------------------------------
    for i, (crop_img, crop_path) in enumerate(crops_data):
        row = i // ncols
        col = i % ncols
        
        # Subplot inside the background area (offset row by 1 for the title)
        ax_crop = fig.add_subplot(gs[row + 1, 2 + col])
        ax_crop.imshow(crop_img)
        ax_crop.axis("off")
        ax_crop.set_zorder(5) # Ensure it appears above the background
        
        # Look up this crop in the single_instances JSON to get the green bounding box
        crop_img_infos = [img for img in coco_crops.dataset['images'] if img['file_name'] == crop_path.name]
        if crop_img_infos:
            crop_img_info = crop_img_infos[0]
            crop_ann_ids = coco_crops.getAnnIds(imgIds=[crop_img_info['id']])
            
            if crop_ann_ids:
                crop_ann = coco_crops.loadAnns(crop_ann_ids)[0]
                x, y, w, h = crop_ann['bbox']
                cat = coco_crops.loadCats([crop_ann['category_id']])[0]
                label_name = cat['name']
                
                # Draw green bounding box (with transparency)
                rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='#00aa00', facecolor='none', alpha=0.5)
                ax_crop.add_patch(rect)
                
                # Add bold label
                ax_crop.text(
                    x, max(0, y - 5), label_name, 
                    color='white', fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='#00aa00', alpha=0.9, edgecolor='none', pad=2)
                )
        
    # Adjust layout to make sure nothing is cut off
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    # 6. Save the figure
    OUTPUT_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FIG_PATH, dpi=300, bbox_inches="tight")
    print(f"Success! Visualization saved to {OUTPUT_FIG_PATH}")

if __name__ == "__main__":
    main()
