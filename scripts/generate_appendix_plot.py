import os
import random
from PIL import Image
import matplotlib.pyplot as plt

def create_composite(image_paths, grid_shape=(5, 10), img_size=(256, 256)):
    rows, cols = grid_shape
    width, height = img_size
    composite = Image.new('RGB', (cols * width, rows * height))
    for i, path in enumerate(image_paths):
        row = i // cols
        col = i % cols
        img = Image.open(path).convert('RGB').resize((width, height))
        composite.paste(img, (col * width, row * height))
    return composite

def main():
    # Ensure random selection every time the script is run
    # random.seed(42)  # Removed to allow new random images each run
    
    # Path setup based on the repository structure
    base_dir = "data_generation_outputs"
    
    techniques = {
        "ControlNet": {
            "filename": "appendix_plot_cn.png",
            "classes": [
                {
                    "class_name": "Hair Drier",
                    "folder": os.path.join(base_dir, "diffusion_based_augmentation_cn", "hair_drier")
                },
                {
                    "class_name": "Toaster",
                    "folder": os.path.join(base_dir, "diffusion_based_augmentation_cn", "toaster")
                }
            ]
        },
        "IP-Adapter": {
            "filename": "appendix_plot_ip.png",
            "classes": [
                {
                    "class_name": "Hair Drier",
                    "folder": os.path.join(base_dir, "diffusion_based_augmentations_ip", "hair drier")
                },
                {
                    "class_name": "Toaster",
                    "folder": os.path.join(base_dir, "diffusion_based_augmentations_ip", "toaster")
                }
            ]
        }
    }
    
    os.makedirs("experiments/figures", exist_ok=True)
    
    for technique_name, data in techniques.items():
        # 2 rows, 1 column. 15 inches wide, 10 inches tall.
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, class_data in enumerate(data["classes"]):
            folder = class_data["folder"]
            class_name = class_data["class_name"]
            
            if not os.path.exists(folder):
                print(f"Warning: {folder} does not exist.")
                continue
                
            all_images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(all_images) == 0:
                print(f"Warning: No images found in {folder}.")
                continue
                
            random.shuffle(all_images)
            valid_images = []
            
            for path in all_images:
                try:
                    with Image.open(path) as img:
                        # Check if max pixel value is > 0 (not completely black)
                        if img.convert("L").getextrema()[1] > 0:
                            valid_images.append(path)
                except Exception:
                    pass
                
                if len(valid_images) == 40:
                    break
                    
            if len(valid_images) < 40:
                print(f"Warning: Not enough valid (non-black) images in {folder} (found {len(valid_images)}). Padding with duplicates.")
                if len(valid_images) > 0:
                    selected_images = valid_images + random.choices(valid_images, k=40 - len(valid_images))
                else:
                    print(f"Error: No valid images found in {folder}!")
                    continue
            else:
                selected_images = valid_images
                
            # Create a 4x10 grid with no spacing in between images
            composite = create_composite(selected_images, grid_shape=(4, 10), img_size=(256, 256))
            
            ax = axes[idx]
            ax.imshow(composite)
            ax.axis('off')
            ax.set_title(f"{class_name} - {technique_name}", fontsize=16, pad=15)
            
        plt.tight_layout(pad=3.0)
        output_path = os.path.join("experiments/figures", data["filename"])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot successfully generated and saved to {output_path}")

if __name__ == "__main__":
    main()
