from ddpm import DDPMSampler
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. Setup
generator = torch.Generator()
generator.manual_seed(0)
ddpm_sampler = DDPMSampler(generator)

# 2. Load and prep image
img_path = "/Users/susheelutagi/Documents/GitHub/Honours_2026/coco_dataset/contextual_crops/images/train/50948_590326.jpg"
img = Image.open(img_path)
img_tensor = torch.tensor(np.array(img)).float()
img_tensor = ((img_tensor / 255.0) * 2.0) - 1.0
img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension: [1, H, W, C]

# 3. Apply noise
# text2img: pure noise (timestep 999)
t_text = torch.tensor([999])
noisy_text = ddpm_sampler.add_noise(img_tensor, t_text)

# img2img: partial noise (strength 0.4 -> timestep 400)
t_img = torch.tensor([600])
noisy_img_2 = ddpm_sampler.add_noise(img_tensor, t_img)

def to_pil(tensor):
    tensor = (tensor.clamp(-1, 1) + 1) / 2
    tensor = (tensor * 255).type(torch.uint8)
    return Image.fromarray(tensor.squeeze(0).numpy(), 'RGB')

# 4. Create 2x1 Figure (Transposed)
fig, axes = plt.subplots(2, 1, figsize=(6, 12))

axes[0].imshow(to_pil(noisy_text))
axes[0].set_title("text2img starting point", fontsize=24, pad=15)
axes[0].axis('off')

axes[1].imshow(to_pil(noisy_img_2))
axes[1].set_title("img2img starting point (strength 0.6)", fontsize=24, pad=15)
axes[1].axis('off')

plt.tight_layout(pad=3.0)
out_path = "experiments/figures/diffusion_starting_points.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved transposed comparison figure to {out_path}")