import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch

def main():
    base_dir = "/Users/susheelutagi/Documents/GitHub/Honours_2026"
    real_img_path = os.path.join(base_dir, "london_zebra.JPG")
    
    # We will put CN on top and IP on bottom, or vice-versa. 
    # Let's say IP on top, CN on bottom.
    ip_dir = os.path.join(base_dir, "data_generation_outputs/ip_cn_examples/ip")
    cn_dir = os.path.join(base_dir, "data_generation_outputs/ip_cn_examples/cn")

    ip_images = sorted(glob.glob(os.path.join(ip_dir, "*.png")))[:5]
    cn_images = sorted(glob.glob(os.path.join(cn_dir, "*.png")))[:5]

    if not os.path.exists(real_img_path):
        print(f"Error: Could not find {real_img_path}")
        return

    fig = plt.figure(figsize=(20, 6))
    
    # Concatenate images horizontally to ensure exactly 0 padding
    row_ip_img = np.concatenate([mpimg.imread(p) for p in ip_images], axis=1)
    row_cn_img = np.concatenate([mpimg.imread(p) for p in cn_images], axis=1)

    # 2 rows, 3 columns: 1 for real, 1 empty for arrows, 1 for the concatenated block
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.5, 0.5, 5], wspace=0.0, hspace=0.5)

    # Real image
    ax_real = fig.add_subplot(gs[:, 0])
    img_real = mpimg.imread(real_img_path)
    ax_real.imshow(img_real)
    ax_real.axis('off')
    ax_real.set_title("Real Image", fontsize=16, pad=15)

    # IP Images (Top Row)
    ax_ip = fig.add_subplot(gs[0, 2])
    ax_ip.imshow(row_ip_img)
    ax_ip.axis('off')
    ax_ip.set_title("IP-Adapter", fontsize=16, pad=15, loc='left')

    # CN Images (Bottom Row)
    ax_cn = fig.add_subplot(gs[1, 2])
    ax_cn.imshow(row_cn_img)
    ax_cn.axis('off')
    ax_cn.set_title("ControlNet", fontsize=16, pad=15, loc='left')

    # Add arrows
    arrow1 = ConnectionPatch(xyA=(1.0, 0.75), xyB=(0.0, 0.5), 
                             coordsA='axes fraction', coordsB='axes fraction',
                             axesA=ax_real, axesB=ax_ip,
                             arrowstyle="-|>", mutation_scale=25, color="black", linewidth=2.5)
    ax_real.add_artist(arrow1)

    arrow2 = ConnectionPatch(xyA=(1.0, 0.25), xyB=(0.0, 0.5), 
                             coordsA='axes fraction', coordsB='axes fraction',
                             axesA=ax_real, axesB=ax_cn,
                             arrowstyle="-|>", mutation_scale=25, color="black", linewidth=2.5)
    ax_real.add_artist(arrow2)

    # Removed plt.tight_layout() to avoid overriding wspace=0.0
    out_path = os.path.join(base_dir, "experiments", "figures", "ip_cn_visual.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved visual to {out_path}")

if __name__ == "__main__":
    main()
