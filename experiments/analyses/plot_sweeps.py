import os
import glob
import re
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

def extract_value(fname, technique):
    param_key = "ips" if technique == "ip" else "cs"
    match = re.search(rf"_{param_key}([\d]+(?:\.[\d]+)?)", fname)
    if match:
        return float(match.group(1))
    return 0.0

def create_figure_for_sweep(sweep_dir, save_path, technique):
    param_key = "ips" if technique == "ip" else "cs"
    input_images = glob.glob(os.path.join(sweep_dir, "input.*"))
    input_image_path = input_images[0] if input_images else None
    
    gen_images = [fp for fp in os.listdir(sweep_dir) if f"_{param_key}" in fp and fp.lower().endswith((".png", ".jpg", ".jpeg"))]
    gen_images.sort(key=lambda x: extract_value(x, technique))
    
    if not gen_images:
        print(f"Skipping {sweep_dir}, no generated images found.")
        return
        
    # Load all images
    imgs = []
    labels = []
    
    if input_image_path:
        imgs.append(mpimg.imread(input_image_path))
        labels.append("Input")
        
    for fname in gen_images:
        imgs.append(mpimg.imread(os.path.join(sweep_dir, fname)))
        labels.append(f"{extract_value(fname, technique):.1f}")
        
    widths = [img.shape[1] for img in imgs]
    
    # To handle spacer, we insert a dummy width
    if input_image_path:
        # Spacer width is 20% of an average generated image's width
        avg_w = sum(widths[1:]) / len(widths[1:]) if len(widths) > 1 else widths[0]
        spacer_w = avg_w * 0.2
        plot_widths = [widths[0], spacer_w] + widths[1:]
    else:
        plot_widths = widths
        
    total_w = sum(plot_widths)
    max_h = max([img.shape[0] for img in imgs])
    
    # We set figsize proportional to total pixels
    # Let 100 pixels = 1 inch
    fig_w = total_w / 100.0
    fig_h = max_h / 100.0 * 1.3 # add 30% space for titles
    
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    gs = gridspec.GridSpec(1, len(plot_widths), width_ratios=plot_widths)
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, bottom=0.0, top=0.85)
    
    ax_idx = 0
    for i, (img, label) in enumerate(zip(imgs, labels)):
        if i == 1 and input_image_path:
            # Add spacer
            ax_spacer = fig.add_subplot(gs[ax_idx])
            ax_spacer.axis('off')
            ax_idx += 1
            
        ax = fig.add_subplot(gs[ax_idx])
        ax.imshow(img, aspect='equal')
        ax.axis('off')
        ax.text(0.5, -0.05, label, ha='center', va='top', transform=ax.transAxes, fontsize=40, fontweight='bold')
        ax_idx += 1
        
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot image grid for parameter sweeps.")
    parser.add_argument("--synth-root", type=str, required=True, help="Directory containing the sweep subdirectories (e.g. data_generation_outputs/ip_sweeps or data_generation_outputs/cn_sweeps).")
    parser.add_argument("--technique", type=str, choices=["ip", "cn"], required=True, help="Which technique to plot (ip for IP-Adapter, cn for ControlNet).")
    parser.add_argument("--out-dir", type=str, default="/Users/susheelutagi/Documents/GitHub/Honours_2026/experiments/figures", help="Directory to save the plots.")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.synth_root)
    out_dir = os.path.abspath(args.out_dir)
    
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(base_dir):
        print(f"Base directory does not exist: {base_dir}")
        return
        
    for subdir in os.listdir(base_dir):
        sweep_dir = os.path.join(base_dir, subdir)
        if os.path.isdir(sweep_dir):
            save_path = os.path.join(out_dir, f"{subdir}_figure.png")
            create_figure_for_sweep(sweep_dir, save_path, technique=args.technique)

if __name__ == "__main__":
    main()
