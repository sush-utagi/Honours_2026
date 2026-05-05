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
    input_img = mpimg.imread(input_image_path) if input_image_path else None
    synth_imgs = []
    synth_labels = []
    
    for fname in gen_images:
        synth_imgs.append(mpimg.imread(os.path.join(sweep_dir, fname)))
        synth_labels.append(f"{extract_value(fname, technique):.1f}")
        
    n_synth = len(synth_imgs)
    cols_synth = (n_synth + 1) // 2
    
    # Calculate widths for GridSpec
    avg_synth_w = sum(img.shape[1] for img in synth_imgs) / n_synth
    spacer_w = avg_synth_w * 0.2
    
    # We'll use a fixed width ratio for the synth cols based on average width
    # to keep the grid aligned even if images vary slightly
    input_w = input_img.shape[1] if input_img is not None else 0
    
    plot_widths = []
    if input_img is not None:
        plot_widths.append(input_w)
        plot_widths.append(spacer_w)
    
    for _ in range(cols_synth):
        plot_widths.append(avg_synth_w)
        
    total_w = sum(plot_widths)
    max_h_single = max(img.shape[0] for img in synth_imgs)
    if input_img is not None:
        max_h_single = max(max_h_single, input_img.shape[0] / 2)
        
    fig_w = total_w / 100.0
    # Height for 2 rows + spacing for labels
    fig_h = (max_h_single * 2) / 100.0 * 1.4 
    
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(2, len(plot_widths), width_ratios=plot_widths)
    # Significantly reduced hspace for a tight vertical fit
    gs.update(wspace=0.0, hspace=0.1, left=0.0, right=1.0, bottom=0.05, top=0.9)
    
    # Plot Input Image (spanning both rows)
    if input_img is not None:
        tech_name = "IP-Adapter" if technique == "ip" else "ControlNet"
        ax_input = fig.add_subplot(gs[:, 0])
        ax_input.imshow(input_img, aspect='equal')
        ax_input.axis('off')
        ax_input.text(0.5, -0.04, f"Input ({tech_name})", ha='center', va='top', transform=ax_input.transAxes, fontsize=40, fontweight='bold')
        
        # Spacer
        ax_spacer = fig.add_subplot(gs[:, 1])
        ax_spacer.axis('off')

    # Plot Synthetic Images
    start_col = 2 if input_img is not None else 0
    for i in range(n_synth):
        row = 0 if i < cols_synth else 1
        col = start_col + (i % cols_synth)
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(synth_imgs[i], aspect='equal')
        ax.axis('off')
        ax.text(0.5, -0.04, synth_labels[i], ha='center', va='top', transform=ax.transAxes, fontsize=40, fontweight='bold')
        
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)
    print(f"Saved 2-row figure: {save_path}")

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
