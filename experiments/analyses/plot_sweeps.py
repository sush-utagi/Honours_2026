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

def create_thesis_figure(base_dir, save_path, technique):
    """Create a single compact figure with all sweeps stacked as rows.
    
    Layout:
        - Left column: real/input images stacked vertically, single "Real" label at bottom
        - Right columns: synthetic images in a single row per sweep, no inter-image padding
        - Strength labels only on the bottom row
    """
    param_key = "ips" if technique == "ip" else "cs"
    
    # Collect all sweep subdirectories and their data
    sweep_data = []
    for subdir in sorted(os.listdir(base_dir)):
        sweep_dir = os.path.join(base_dir, subdir)
        if not os.path.isdir(sweep_dir):
            continue
            
        input_images = glob.glob(os.path.join(sweep_dir, "input.*"))
        input_image_path = input_images[0] if input_images else None
        
        gen_images = [fp for fp in os.listdir(sweep_dir)
                      if f"_{param_key}" in fp and fp.lower().endswith((".png", ".jpg", ".jpeg"))]
        gen_images.sort(key=lambda x: extract_value(x, technique))
        
        if not gen_images:
            print(f"Skipping {sweep_dir}, no generated images found.")
            continue
        
        input_img = mpimg.imread(input_image_path) if input_image_path else None
        synth_imgs = [mpimg.imread(os.path.join(sweep_dir, f)) for f in gen_images]
        synth_labels = [f"{extract_value(f, technique):.1f}" for f in gen_images]
        
        sweep_data.append({
            "name": subdir,
            "input_img": input_img,
            "synth_imgs": synth_imgs,
            "synth_labels": synth_labels,
        })
    
    if not sweep_data:
        print("No sweep data found.")
        return
    
    n_rows = len(sweep_data)
    n_synth = len(sweep_data[0]["synth_imgs"])  # assume consistent across sweeps
    has_input = any(s["input_img"] is not None for s in sweep_data)
    
    # Build GridSpec columns: [input | spacer | synth_0 | synth_1 | ... | synth_n-1]
    spacer_ratio = 0.15  # thin gap between real and synthetic columns
    if has_input:
        n_cols = 1 + 1 + n_synth  # input + spacer + synths
        width_ratios = [1, spacer_ratio] + [1] * n_synth
    else:
        n_cols = n_synth
        width_ratios = [1] * n_cols
    
    # Figure sizing: make each cell roughly square, compact
    cell_size = 1.6  # inches per cell
    label_margin = 0.25  # extra space at the bottom for labels
    fig_w = cell_size * (sum(width_ratios))
    fig_h = cell_size * n_rows + label_margin
    
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(n_rows, n_cols, width_ratios=width_ratios,
                           wspace=0.0, hspace=0.0)
    # Tight margins: small bottom margin for the strength labels
    gs.update(left=0.0, right=1.0, top=1.0, bottom=label_margin / fig_h)
    
    for row_idx, sweep in enumerate(sweep_data):
        is_last_row = (row_idx == n_rows - 1)
        
        # -- Input image column --
        if has_input:
            ax_in = fig.add_subplot(gs[row_idx, 0])
            if sweep["input_img"] is not None:
                ax_in.imshow(sweep["input_img"], aspect='equal')
            ax_in.axis('off')
            if is_last_row:
                ax_in.text(0.5, -0.06, "Real", ha='center', va='top',
                           transform=ax_in.transAxes, fontsize=16, fontweight='bold')
            # Spacer column (invisible)
            ax_sp = fig.add_subplot(gs[row_idx, 1])
            ax_sp.axis('off')
        
        # -- Synthetic images --
        synth_start_col = 2 if has_input else 0
        for i, img in enumerate(sweep["synth_imgs"]):
            col = synth_start_col + i
            ax = fig.add_subplot(gs[row_idx, col])
            ax.imshow(img, aspect='equal')
            ax.axis('off')
            
            # Strength label only on the bottom row
            if is_last_row:
                ax.text(0.5, -0.06, sweep["synth_labels"][i], ha='center', va='top',
                        transform=ax.transAxes, fontsize=16, fontweight='bold')
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=300)
    plt.close(fig)
    print(f"Saved thesis figure ({n_rows} sweeps): {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot image grid for parameter sweeps.")
    parser.add_argument("--synth-root", type=str, required=True, help="Directory containing the sweep subdirectories (e.g. data_generation_outputs/ip_sweeps or data_generation_outputs/cn_sweeps).")
    parser.add_argument("--technique", type=str, choices=["ip", "cn"], required=True, help="Which technique to plot (ip for IP-Adapter, cn for ControlNet).")
    parser.add_argument("--out-dir", type=str, default="/Users/susheelutagi/Documents/GitHub/Honours_2026/experiments/figures", help="Directory to save the plots.")
    parser.add_argument("--thesis", action="store_true", help="Thesis mode: combine all sweeps into a single compact figure with minimal padding.")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.synth_root)
    out_dir = os.path.abspath(args.out_dir)
    
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(base_dir):
        print(f"Base directory does not exist: {base_dir}")
        return
    
    if args.thesis:
        tech_label = "ip_adapter" if args.technique == "ip" else "controlnet"
        save_path = os.path.join(out_dir, f"thesis_{tech_label}_sweeps.png")
        create_thesis_figure(base_dir, save_path, technique=args.technique)
    else:
        for subdir in os.listdir(base_dir):
            sweep_dir = os.path.join(base_dir, subdir)
            if os.path.isdir(sweep_dir):
                save_path = os.path.join(out_dir, f"{subdir}_figure.png")
                create_figure_for_sweep(sweep_dir, save_path, technique=args.technique)

if __name__ == "__main__":
    main()
