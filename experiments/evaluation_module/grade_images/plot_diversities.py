import os
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def _load_json_list(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    base_dir = os.path.join(
        os.path.dirname(__file__),
        '..', '..', '..', 'results', 'clip_analysis'
    )
    base_dir = os.path.abspath(base_dir)
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist.")
        return

    classes = ["toaster", "hair_drier"]
    
    configs = []
    
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        
        for tech_dir_name in sorted(os.listdir(cls_dir)):
            tech_dir = os.path.join(cls_dir, tech_dir_name)
            if not os.path.isdir(tech_dir):
                continue
            
            raw_dir = os.path.join(tech_dir, "raw_data")
            if not os.path.isdir(raw_dir):
                continue
                
            parts = tech_dir_name.split('_')
            tech_raw = parts[-1].lower()
            if tech_raw == 'ip':
                tech = 'IP-Adapter'
            elif tech_raw == 'cn':
                tech = 'ControlNet'
            elif tech_raw == 'ti':
                tech = 'Textual Inversion'
            else:
                tech = tech_raw.upper()
                
            cls_name = cls.replace('_', ' ').title()
            label = f"{cls_name} ({tech})"
            
            paths = {
                "real_structural": os.path.join(raw_dir, "real_structural_distances.json"),
                "synth_structural": os.path.join(raw_dir, "synth_structural_distances.json"),
                "real_semantic": os.path.join(raw_dir, "real_semantic_distances.json"),
                "synth_semantic": os.path.join(raw_dir, "synth_semantic_distances.json"),
            }
            
            if all(os.path.exists(p) for p in paths.values()):
                configs.append({
                    "label": label,
                    "paths": paths
                })

    if not configs:
        print("No raw data found to plot. Have you re-run analyse.py to generate .json arrays?")
        return

    n_rows = len(configs)
    n_cols = 2
    
    # Increase the figure size a bit to perfectly fit the twin axes space without overlap
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    if n_rows == 1:
        axes = np.array([axes])

    for i, config in enumerate(configs):
        label = config["label"]
        paths = config["paths"]
        
        real_struct = _load_json_list(paths["real_structural"])
        synth_struct = _load_json_list(paths["synth_structural"])
        real_sem = _load_json_list(paths["real_semantic"])
        synth_sem = _load_json_list(paths["synth_semantic"])
        
        ax_struct = axes[i, 0]
        ax_sem = axes[i, 1]
        
        def plot_dual_hist(ax, real_data, synth_data, bins, title, xlabel):
            color_real = '#4A90D9'
            color_synth = '#E8833A'
            
            text_color_real = '#1565C0'   # Dark Blue
            text_color_synth = '#BF360C'  # Dark Orange/Rust
            
            bin_width = bins[1] - bins[0]
            x_fit = np.linspace(bins[0], bins[-1], 200)
            
            # Primary axis for Real Data
            ax.hist(real_data, bins=bins, alpha=0.5, color=color_real, label='Real')
            ax.set_ylabel("Count (Real)", color=text_color_real, fontweight='bold')
            ax.tick_params(axis='y', labelcolor=text_color_real)
            
            real_mean, real_std = np.mean(real_data), np.std(real_data)
            real_pdf = norm.pdf(x_fit, real_mean, real_std) * len(real_data) * bin_width
            ax.plot(x_fit, real_pdf, color='#1565C0', linewidth=2.5, alpha=0.75, zorder=4)
            
            # Create independent secondary axis for Synthetic Data
            ax2 = ax.twinx()
            ax2.hist(synth_data, bins=bins, alpha=0.4, color=color_synth, label='Synthetic')
            ax2.set_ylabel("Count (Synthetic)", color=text_color_synth, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor=text_color_synth)
            
            synth_mean, synth_std = np.mean(synth_data), np.std(synth_data)
            synth_pdf = norm.pdf(x_fit, synth_mean, synth_std) * len(synth_data) * bin_width
            ax2.plot(x_fit, synth_pdf, color='#D84315', linewidth=2.5, alpha=0.75, zorder=4)
            
            # Plot Real mean vertical dash ON AX2
            ax2.axvline(real_mean, color='#1565C0', linestyle='--', linewidth=2, zorder=5, label=f'Real Mean: {real_mean:.3f}')
            
            # Plot Synth mean vertical dash
            ax2.axvline(synth_mean, color='#D84315', linestyle='--', linewidth=2, zorder=5, label=f'Synth Mean: {synth_mean:.3f}')
            
            # Style & Labels
            ax.set_title(title, fontsize=12, pad=15, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=11)
            ax.grid(axis="y", alpha=0.2)
            
            # We must coalesce both legends properly
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper right", framealpha=0.9, fontsize=9)

        # 1. Structural Diversity plotting
        bins_struct = np.linspace(0.0, 1.2, 50)
        plot_dual_hist(ax_struct, real_struct, synth_struct, bins_struct, 
                       f"{label} - Structural Diversity (LPIPS)", "LPIPS Distance")
                       
        # 2. Semantic Diversity plotting
        bins_sem = np.linspace(0.0, 1.0, 50)
        plot_dual_hist(ax_sem, real_sem, synth_sem, bins_sem, 
                       f"{label} - Semantic Diversity (CLIP)", "Cosine Distance")

    fig.tight_layout(h_pad=3.0)  # slightly more vertical padding for readability
    
    out_path = os.path.join(base_dir, "diversity_histograms.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSuccessfully generated histogram grid: {out_path}")

if __name__ == '__main__':
    main()
