import os
import json
import matplotlib.pyplot as plt
import numpy as np

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
            tech = parts[-1].upper()
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
        
        # --- Helper function for drawing dual axis histograms ---
        def plot_dual_hist(ax, real_data, synth_data, bins, title, xlabel):
            color_real = '#4A90D9'
            color_synth = '#E8833A'
            
            # Primary axis for Real Data
            ax.hist(real_data, bins=bins, alpha=0.5, color=color_real, label='Real')
            ax.set_ylabel("Count (Real)", color=color_real, fontweight='bold')
            ax.tick_params(axis='y', labelcolor=color_real)
            
            # Plot Real mean vertical dash
            real_mean = np.mean(real_data)
            ax.axvline(real_mean, color='#1565C0', linestyle='--', linewidth=2, label=f'Real Mean: {real_mean:.3f}')
            
            # Create independent secondary axis for Synthetic Data
            ax2 = ax.twinx()
            ax2.hist(synth_data, bins=bins, alpha=0.4, color=color_synth, label='Synthetic')
            ax2.set_ylabel("Count (Synthetic)", color=color_synth, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor=color_synth)
            
            # Plot Synth mean vertical dash
            synth_mean = np.mean(synth_data)
            ax2.axvline(synth_mean, color='#D84315', linestyle='--', linewidth=2, label=f'Synth Mean: {synth_mean:.3f}')
            
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
