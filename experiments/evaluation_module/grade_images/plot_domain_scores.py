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
                "real": os.path.join(raw_dir, "real_domain_scores.json"),
                "synth": os.path.join(raw_dir, "synth_domain_scores.json"),
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
    
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 5 * n_rows))
    
    if n_rows == 1:
        axes = np.array([axes])

    for i, config in enumerate(configs):
        label = config["label"]
        paths = config["paths"]
        
        real_scores = _load_json_list(paths["real"])
        synth_scores = _load_json_list(paths["synth"])
        
        ax = axes[i]
        
        # Determine the shared binning across both so they align perfectly
        min_val = min(np.min(real_scores), np.min(synth_scores))
        max_val = max(np.max(real_scores), np.max(synth_scores))
        bins = np.linspace(min_val * 0.95, max_val * 1.05, 50)
        
        color_real = '#673AB7'    # Deep Purple
        color_synth = '#00BFA5'   # Teal
        
        # Primary axis for Real Data
        ax.hist(real_scores, bins=bins, alpha=0.5, color=color_real, label='Real')
        ax.set_ylabel("Count (Real)", color=color_real, fontweight='bold')
        ax.tick_params(axis='y', labelcolor=color_real)
        
        # Create independent secondary axis for Synthetic Data
        ax2 = ax.twinx()
        ax2.hist(synth_scores, bins=bins, alpha=0.4, color=color_synth, label='Synthetic')
        ax2.set_ylabel("Count (Synthetic)", color=color_synth, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color_synth)
        
        # Plot Real mean vertical dash ON AX2 so it layers above all histograms!
        real_mean = np.mean(real_scores)
        ax2.axvline(real_mean, color='#4527A0', linestyle='--', linewidth=2.5, zorder=5, label=f'Real Mean: {real_mean:.1f}')
        
        # Plot Synth mean vertical dash
        synth_mean = np.mean(synth_scores)
        ax2.axvline(synth_mean, color='#00695C', linestyle='--', linewidth=2.5, zorder=5, label=f'Synth Mean: {synth_mean:.1f}')
        
        # Formating & Titles
        ax.set_title(f"{label} - Domain Level CLIP Scores", fontsize=12, pad=15, fontweight='bold')
        ax.set_xlabel("CLIP Score (higher is better aligned with domain)", fontsize=11)
        ax.grid(axis="y", alpha=0.15)
        
        # Combine legends from both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left", framealpha=0.9, fontsize=9)

    fig.tight_layout(h_pad=3.0)
    
    out_path = os.path.join(base_dir, "domain_scores_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSuccessfully generated histogram column: {out_path}")

if __name__ == '__main__':
    main()
