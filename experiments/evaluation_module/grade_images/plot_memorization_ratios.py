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
                "pixel": os.path.join(raw_dir, f"memorization_ratios_pixel_{tech_raw}.json"),
                "clip": os.path.join(raw_dir, f"memorization_ratios_clip_{tech_raw}.json"),
            }
            
            if all(os.path.exists(p) for p in paths.values()):
                configs.append({
                    "class": cls_name,
                    "tech": tech,
                    "label": label,
                    "paths": paths
                })

    if not configs:
        print("No raw data found to plot. Run analyse.py first.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Distinct colour palette for the configurations
    color_map = {
        "Toaster (IP-Adapter)": {"fill": "#E8833A", "curve": "#D84315"},  # Orange
        "Toaster (ControlNet)": {"fill": "#AB47BC", "curve": "#7B1FA2"},  # Purple
        "Hair Drier (IP-Adapter)": {"fill": "#26A69A", "curve": "#00796B"}, # Teal
        "Hair Drier (ControlNet)": {"fill": "#4A90D9", "curve": "#1565C0"}, # Blue
    }
    
    fallback_palette = [
        {"fill": "#E8833A", "curve": "#D84315"}, 
        {"fill": "#4A90D9", "curve": "#1565C0"}, 
        {"fill": "#26A69A", "curve": "#00796B"}, 
        {"fill": "#AB47BC", "curve": "#7B1FA2"}, 
    ]

    toaster_configs = [c for c in configs if c["class"].lower() == "toaster"]
    hair_drier_configs = [c for c in configs if c["class"].lower() == "hair drier"]

    def plot_distributions(ax, target_configs, space_key, title, xlabel, is_bottom_row):
        all_vals = []
        for c in target_configs:
            vals = _load_json_list(c["paths"][space_key])
            all_vals.extend(vals)
            
        if not all_vals:
            ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha='center')
            return
            
        x_min = min(all_vals)
        x_max = max(all_vals)
        margin = (x_max - x_min) * 0.08
        x_lo = max(0.0, x_min - margin)
        x_hi = x_max + margin
        
        bins = np.linspace(x_lo, x_hi, 50)
        bin_width = bins[1] - bins[0]
        x_fit = np.linspace(x_lo, x_hi, 300)
        
        for i, config in enumerate(target_configs):
            label = config["label"]
            vals = _load_json_list(config["paths"][space_key])
            
            pal = color_map.get(label, fallback_palette[i % len(fallback_palette)])
            
            # Plot transparent histogram
            ax.hist(vals, bins=bins, alpha=0.15, color=pal["fill"])
            
            # Plot Gaussian curve of best fit
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            pdf = norm.pdf(x_fit, mean_val, std_val) * len(vals) * bin_width
            ax.plot(x_fit, pdf, color=pal["curve"], linewidth=2.5, alpha=0.85, label=f"{label} ($\mu={mean_val:.2f}$)")
            
            # Plot vertical line for mean
            ax.axvline(mean_val, color=pal["curve"], linestyle='--', linewidth=2, zorder=5)
            
        # R=1 reference line
        ax.axvline(1.0, color="#424242", linestyle="-", linewidth=2.5, zorder=6, label="R = 1 (Equidistant)")
        
        ax.set_title(title, fontsize=16, pad=15, fontweight='bold')
        if is_bottom_row:
            ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
        else:
            ax.set_xlabel('')
        ax.tick_params(axis='x', labelsize=12)
            
        ax.set_ylabel("Count", fontsize=14, fontweight='bold')
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(axis="y", alpha=0.2)
        
        ax.legend(fontsize=11, loc="upper right", framealpha=0.9)

    # Plot Toaster (Row 0)
    plot_distributions(axes[0, 0], toaster_configs, "pixel", "Toaster – Pixel Space Memorization Ratios", "R = d(syn, src) / d(syn, next_nearest_neighbour)", False)
    plot_distributions(axes[0, 1], toaster_configs, "clip", "Toaster – CLIP Space Memorization Ratios", "R = d(syn, src) / d(syn, next_nearest_neighbour)", False)
    
    # Plot Hair Drier (Row 1)
    plot_distributions(axes[1, 0], hair_drier_configs, "pixel", "Hair Drier – Pixel Space Memorization Ratios", "R = d(syn, src) / d(syn, next_nearest_neighbour)", True)
    plot_distributions(axes[1, 1], hair_drier_configs, "clip", "Hair Drier – CLIP Space Memorization Ratios", "R = d(syn, src) / d(syn, next_nearest_neighbour)", True)
    
    fig.tight_layout(pad=3.0)
    
    out_path = os.path.join(base_dir, "memorization_ratios_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSuccessfully generated memorization ratios plot: {out_path}")

if __name__ == '__main__':
    main()
