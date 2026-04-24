import os
import json
import matplotlib.pyplot as plt
import numpy as np  

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
    
    # Structure: {"Toaster": {"IP-Adapter": (cfd_val, wddp_val), ...}, "Hair Drier": {...}}
    data_points = {}
    
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        
        cls_name = cls.replace('_', ' ').title()
        if cls_name not in data_points:
            data_points[cls_name] = {}
        
        for tech_dir_name in sorted(os.listdir(cls_dir)):
            tech_dir = os.path.join(cls_dir, tech_dir_name)
            if not os.path.isdir(tech_dir):
                continue
            
            json_path = os.path.join(tech_dir, "metrics_summary.json")
            if not os.path.exists(json_path):
                continue
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract technique
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
                
            cfd = data.get("clip_frechet_distance", None)
            wddp = data.get("wasserstein_distance_domain_projection", None)
            
            if cfd is not None and wddp is not None:
                data_points[cls_name][tech] = (cfd, wddp)

    if not any(data_points.values()):
        print("No valid data found to plot.")
        return

    # Visual Configurations
    # Colors represent the Class
    color_map = {"Toaster": "#2980b9", "Hair Drier": "#d35400"}
    # Shapes represent the generative technique
    marker_map = {"IP-Adapter": "o", "ControlNet": "^", "Textual Inversion": "s"}
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    for cls_name, techs in data_points.items():
        color = color_map.get(cls_name, "black")
        
        # Connect points of the same class with a dashed line to show the "shift"
        if len(techs) > 1:
            pts = list(techs.values())
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=color, linestyle='--', alpha=0.4, linewidth=2, zorder=1)
        
        # Plot each point
        for tech, (cfd, wddp) in techs.items():
            marker = marker_map.get(tech, "x")
            # The label combines both so the legend clarifies shape vs color mapping
            label_text = f"{cls_name} ({tech})"
            
            ax.scatter(cfd, wddp, color=color, marker=marker, s=200, 
                       edgecolors='white', linewidth=1.5, zorder=2, label=label_text)
            
            # Annotate the point so the user doesn't have to bounce eyes back from legend
            ax.annotate(tech, (cfd, wddp), xytext=(12, -4), textcoords='offset points', 
                        fontsize=10, color=color, fontweight='bold')

    ax.set_title("Global Synthetic Data Shift: CLIP Fréchet vs. 1D Wasserstein", fontsize=14, fontweight='bold', pad=15)
    
    # Important: Smaller values are typically better (closer distribution match)
    ax.set_xlabel("CLIP Fréchet Distance (lower is closer)", fontsize=11, fontweight='bold')
    ax.set_ylabel("1D Wasserstein Domain Projection Dist. (lower is closer)", fontsize=11, fontweight='bold')
    
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Sort legend so it groups neatly
    handles, labels = ax.get_legend_handles_labels()
    sorted_idx = np.argsort(labels)
    ax.legend([handles[i] for i in sorted_idx], [labels[i] for i in sorted_idx], 
              loc='upper right', framealpha=0.9, title="Configurations")
    
    plt.tight_layout()
    
    out_path = os.path.join(base_dir, "distance_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccessfully generated 2D Scatter plot: {out_path}")

if __name__ == '__main__':
    main()
