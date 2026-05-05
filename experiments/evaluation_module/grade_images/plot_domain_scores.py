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
    data = {} # {class: {tech: cfd}}
    
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        
        data[cls] = {}
        
        for tech_dir_name in sorted(os.listdir(cls_dir)):
            tech_dir = os.path.join(cls_dir, tech_dir_name)
            if not os.path.isdir(tech_dir):
                continue
            
            metrics_path = os.path.join(tech_dir, "metrics_summary.json")
            if not os.path.exists(metrics_path):
                continue
                
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                cfd = metrics.get("clip_frechet_distance")
                if cfd is None: continue
            except Exception:
                continue

            # Identify technique
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
                
            data[cls][tech] = cfd

    if not data:
        print("No CFD data found. Have you re-run analyse.py?")
        return

    # Prepare for plotting
    all_techs = sorted(list(set(t for c in data.values() for t in c.keys())))
    plot_classes = sorted(data.keys())
    
    x = np.arange(len(plot_classes))
    # Thinner bars: reduced multiplier from 0.8 to 0.5
    width = 0.5 / len(all_techs)
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    colors = ['#673AB7', '#00BFA5', '#FF9800', '#2196F3', '#E91E63']
    
    for i, tech in enumerate(all_techs):
        vals = [data[cls].get(tech, 0) for cls in plot_classes]
        offset = (i - (len(all_techs)-1)/2) * width
        rects = ax.bar(x + offset, vals, width, label=tech, color=colors[i % len(colors)], alpha=0.85, edgecolor='black', linewidth=1)
        
        # Add value labels above bars (larger font)
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('CLIP Fréchet Distance (CFD)', fontsize=14, fontweight='bold')
    ax.set_title('Axis 1: Semantic Coherence (Lower is Better)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in plot_classes], fontsize=14, fontweight='bold')
    # Larger legend text
    ax.legend(fontsize=16, frameon=True, shadow=True)
    # Grid removed
    
    fig.tight_layout()
    
    out_path = os.path.join(base_dir, "clip_frechet_distance_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSuccessfully generated CFD comparison bar chart: {out_path}")

if __name__ == '__main__':
    main()
