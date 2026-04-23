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
    
    labels = []
    cfd_scores = []
    wddp_scores = []
    
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"Warning: Class directory missing: {cls_dir}")
            continue
        
        for tech_dir_name in sorted(os.listdir(cls_dir)):
            tech_dir = os.path.join(cls_dir, tech_dir_name)
            if not os.path.isdir(tech_dir):
                continue
            
            json_path = os.path.join(tech_dir, "metrics_summary.json")
            if not os.path.exists(json_path):
                print(f"Warning: Missing {json_path}")
                continue
            
            print(f"Processing {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Create a label like "Toaster (IP)"
            parts = tech_dir_name.split('_')
            tech = parts[-1].upper()
            cls_name = cls.replace('_', ' ').title()
            label = f"{cls_name} ({tech})"
            
            labels.append(label)
            # Default to 0 if the keys are missing for any reason
            cfd_scores.append(data.get("clip_frechet_distance", 0))
            wddp_scores.append(data.get("wasserstein_distance_domain_projection", 0))

    if not labels:
        print("No valid data found to plot.")
        return

    # Plotting
    x = np.arange(len(labels))
    width = 0.5
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: CLIP Frechet Distance
    bars1 = ax1.bar(x, cfd_scores, width, color='skyblue')
    ax1.set_ylabel('CLIP Frechet Distance')
    ax1.set_title('CLIP Frechet Distance by Class & Technique')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add values on top of bars for readability
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Subplot 2: Wasserstein Distance (Domain Projection)
    bars2 = ax2.bar(x, wddp_scores, width, color='salmon')
    ax2.set_ylabel('Wasserstein Distance')
    ax2.set_title('Wasserstein Dist. (Domain Projection) by Class & Technique')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add values on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    
    out_path = os.path.join(base_dir, "distance_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccessfully generated plot: {out_path}")

if __name__ == '__main__':
    main()
