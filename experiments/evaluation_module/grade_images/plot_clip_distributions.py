import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_scores(class_name, tech):
    base_dir = os.path.join(
        os.path.dirname(__file__),
        '..', '..', '..', 'results', 'clip_analysis'
    )
    path = os.path.join(base_dir, class_name, f"{class_name}_{tech}", "raw_data", "synth_domain_scores.json")
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return []
    with open(path, 'r') as f:
        return json.load(f)

def main():
    classes = ["toaster", "hair_drier"]
    techs = ["cn", "ip"]  # ControlNet, IP-Adapter
    
    tech_names = {"cn": "ControlNet", "ip": "IP-Adapter"}
    
    fig, axes = plt.subplots(1, len(techs), figsize=(12, 5), sharey=True)
    if len(techs) == 1:
        axes = [axes]
        
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'clip_analysis')
    os.makedirs(out_dir, exist_ok=True)
    
    colors = {"toaster": "#FF9800", "hair_drier": "#2196F3"}
    
    for i, tech in enumerate(techs):
        ax = axes[i]
        
        for cls in classes:
            scores = load_scores(cls, tech)
            if not scores:
                continue
                
            label = f"{cls.replace('_', ' ').title()}"
            # Use matplotlib hist instead of seaborn kdeplot
            ax.hist(scores, bins=30, density=True, alpha=0.5, label=label, color=colors[cls], edgecolor='black', linewidth=0.5)
            
            # Add a KDE line using scipy if available
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(scores)
                x_range = np.linspace(min(scores), max(scores), 200)
                ax.plot(x_range, kde(x_range), color=colors[cls], linewidth=2)
            except ImportError:
                pass
            
        ax.set_title(f"CLIP Scores: {tech_names[tech]}", fontsize=14, fontweight='bold')
        ax.set_xlabel("CLIP Score", fontsize=12)
        if i == 0:
            ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    fig.suptitle("CLIP Score Distributions: Synthetic Toasters vs Hair Dryers", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, "clip_score_distributions_toaster_vs_hairdrier.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
