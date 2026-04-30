import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
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
    
    # ── Group techniques by class ────────────────────────────────────────
    # class_data[cls_name] = {
    #     "real_structural": [...],  "real_semantic": [...],
    #     "techniques": [ { "tech": "IP-Adapter", "synth_structural": [...], "synth_semantic": [...] }, ... ]
    # }
    class_data = OrderedDict()
    
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        
        cls_name = cls.replace('_', ' ').title()
        real_struct = None
        real_sem = None
        techniques = []
        
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
            
            paths = {
                "real_structural": os.path.join(raw_dir, "real_structural_distances.json"),
                "synth_structural": os.path.join(raw_dir, "synth_structural_distances.json"),
                "real_semantic": os.path.join(raw_dir, "real_semantic_distances.json"),
                "synth_semantic": os.path.join(raw_dir, "synth_semantic_distances.json"),
            }
            
            if not all(os.path.exists(p) for p in paths.values()):
                continue
            
            # Real data is the same across techniques for a given class;
            # load it from the first technique we encounter.
            if real_struct is None:
                real_struct = _load_json_list(paths["real_structural"])
                real_sem = _load_json_list(paths["real_semantic"])
            
            techniques.append({
                "tech": tech,
                "synth_structural": _load_json_list(paths["synth_structural"]),
                "synth_semantic": _load_json_list(paths["synth_semantic"]),
            })
        
        if real_struct is not None and techniques:
            class_data[cls_name] = {
                "real_structural": real_struct,
                "real_semantic": real_sem,
                "techniques": techniques,
            }

    if not class_data:
        print("No raw data found to plot. Have you re-run analyse.py to generate .json arrays?")
        return

    # ── Colour palette ───────────────────────────────────────────────────
    # Real keeps the original blue.
    # Each technique gets its own warm colour for its synthetic distribution.
    color_real      = '#4A90D9'
    text_color_real = '#1565C0'
    curve_color_real = '#1565C0'
    
    synth_palette = [
        {'fill': '#E8833A', 'curve': '#D84315', 'text': '#BF360C', 'name_suffix': ''},   # Orange
        {'fill': '#AB47BC', 'curve': '#7B1FA2', 'text': '#6A1B9A', 'name_suffix': ''},   # Purple
        {'fill': '#26A69A', 'curve': '#00796B', 'text': '#00695C', 'name_suffix': ''},   # Teal (spare)
    ]

    # ── Build figure ─────────────────────────────────────────────────────
    n_rows = len(class_data)
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    if n_rows == 1:
        axes = np.array([axes])

    def plot_combined_hist(ax, real_data, synth_entries, bins, title, xlabel,
                           is_bottom_row=True):
        """
        Plot Real on the primary y-axis and all synthetic sets on a shared
        secondary y-axis, each with its own colour.
        
        synth_entries: list of (tech_label, synth_data, palette_dict)
        is_bottom_row: if False, x-axis label and tick labels are hidden
        """
        # ── Data-driven x-limits ─────────────────────────────────────────
        all_values = list(real_data)
        for _, sd, _ in synth_entries:
            all_values.extend(sd)
        x_min = min(all_values)
        x_max = max(all_values)
        margin = (x_max - x_min) * 0.08  # 8 % breathing room each side
        x_lo = max(0.0, x_min - margin)
        x_hi = x_max + margin
        
        # Re-bin within the zoomed range
        bins = np.linspace(x_lo, x_hi, 50)
        bin_width = bins[1] - bins[0]
        x_fit = np.linspace(x_lo, x_hi, 300)
        
        # ── Primary axis: Real ───────────────────────────────────────────
        ax.hist(real_data, bins=bins, alpha=0.15, color=color_real, label='Real')
        ax.set_ylabel("Count (Real)", color=text_color_real, fontweight='bold')
        ax.tick_params(axis='y', labelcolor=text_color_real)
        
        real_mean, real_std = np.mean(real_data), np.std(real_data)
        real_pdf = norm.pdf(x_fit, real_mean, real_std) * len(real_data) * bin_width
        ax.plot(x_fit, real_pdf, color=curve_color_real, linewidth=2.5, alpha=0.85, zorder=4)
        
        # ── Secondary axis: Synthetic sets ───────────────────────────────
        ax2 = ax.twinx()
        ax2.set_ylabel("Count (Synthetic)", fontweight='bold')
        
        for tech_label, synth_data, pal in synth_entries:
            ax2.hist(synth_data, bins=bins, alpha=0.12, color=pal['fill'],
                     label=f'Synth ({tech_label})')
            
            s_mean, s_std = np.mean(synth_data), np.std(synth_data)
            s_pdf = norm.pdf(x_fit, s_mean, s_std) * len(synth_data) * bin_width
            ax2.plot(x_fit, s_pdf, color=pal['curve'], linewidth=2.5, alpha=0.85, zorder=4)
            ax2.axvline(s_mean, color=pal['curve'], linestyle='--', linewidth=2, zorder=5,
                        label=f'{tech_label} Mean: {s_mean:.3f}')
        
        # Real mean line (drawn on ax2 so it shares the x-range)
        ax2.axvline(real_mean, color=curve_color_real, linestyle='--', linewidth=2, zorder=5,
                    label=f'Real Mean: {real_mean:.3f}')
        
        # ── Zoom x-axis ──────────────────────────────────────────────────
        ax.set_xlim(x_lo, x_hi)
        ax2.set_xlim(x_lo, x_hi)
        
        # ── X-axis: only label & tick on the bottom row ──────────────────
        if is_bottom_row:
            ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelbottom=False)
        
        # ── Style & combined legend ──────────────────────────────────────
        ax.set_title(title, fontsize=12, pad=15, fontweight='bold')
        ax.grid(axis="y", alpha=0.2)
        
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", framealpha=0.9, fontsize=9)

    # ── Iterate per class ────────────────────────────────────────────────
    for row_idx, (cls_name, data) in enumerate(class_data.items()):
        ax_struct = axes[row_idx, 0]
        ax_sem    = axes[row_idx, 1]
        is_last_row = (row_idx == n_rows - 1)
        
        # Build per-technique entries with palette assignment
        synth_struct_entries = []
        synth_sem_entries = []
        for t_idx, t in enumerate(data["techniques"]):
            pal = synth_palette[t_idx % len(synth_palette)]
            synth_struct_entries.append((t["tech"], t["synth_structural"], pal))
            synth_sem_entries.append((t["tech"], t["synth_semantic"], pal))
        
        # bins arg is now only a placeholder; plot_combined_hist recomputes
        # bins from the actual data range.
        plot_combined_hist(ax_struct, data["real_structural"], synth_struct_entries,
                           None,
                           f"{cls_name} – Structural Diversity (LPIPS)", "LPIPS Distance",
                           is_bottom_row=is_last_row)
        
        plot_combined_hist(ax_sem, data["real_semantic"], synth_sem_entries,
                           None,
                           f"{cls_name} – Semantic Diversity (CLIP)", "Cosine Distance",
                           is_bottom_row=is_last_row)

    fig.tight_layout(h_pad=3.0)  # slightly more vertical padding for readability
    
    out_path = os.path.join(base_dir, "diversity_histograms.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSuccessfully generated histogram grid: {out_path}")

if __name__ == '__main__':
    main()
