import matplotlib.pyplot as plt
import re
import os

def parse_report(report_path):
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Per-class metrics (sorted by F1 desc):
    # Idx Class                     Support    Prec     Rec      F1
    #  22 zebra                         549  0.9645  0.9399  0.9520
    
    pattern = r'^\s*(\d+)\s+([a-z ]+?)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    matches = re.findall(pattern, content, re.MULTILINE)
    
    data = []
    for m in matches:
        data.append({
            'idx': int(m[0]),
            'class': m[1].strip(),
            'support': int(m[2]),
            'precision': float(m[3]),
            'recall': float(m[4]),
            'f1': float(m[5])
        })
    return data

def plot_f1_scores(data, output_path):
    # Take only the bottom 5 classes
    data.sort(key=lambda x: x['f1'], reverse=True)
    bottom_5 = data[-5:]
    
    classes = [d['class'] for d in bottom_5]
    f1_scores = [d['f1'] for d in bottom_5]
    
    # Custom colors
    color_main = '#673AB7'  # Purple
    color_highlight = '#00BFA5' # Teal
    
    plt.figure(figsize=(10, 6), dpi=300)
    
    colors = []
    for d in bottom_5:
        if d['class'] in ['toaster', 'hair drier']:
            colors.append(color_highlight)
        else:
            colors.append(color_main)
            
    bars = plt.bar(range(len(classes)), f1_scores, color=colors, alpha=0.8, width=0.6)
    
    plt.xticks(range(len(classes)), classes, rotation=0, fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontweight='bold', fontsize=12)
    plt.title('Baseline Model: Lowest Performing Classes (F1 Score)', fontweight='bold', fontsize=14)
    plt.ylim(0, 0.6) # Provide some headroom for labels
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Highlight the target classes with labels and dotted lines
    for i, d in enumerate(bottom_5):
        if d['class'] in ['toaster', 'hair drier']:
            label_y = d['f1'] + 0.15 # Adjusted for smaller scale
            plt.text(i, label_y + 0.02, d['class'], ha='center', va='bottom', 
                     color=color_highlight, fontweight='bold', fontsize=14)
            # Draw dotted line from top of bar to label
            plt.plot([i, i], [d['f1'], label_y], color=color_highlight, linestyle=':', linewidth=2)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    report_file = "/Users/susheelutagi/Documents/GitHub/Honours_2026/results/baseline_model_A/test/20260426_151826_baseline_report.txt"
    output_file = "/Users/susheelutagi/Documents/GitHub/Honours_2026/experiments/figures/baseline_per_class_f1.png"
    
    if os.path.exists(report_file):
        data = parse_report(report_file)
        plot_f1_scores(data, output_file)
    else:
        print(f"Report file not found: {report_file}")
