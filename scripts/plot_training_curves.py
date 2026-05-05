import os
import re
import matplotlib.pyplot as plt
import glob

def parse_log_file(filepath):
    """Parses a training log file and returns a dictionary of metrics."""
    data = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Regex for parsing the epoch lines
    # Format: Epoch 1/8: train_loss=2.4177 val_loss=1.8849 train_acc=0.372 val_acc=0.486
    epoch_re = re.compile(r"Epoch (\d+)/\d+: train_loss=([\d\.]+) val_loss=([\d\.]+) train_acc=([\d\.]+) val_acc=([\d\.]+)")
    
    with open(filepath, 'r') as f:
        for line in f:
            match = epoch_re.search(line)
            if match:
                epoch = int(match.group(1))
                t_loss = float(match.group(2))
                v_loss = float(match.group(3))
                t_acc = float(match.group(4))
                v_acc = float(match.group(5))
                
                # Handle resumes by keeping the latest data for each epoch index
                # (Assuming epochs are logged sequentially)
                if epoch in data['epochs']:
                    idx = data['epochs'].index(epoch)
                    data['train_loss'][idx] = t_loss
                    data['val_loss'][idx] = v_loss
                    data['train_acc'][idx] = t_acc
                    data['val_acc'][idx] = v_acc
                else:
                    data['epochs'].append(epoch)
                    data['train_loss'].append(t_loss)
                    data['val_loss'].append(v_loss)
                    data['train_acc'].append(t_acc)
                    data['val_acc'].append(v_acc)
    
    # Sort by epoch just in case
    combined = sorted(zip(data['epochs'], data['train_loss'], data['val_loss'], data['train_acc'], data['val_acc']))
    if not combined:
        return None
        
    data['epochs'], data['train_loss'], data['val_loss'], data['train_acc'], data['val_acc'] = zip(*combined)
    return data

def plot_metrics(all_data, metric_type, title, ylabel, out_path):
    """Generates and saves a plot for a specific metric (loss or acc)."""
    plt.figure(figsize=(12, 8))
    
    # Project Standard Colors
    color_map = {
        "Baseline A": "#2196F3",      # Blue
        "Baseline B": "#00ACC1",      # Teal
        "Experimental A": "#E53935",  # Red
        "Experimental B": "#FF9800"   # Orange
    }
    
    for name, data in all_data.items():
        color = color_map.get(name, '#607D8B') # Default grey if name not found
        
        train_key = f'train_{metric_type}'
        val_key = f'val_{metric_type}'
        
        # Plot training (solid)
        plt.plot(data['epochs'], data[train_key], label=f'{name} (Train)', 
                 color=color, linestyle='-', linewidth=2, alpha=0.45)
        
        # Plot validation (dashed)
        plt.plot(data['epochs'], data[val_key], label=f'{name} (Val)', 
                 color=color, linestyle='--', linewidth=2.5)
        
        # Mark the best/last point
        best_idx = 0
        if metric_type == 'acc':
            best_val = max(data[val_key])
            best_idx = list(data[val_key]).index(best_val)
        else:
            best_val = min(data[val_key])
            best_idx = list(data[val_key]).index(best_val)
            
        plt.scatter(data['epochs'][best_idx], data[val_key][best_idx], color=color, s=100, edgecolors='white', zorder=5)

    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {metric_type} plot to {out_path}")

def main():
    log_dir = "logs"
    out_dir = "experiments/figures"
    os.makedirs(out_dir, exist_ok=True)
    
    log_files = glob.glob(os.path.join(log_dir, "*.txt"))
    if not log_files:
        print("No log files found in logs/ directory.")
        return
        
    all_data = {}
    for filepath in sorted(log_files):
        # Extract model name from filename (e.g., log_baseline_A.txt -> baseline_A)
        name = os.path.basename(filepath).replace('log_', '').replace('.txt', '').replace('_', ' ').title()
        data = parse_log_file(filepath)
        if data:
            all_data[name] = data

    if not all_data:
        print("Could not parse any training data from logs.")
        return

    # Plot Loss
    plot_metrics(all_data, 'loss', 
                 'Training and Validation Loss', 
                 'Cross-Entropy Loss', 
                 os.path.join(out_dir, "training_loss_curves.png"))
    
    # Plot Accuracy
    plot_metrics(all_data, 'acc', 
                 'Training and Validation Accuracy', 
                 'Top-1 Accuracy', 
                 os.path.join(out_dir, "training_accuracy_curves.png"))

if __name__ == "__main__":
    main()
