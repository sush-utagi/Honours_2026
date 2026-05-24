import os
from pathlib import Path

def get_latest_report(model_dir: Path, split: str = "test"):
    reports = list(model_dir.glob(f"{split}/*report.txt"))
    if not reports:
        return None
    # return the newest report based on filename (they start with a timestamp)
    return sorted(reports)[-1]

def parse_target_class_metrics(filepath: Path, target_classes=["toaster", "hair drier"]):
    metrics = {cls: {"Support": "N/A", "Prec": "N/A", "Rec": "N/A", "F1": "N/A"} for cls in target_classes}
    try:
        with open(filepath, "r") as f:
            for line in f:
                # Example line: " 70 toaster                        24  0.5000  0.0417  0.0769"
                # " 78 hair drier                     21  0.0000  0.0000  0.0000"
                for cls in target_classes:
                    # A robust way to check is if the line contains the class name padded with spaces
                    if f" {cls} " in line or line.strip().endswith(cls) or f" {cls}" in line and len(line.split()) >= 5:
                        parts = line.strip().split()
                        
                        # Handle multi-word classes like 'hair drier'
                        if cls == "hair drier":
                            # parts: ['78', 'hair', 'drier', '21', '0.0000', '0.0000', '0.0000']
                            if parts[1] == "hair" and parts[2] == "drier":
                                metrics[cls]["Support"] = parts[-4]
                                metrics[cls]["Prec"] = parts[-3]
                                metrics[cls]["Rec"] = parts[-2]
                                metrics[cls]["F1"] = parts[-1]
                        elif cls == "toaster":
                            # parts: ['70', 'toaster', '24', '0.5000', '0.0417', '0.0769']
                            if parts[1] == "toaster":
                                metrics[cls]["Support"] = parts[-4]
                                metrics[cls]["Prec"] = parts[-3]
                                metrics[cls]["Rec"] = parts[-2]
                                metrics[cls]["F1"] = parts[-1]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return metrics

def main():
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    
    models = {
        "baseline_model_A": "Baseline (Real Only)",
        "baseline_model_B": "Baseline (Classical Aug)",
        "experimental_model_A": "Experimental (IP-Adapter)",
        "experimental_model_B": "Experimental (ControlNet)",
    }
    
    target_classes = ["toaster", "hair drier"]
    
    print(f"{'Model':<30} | {'Class':<12} | {'Support':<8} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 92)
    
    for model_id, label in models.items():
        model_dir = results_dir / model_id
        if not model_dir.exists():
            print(f"{label:<30} | Directory not found")
            continue
            
        report_path = get_latest_report(model_dir, split="test")
        if not report_path:
            print(f"{label:<30} | Test report not found")
            continue
            
        metrics = parse_target_class_metrics(report_path, target_classes)
        
        for cls in target_classes:
            cls_metrics = metrics[cls]
            # Print model label only on the first row for neatness, or on all rows
            model_display = label if cls == target_classes[0] else ""
            
            print(f"{model_display:<30} | {cls:<12} | {cls_metrics['Support']:<8} | {cls_metrics['Prec']:<10} | {cls_metrics['Rec']:<10} | {cls_metrics['F1']:<10}")
        print("-" * 92)

if __name__ == "__main__":
    main()