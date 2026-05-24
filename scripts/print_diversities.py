import json
import os

def main():
    base_dir = "results/clip_analysis"
    classes = ["hair_drier", "toaster"]
    methods = ["cn", "ip"]
    
    print(f"{'Class':<12} | {'Method':<6} | {'Set':<9} | {'Semantic Diversity (CLIP)':<35} | {'Structural Diversity (LPIPS)':<35}")
    print("-" * 110)
    
    for class_name in classes:
        for method in methods:
            file_path = os.path.join(base_dir, class_name, f"{class_name}_{method}", "metrics_summary.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    
                    # Extract real values for comparison
                    real_data = data.get("real", {})
                    real_sem_mean = real_data.get("semantic_diversity_clip", {}).get("mean", "N/A")
                    real_struct_mean = real_data.get("structural_diversity_lpips", {}).get("mean", "N/A")
                    
                    for set_type in ["real", "synthetic"]:
                        set_data = data.get(set_type, {})
                        
                        sem_div_data = set_data.get("semantic_diversity_clip", {})
                        struct_div_data = set_data.get("structural_diversity_lpips", {})
                        
                        sem_mean = sem_div_data.get("mean", "N/A")
                        struct_mean = struct_div_data.get("mean", "N/A")
                        
                        # Format means if they are numbers
                        sem_str = f"{sem_mean:.4f}" if isinstance(sem_mean, (float, int)) else str(sem_mean)
                        struct_str = f"{struct_mean:.4f}" if isinstance(struct_mean, (float, int)) else str(struct_mean)
                        
                        # Add difference if synthetic
                        if set_type == "synthetic":
                            if isinstance(sem_mean, (float, int)) and isinstance(real_sem_mean, (float, int)):
                                sem_diff = sem_mean - real_sem_mean
                                sem_str += f" ({sem_diff:+.4f})"
                            if isinstance(struct_mean, (float, int)) and isinstance(real_struct_mean, (float, int)):
                                struct_diff = struct_mean - real_struct_mean
                                struct_str += f" ({struct_diff:+.4f})"
                        
                        print(f"{class_name:<12} | {method:<6} | {set_type:<9} | {sem_str:<35} | {struct_str:<35}")
                except Exception as e:
                    print(f"{class_name:<12} | {method:<6} | Error reading file: {e}")
            else:
                print(f"{class_name:<12} | {method:<6} | File not found")

if __name__ == "__main__":
    main()