import json
import os

def main():
    base_dir = "results/clip_analysis"
    classes = ["hair_drier", "toaster"]
    methods = ["cn", "ip"]
    
    print(f"{'Class':<12} | {'Method':<6} | {'Set':<9} | {'Pixel Memorization Ratio':<28} | {'CLIP Memorization Ratio':<28}")
    print("-" * 95)
    
    for class_name in classes:
        for method in methods:
            file_path = os.path.join(base_dir, class_name, f"{class_name}_{method}", "metrics_summary.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    
                    # Extract real values for comparison
                    real_data = data.get("real", {})
                    real_mem_ratios = real_data.get("memorization_ratio", {})
                    real_pixel_mean = real_mem_ratios.get("pixel", {}).get("mean_ratio", "N/A")
                    real_clip_mean = real_mem_ratios.get("clip", {}).get("mean_ratio", "N/A")
                    
                    for set_type in ["real", "synthetic"]:
                        set_data = data.get(set_type, {})
                        mem_ratios = set_data.get("memorization_ratio", {})
                        
                        pixel_mean = mem_ratios.get("pixel", {}).get("mean_ratio", "N/A")
                        clip_mean = mem_ratios.get("clip", {}).get("mean_ratio", "N/A")
                        
                        # Format means if they are numbers
                        pixel_str = f"{pixel_mean:.4f}" if isinstance(pixel_mean, (float, int)) else str(pixel_mean)
                        clip_str = f"{clip_mean:.4f}" if isinstance(clip_mean, (float, int)) else str(clip_mean)
                        
                        # Add difference if synthetic
                        if set_type == "synthetic":
                            if isinstance(pixel_mean, (float, int)) and isinstance(real_pixel_mean, (float, int)):
                                pixel_diff = pixel_mean - real_pixel_mean
                                pixel_str += f" ({pixel_diff:+.4f})"
                            if isinstance(clip_mean, (float, int)) and isinstance(real_clip_mean, (float, int)):
                                clip_diff = clip_mean - real_clip_mean
                                clip_str += f" ({clip_diff:+.4f})"
                        
                        print(f"{class_name:<12} | {method:<6} | {set_type:<9} | {pixel_str:<28} | {clip_str:<28}")
                except Exception as e:
                    print(f"{class_name:<12} | {method:<6} | Error reading file: {e}")
            else:
                print(f"{class_name:<12} | {method:<6} | File not found")

if __name__ == "__main__":
    main()