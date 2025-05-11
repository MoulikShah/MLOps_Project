import os
import json
import random

def sample_classes(training_folder, num_classes=10000):
    # Get all class folders
    class_folders = [d for d in os.listdir(training_folder) 
                    if os.path.isdir(os.path.join(training_folder, d))]
    
    # Check if we have enough classes
    total_classes = len(class_folders)
    if total_classes < num_classes:
        raise ValueError(f"Only {total_classes} classes available, cannot sample {num_classes}")
    
    # Randomly sample classes
    sampled_classes = random.sample(class_folders, num_classes)
    
    # Save to JSON file
    output_file = "/home/cc/MLOps_Project/training_scripts/arcface_torch/sampled_classes.json"
    with open(output_file, 'w') as f:
        json.dump({
            "total_classes": total_classes,
            "sampled_classes": sampled_classes,
            "num_sampled": len(sampled_classes)
        }, f, indent=4)
    
    print(f"Sampled {num_classes} classes and saved to {output_file}")
    return sampled_classes

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Replace this with your training folder path
    training_folder = "/mnt/object/dataset/datasets/train"
    
    try:
        sampled_classes = sample_classes(training_folder)
    except Exception as e:
        print(f"Error: {e}")