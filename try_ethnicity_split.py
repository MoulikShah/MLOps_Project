import os
import shutil
from deepface import DeepFace
import cv2

# Input and output directories
input_dir = "path_to_dataset"  # Replace with the path to your dataset
output_dir = "sorted_by_race"
os.makedirs(output_dir, exist_ok=True)

# Iterate through each identity folder
for identity_folder in os.listdir(input_dir):
    identity_path = os.path.join(input_dir, identity_folder)
    if not os.path.isdir(identity_path):
        continue

    # Get the first image in the folder
    images = [img for img in os.listdir(identity_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        continue

    first_image_path = os.path.join(identity_path, images[0])

    try:
        # Read the first image
        img = cv2.imread(first_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Analyze the race using DeepFace
        result = DeepFace.analyze(img, actions=['race'], enforce_detection=False)
        dominant_race = result[0]['dominant_race']

        # Create a directory for the race if it doesn't exist
        race_dir = os.path.join(output_dir, dominant_race)
        os.makedirs(race_dir, exist_ok=True)

        # Create a directory for the identity within the race folder
        identity_race_dir = os.path.join(race_dir, identity_folder)
        os.makedirs(identity_race_dir, exist_ok=True)

        # Copy all images from the identity folder to the new race-based folder
        for img_file in images:
            src_path = os.path.join(identity_path, img_file)
            dst_path = os.path.join(identity_race_dir, img_file)
            shutil.copy(src_path, dst_path)

    except Exception as e:
        print(f"Error processing folder {identity_folder}: {str(e)}")

print(f"Images sorted by race and saved in: {output_dir}")
