import os
import json
from deepface import DeepFace
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Input directory and output JSON file path
input_dir = "/home/aryan/FSCIL/datasets/ms1m"  # Replace with the path to your dataset
output_json = "identity_race_mapping.json"

# Dictionary to store identity to race mapping
identity_race_map = {}
race_count = {}

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

        # Add to dictionary
        identity_race_map[identity_folder] = dominant_race
        if dominant_race in race_count:
            race_count[dominant_race] += 1
        else:
            race_count[dominant_race] = 0


    except Exception as e:
        print(f"Error processing folder {identity_folder}: {str(e)}")

# Save the dictionary to a JSON file
with open(output_json, 'w') as f:
    json.dump(identity_race_map, f, indent=4)

print(f"Race classification mapping saved in: {output_json}")

for each in race_count:
    print(each, race_count[each])