import pytest
import random
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import yaml
import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from iresnet import iresnet100

@pytest.fixture(scope="session")
def model():
    model = iresnet100(pretrained=False)
    model.load_state_dict(torch.load('model.pth', map_location='cpu'))
    model.eval()
    return model

def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found at {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = np.transpose(img, (2, 0, 1))  # Channels first: (3, 112, 112)
    img = img.astype(np.float32)
    img = (img / 255.0 - 0.5) / 0.5  # Normalize
    img = torch.from_numpy(img)      # <-- CONVERT numpy -> torch.Tensor
    img = img.unsqueeze(0)           # Add batch dimension: (1, 3, 112, 112)
    return img

def get_embedding(img_path, model):
    img_tensor = preprocess(img_path)
    with torch.no_grad():
        emb = model(img_tensor)
        emb = F.normalize(emb, p=2, dim=1)
    return emb

@pytest.fixture(scope="session")
def data():
    with open('MLOps_Project/data_ref/eval_1_analysis.json', 'r') as f:
        data = json.load(f)
    return data

@pytest.fixture(scope="session")
def pairs(data):

    target_ethnicities = ['indian', 'white', 'black', 'middle eastern', 'asian']
    pairs_per_ethnicity = 150
    identities_root = 'mnt/object/dataset/datasets/post_training_opt'

    # Initialize separate lists for different categories
    pairs = []
    indian_pairs = []
    white_pairs = []
    black_pairs = []
    middle_eastern_pairs = []
    asian_pairs = []
    male_pairs = []
    female_pairs = []

    # Prepare: group eligible identities by ethnicity
    eligible = {eth: [] for eth in target_ethnicities}
    for identity, info in data.items():
        eth = info['ethnicity'].lower()
        if eth in eligible:
            if info["gender"]["Man"] > 0.8:
                info["gender"] = "Male"
            elif info["gender"]["Woman"] > 0.8:
                info["gender"] = "Female" 
            else:
                continue       
            eligible[eth].append((identity, info))

    for eth in target_ethnicities:
        random.shuffle(eligible[eth])
        count = 0
        for identity, info in eligible[eth]:
            folder = os.path.join(identities_root, identity)
            images = os.listdir(folder)
            if len(images) < 3:  # Need at least 3 images (1 anchor, 2 positive)
                continue
            
            # Select anchor and two positive images
            anchor_img = random.choice(images)
            remaining_images = [img for img in images if img != anchor_img]
            positive_imgs = random.sample(remaining_images, 2)
            
            # Find two negatives
            candidates = [
                (neg_id, neg_info) for neg_id, neg_info in eligible[eth]
                if neg_id != identity and neg_info['gender'] == info['gender']
            ]
            if len(candidates) < 2:
                continue
                
            negative_identities = random.sample(candidates, 2)
            current_pairs = []
            
            for i in range(2):
                anchor_path = os.path.join(folder, anchor_img)
                positive_path = os.path.join(folder, positive_imgs[i])
                
                neg_identity, _ = negative_identities[i]
                neg_folder = os.path.join(identities_root, neg_identity)
                neg_images = os.listdir(neg_folder)
                if not neg_images:
                    continue
                negative_img = random.choice(neg_images)
                negative_path = os.path.join(neg_folder, negative_img)
                
                pair = (anchor_path, positive_path, negative_path, identity, neg_identity)
                current_pairs.append(pair)
            
            if len(current_pairs) == 2:
                pairs.extend(current_pairs)
                # Add to specific ethnicity lists
                if eth == 'indian':
                    indian_pairs.extend(current_pairs)
                elif eth == 'white':
                    white_pairs.extend(current_pairs)
                elif eth == 'black':
                    black_pairs.extend(current_pairs)
                elif eth == 'middle eastern':
                    middle_eastern_pairs.extend(current_pairs)
                elif eth == 'asian':
                    asian_pairs.extend(current_pairs)
                    
                # Add to gender-specific lists
                if info['gender'] == 'Male':
                    male_pairs.extend(current_pairs)
                else:
                    female_pairs.extend(current_pairs)
                    
                count += 1
                if count >= pairs_per_ethnicity:
                    break

    return {"all_pairs": pairs, "indian_pairs": indian_pairs, "white_pairs": white_pairs,
            "black_pairs": black_pairs, "middle_eastern_pairs": middle_eastern_pairs,
            "asian_pairs": asian_pairs, "male_pairs": male_pairs, "female_pairs": female_pairs}



def calc_accuracy(subset, model, threshold):
    correct = 0
    total = 0
    for pair in subset:
        anchor_path, pos_path, neg_path = pair[0], pair[1], pair[2]
        
        # Get embeddings
        anchor_emb= get_embedding(anchor_path, model)
        pos_emb = get_embedding(pos_path, model)
        neg_emb = get_embedding(neg_path, model)
        
        # Calculate similarities
        pos_sim = torch.nn.functional.cosine_similarity(anchor_emb, pos_emb).item()
        neg_sim = torch.nn.functional.cosine_similarity(anchor_emb, neg_emb).item()
        
        # Check if predictions are correct
        if pos_sim > threshold and neg_sim < threshold:
            correct += 1
        
        total += 1
    
    return correct / total if total > 0 else 0
