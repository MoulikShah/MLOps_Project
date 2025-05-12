from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import io
import logging
import uvicorn
from typing import List
import sys
import os
from pathlib import Path

from backbones.iresnet import iresnet100  # Adjust import path as needed

app = FastAPI(title="Face Similarity API",
              description="API for checking if two face images belong to the same person",
              version="1.0.0")

# Global model variable
MODEL_PATH = "model.pth"

# Try loading a state_dict first, otherwise assume the file *is* the model
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
if isinstance(checkpoint, dict):
    model = iresnet100(pretrained=False)
    model.load_state_dict(checkpoint)
else:
    model = checkpoint

model.eval()


def preprocess_image(image_bytes):
    """Preprocess the image from bytes"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to required dimensions
        img = cv2.resize(img, (112, 112))
        
        # Transpose to channel-first format
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        
        # Convert to float and normalize
        img = img.astype(np.float32)
        img = (img / 255.0 - 0.5) / 0.5
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.unsqueeze(0)  # (1, 3, 112, 112)
        
        return img_tensor
    except Exception as e:
        raise

def get_embedding(img_tensor):
    """Get embedding from preprocessed image tensor"""        
    with torch.no_grad():
        emb = model(img_tensor)
        emb = F.normalize(emb, p=2, dim=1)
    return emb

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Face Similarity API is running", "status": "ok"}

@app.post("/compare")
async def compare_faces(image1: UploadFile = File(...), image2: UploadFile = File(...), threshold: float = 0.5):
    """
    Compare two face images and determine if they belong to the same person
    
    Parameters:
    - image1: First face image
    - image2: Second face image
    - threshold: Similarity threshold (default: 0.5)
    
    Returns:
    - JSON with comparison results
    """
    try:
        # Read images
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()
        
        # Preprocess images
        img1_tensor = preprocess_image(img1_bytes)
        img2_tensor = preprocess_image(img2_bytes)
        
        # Get embeddings
        emb1 = get_embedding(img1_tensor)
        emb2 = get_embedding(img2_tensor)
        
        # Calculate similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
        
        # Determine if same person
        is_same_person = similarity > threshold
        
        return {
            "is_same_person": bool(is_same_person),
            "similarity_score": float(similarity),
            "threshold": float(threshold)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
