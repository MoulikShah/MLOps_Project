import pytest
import torch
from conftest import get_embedding

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

def test_indian_middle_east(model, pairs, threshold=0.5):
    """Calculate accuracy for a list of pairs."""
    
    subset = pairs["indian_pairs"] + pairs["middle_eastern_pairs"]
    
    accuracy = calc_accuracy(subset, model, threshold)

    assert accuracy >= 0.7, f"Accuracy for Indian/middle eastern faces is {accuracy:.2f}, which is below the threshold of 0.7"

def test_asian(model, pairs, threshold=0.5):
    """Calculate accuracy for a list of pairs."""
    
    subset = pairs["asian_pairs"]
    
    accuracy = calc_accuracy(subset, model, threshold)

    assert accuracy >= 0.7, f"Accuracy for asian faces is {accuracy:.2f}, which is below the threshold of 0.7"

def test_caucasian(model, pairs, threshold=0.5):
    """Calculate accuracy for a list of pairs."""
    
    subset = pairs["white_pairs"]
    
    accuracy = calc_accuracy(subset, model, threshold)

    assert accuracy >= 0.7, f"Accuracy for caucasian faces is {accuracy:.2f}, which is below the threshold of 0.7"

def test_black(model, pairs, threshold=0.5):
    """Calculate accuracy for a list of pairs."""
    
    subset = pairs["black_pairs"]
    
    accuracy = calc_accuracy(subset, model, threshold)

    assert accuracy >= 0.7, f"Accuracy for caucasian faces is {accuracy:.2f}, which is below the threshold of 0.7"
