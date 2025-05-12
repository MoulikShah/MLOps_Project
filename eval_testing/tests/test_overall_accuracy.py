import pytest
import torch
from conftest import get_embedding

def test_overall_accuracy(model, pairs, threshold=0.5):
    """Calculate accuracy for a list of pairs."""
    correct = 0
    total = 0
    
    all_pairs = pairs["all_pairs"]
    for pair in all_pairs:
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
    
    accuracy = correct / total if total > 0 else 0

    assert accuracy >= 0.8, f"Overall accuracy is {accuracy:.2f}, which is below the threshold of 0.8"
