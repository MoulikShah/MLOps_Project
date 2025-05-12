import pytest
import torch
from conftest import get_embedding, calc_accuracy

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
