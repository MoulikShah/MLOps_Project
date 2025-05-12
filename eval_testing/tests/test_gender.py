import pytest
import torch
from conftest import get_embedding, calc_accuracy

def test_male(model, pairs, threshold=0.5):
    """Calculate accuracy for a list of pairs."""
    
    subset = pairs["male_pairs"]
    
    accuracy = calc_accuracy(subset, model, threshold)

    assert accuracy >= 0.7, f"Accuracy for male faces is {accuracy:.2f}, which is below the threshold of 0.7"

def test_female(model, pairs, threshold=0.5):
    """Calculate accuracy for a list of pairs."""
    
    subset = pairs["female_pairs"]
    
    accuracy = calc_accuracy(subset, model, threshold)

    assert accuracy >= 0.7, f"Accuracy for female faces is {accuracy:.2f}, which is below the threshold of 0.7"
