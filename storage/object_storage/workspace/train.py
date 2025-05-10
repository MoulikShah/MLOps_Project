#!/usr/bin/env python
# Example training script that loads data from object storage

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time

print("PyTorch version:", torch.__version__)

# Get the data directory from environment variable, or use default
DATA_DIR = os.environ.get('DATA_DIR', '/data')
print(f"Loading data from: {DATA_DIR}")

# Check if the data directory exists and has content
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist. Make sure object storage is mounted correctly.")

data_dirs = os.listdir(DATA_DIR)
print(f"Contents of data directory: {data_dirs}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_model(model_name="resnet18", batch_size=32, num_epochs=5):
    """
    Train a model using data from object storage
    """
    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Check for train and test directories
    train_dir = os.path.join(DATA_DIR, 'train')
    test_dir = os.path.join(DATA_DIR, 'test')
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"Train or test directory not found in {DATA_DIR}. Please check your data structure.")
    
    # Load datasets
    print("Loading datasets...")
    try:
        image_datasets = {
            'train': datasets.ImageFolder(train_dir, data_transforms['train']),
            'test': datasets.ImageFolder(test_dir, data_transforms['test'])
        }
        
        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
            'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)
        }
        
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
        class_names = image_datasets['train'].classes
        
        print(f"Dataset loaded with {dataset_sizes['train']} training images and {dataset_sizes['test']} test images")
        print(f"Classes: {class_names}")
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise
    
    # Initialize model
    print(f"Initializing {model_name} model...")
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Train the model
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass - track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    
    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    
    # Save the model
    model_save_path = os.path.join('models', f"{model_name}_trained.pth")
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model

if __name__ == "__main__":
    print("Starting training script using data from object storage")
    try:
        train_model(model_name="resnet18", batch_size=32, num_epochs=2)
    except Exception as e:
        print(f"Error during training: {e}")
    print("Training script completed") 