import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np

def set_seed(seed=42):
    """
    locks all random number generators for perfect reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #forces cuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ClassicalCNN(nn.Module):
    """
    A lightweight classical CNN baseline.
    Uses Adaptive Average Pooling to remain invariant to input image resolution,
    allowing seamless comparison when images are heavily downscaled for quantum circuits.
    """
    def __init__(self, num_classes: int = 10):
        super(ClassicalCNN, self).__init__()
        # Input channels: 3 (RGB), Output channels: 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # Input channels: 16, Output channels: 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        #adaptive pool ensures the output is always 1x1 per channel before the fully connected layer 
        #(to prevent dimension mismatch errors)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(-1, 32) #flatten
        x = self.fc(x)
        return x

def train_baseline(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    """
    Standard PyTorch training loop.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        #training loop with progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        #validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1} Summary -> Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
    return history