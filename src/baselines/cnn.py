import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np

def set_seed(seed=42):
    """
    Locks all random number generators for perfect reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # forces cuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ResNet18Baseline(nn.Module):
    """
    Replaced the previous approach of using custom 2-layer CNN with ResNet18.
    Transfer learning utilizes pre-trained weights from ImageNet.
    This allows the model to immediately recognize complex shapes and textures.
    """
    def __init__(self, num_classes: int = 10, freeze_backbone: bool = False):
        super(ResNet18Baseline, self).__init__()
        
        # Load the pre-trained ResNet18 architecture and weights
        weights = models.ResNet18_Weights.DEFAULT
        self.resnet = models.resnet18(weights=weights)
        
        # Toggle to allow Fine-Tuning. Unfreezing allows the ImageNet filters to adapt to EuroSAT satellite textures.
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
            
        '''
        The default ResNet is built for 1000 ImageNet classes
        We dynamically grab the input size of its final layer (512) and replace it with a new Linear layer targeting our 10 EuroSAT classes
        '''
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def train_baseline(model, train_loader, val_loader, epochs=15, lr=0.0001, device='cpu'):
    '''
    Standard PyTorch training loop for classical architectures
    '''
    criterion = nn.CrossEntropyLoss()   # Loss Function
    # Optimizes all unfrozen parameters. Note the lower default LR for fine-tuning.
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr) 
    
    model.to(device)
    
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
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
        
        print(f"Epoch {epoch+1} Summary → Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
    return history