import torch
import torch.nn as nn
import pennylane as qml
import torchvision.models as models
import math
from tqdm import tqdm

from src.quantum.qnode import create_qnode

class HybridQCNN(nn.Module):
    """
    Late Hybrid Scheme leveraging a frozen ResNet18 backbone.
    """
    def __init__(self, num_classes=10, num_qubits=4, num_layers=1, entanglement_type="none"):
        super(HybridQCNN, self).__init__()
        
        # Load pre-trained ResNet18
        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)
        
        # Isolate the feature extractor by slicing off the final Linear layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze the classical feature extractor weights to stabilize QML training
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # The Classical Bottleneck: Compress to the exact number of available qubits
        self.bottleneck = nn.Linear(512, num_qubits)
        
        # Pass the string argument to the QNode creator
        qnode = create_qnode(num_qubits, num_layers, entanglement_type=entanglement_type)
        weight_shapes = {"weights": (num_layers, num_qubits)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        
        # Final Classical Classifier
        self.fc = nn.Linear(num_qubits, num_classes)

    def forward(self, x):
        # 1. classical feature extraction
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1) 
        
        # 2. dimensionality reduction
        x = self.bottleneck(x)
        
        # 3. Tanh scaling (bound to [-pi, pi])
        x = torch.tanh(x) * math.pi 
        
        # 4. quantum forward pass
        x = self.qlayer(x)
        
        # 5. final classification
        x = self.fc(x)
        return x

def train_decoupled_hqcnn(model, train_loader, val_loader, epochs=15, device='cpu'):
    """
    Decoupled Learning Rates for Hybrid optimization
    """
    criterion = nn.CrossEntropyLoss()
    
    quantum_params = []
    classical_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'qlayer' in name:
                quantum_params.append(param)
            else:
                classical_params.append(param)
                
    # Independent optimizers: 1e-3 for classical, 1e-4 for quantum
    optimizer = torch.optim.Adam([
        {'params': classical_params, 'lr': 0.001},  
        {'params': quantum_params, 'lr': 0.0001}    
    ])
    
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