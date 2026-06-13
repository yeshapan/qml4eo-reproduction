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
    Supports loading fine-tuned, domain-specific weights and cleans the state_dict.
    Implements a CPU-Quantum Bridge to resolve PyTorch/PennyLane device mismatches.
    """
    def __init__(self, num_classes=10, num_qubits=4, num_layers=1, entanglement_type="none", pretrained_weights_path=None):
        super(HybridQCNN, self).__init__()
        
        # 1. Load base ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 2. Replace the head to match EuroSAT classes
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, num_classes)
        
        # 3. Load custom, fine-tuned EuroSAT weights if provided
        if pretrained_weights_path:
            raw_state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            
            cleaned_state_dict = {}
            for key, value in raw_state_dict.items():
                if key.startswith('resnet.'):
                    cleaned_key = key.replace('resnet.', '', 1)
                    cleaned_state_dict[cleaned_key] = value
                else:
                    cleaned_state_dict[key] = value
                    
            resnet.load_state_dict(cleaned_state_dict)
            
        # 4. Isolate the feature extractor
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # 5. STRICTLY FREEZE the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # 6. The Classical Bottleneck
        self.bottleneck = nn.Linear(512, num_qubits)
        
        # 7. Quantum Layer
        qnode = create_qnode(num_qubits, num_layers, entanglement_type=entanglement_type)
        weight_shapes = {"weights": (num_layers, num_qubits)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        
        # 8. Final Classical Classifier
        self.fc = nn.Linear(num_qubits, num_classes)

    def forward(self, x):
        # Classical Feature Extraction (GPU)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1) 
        
        # Dimensionality Reduction (GPU)
        x = self.bottleneck(x)
        x = torch.tanh(x) * math.pi 
        
        # CPU-QUANTUM BRIDGE
        # PennyLane's default simulator initializes states on the CPU
        # We temporarily cast the data to CPU, execute the circuit and then move back to GPU
        current_device = x.device
        x = x.cpu()
        
        # Quantum Forward Pass (CPU)
        x = self.qlayer(x)
        
        # Move back to GPU
        x = x.to(current_device)
        
        # Final Classification (GPU)
        x = self.fc(x)
        return x

def train_decoupled_hqcnn(model, train_loader, val_loader, epochs=15, device='cpu'):
    """
    Decoupled Learning Rates for Hybrid optimization.
    """
    criterion = nn.CrossEntropyLoss()
    
    # 1. Device Placement Strategy MUST happen before optimizer initialization
    model.to(device)
    # Force the quantum layer to stay on the CPU to prevent mismatch with PennyLane's state vector
    model.qlayer.to('cpu')
    
    # 2. Gather Parameters
    quantum_params = []
    classical_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'qlayer' in name:
                quantum_params.append(param)
            else:
                classical_params.append(param)
                
    # 3. Independent optimizers
    optimizer = torch.optim.Adam([
        {'params': classical_params, 'lr': 0.001},  
        {'params': quantum_params, 'lr': 0.0001}    
    ])
    
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