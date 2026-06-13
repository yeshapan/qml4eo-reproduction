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
    Fixed: Supports loading fine-tuned, domain-specific weights and cleans the state_dict keys to resolve prefix mismatches.
    """
    def __init__(self, num_classes=10, num_qubits=4, num_layers=1, entanglement_type="none", pretrained_weights_path=None):
        super(HybridQCNN, self).__init__()
        
        # 1. Load base ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 2. Replace the head to match EuroSAT classes 
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, num_classes)
        
        # 3. Load our custom, fine-tuned EuroSAT weights if provided
        if pretrained_weights_path:
            raw_state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            
            # Intercept and clean the dictionary keys to remove the 'resnet.' wrapper prefix
            cleaned_state_dict = {}
            for key, value in raw_state_dict.items():
                if key.startswith('resnet.'):
                    cleaned_key = key.replace('resnet.', '', 1)
                    cleaned_state_dict[cleaned_key] = value
                else:
                    cleaned_state_dict[key] = value
                    
            resnet.load_state_dict(cleaned_state_dict)
            print(f"Loaded fine-tuned classical weights from: {pretrained_weights_path}")
            
        # 4. Isolate the feature extractor by slicing off the final Linear layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # 5. STRICTLY FREEZE the feature extractor to prevent Classical Masking
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
        # 1. Classical Feature Extraction
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1) 
        
        # 2. Dimensionality Reduction
        x = self.bottleneck(x)
        
        # 3. Tanh Scaling (Bound to [-pi, pi])
        x = torch.tanh(x) * math.pi 
        
        # 4. Quantum Forward Pass
        x = self.qlayer(x)
        
        # 5. Final Classification
        x = self.fc(x)
        return x

def train_decoupled_hqcnn(model, train_loader, val_loader, epochs=15, device='cpu'):
    """
    Decoupled Learning Rates for Hybrid optimization.
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
                
    # Independent optimizers: 1e-3 for classical bottleneck, 1e-4 for sensitive quantum weights
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