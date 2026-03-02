import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import math

#import our custom qnode builder
from src.quantum.qnode import create_qnode

class HybridQCNN(nn.Module):
    """
    hybrid quantum-classical convolutional neural network.
    extracts features using classical conv layers, compresses them to match 
    the number of qubits, passes them through a quantum circuit, and 
    classifies the output.
    """
    def __init__(self, num_classes=10, num_qubits=4, num_layers=1):
        super(HybridQCNN, self).__init__()
        
        #classical feature extractor (same as our baseline)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        #bottleneck layer: compress 32 classical channels down to exactly the number of qubits
        self.bottleneck = nn.Linear(32, num_qubits)
        
        #quantum layer
        qnode = create_qnode(num_qubits, num_layers)
        
        #pennylane requires a dictionary specifying the shape of the trainable weights
        weight_shapes = {"weights": (num_layers, num_qubits)}
        
        #orchLayer perfectly wraps the quantum node so pytorch treats it like a normal layer
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        
        #final classical classification layer
        self.fc = nn.Linear(num_qubits, num_classes)

    def forward(self, x):
        #classical feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(-1, 32) # flatten
        
        #compress to fit into our qubits
        x = self.bottleneck(x)
        
        #scale the classical values to be between -pi and pi for angle encoding.
        #we use tanh to squish values between -1 and 1, then multiply by pi.
        x = torch.tanh(x) * math.pi 
        
        #quantum forward pass (this is where the simulation happens)
        x = self.qlayer(x)
        
        #final classical pass to get the 10 class logits
        x = self.fc(x)
        return x