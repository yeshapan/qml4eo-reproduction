### **QML4EO Structured Report: Baselines and Discrepancies**

This doc is to track the empirical results of replicating the QML4EO baselines

(It provides a direct comparison between classical convolutional architectures and hybrid quantum-classical networks)

#### **Experimental Setup & Hyperparameters**

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Dataset** | EuroSAT (RGB) | 27,000 images, 10 classes |
| **Train/Val Split** | 80% / 20% | Seed: 42 |
| **Batch Size** | 32 | Standardized across all models |
| **Optimizer** | Adam | Learning Rate: 0.001 |
| **Loss Function** | CrossEntropyLoss | |
| **Hardware** | Google Colab (T4 GPU) | |

---

#### **1. Classical Baseline Metrics (CNN)**

*The classical baseline utilizes a lightweight CNN with Adaptive Average Pooling to remain invariant to the heavy downscaling required by quantum circuits*

| Image Size | Total Parameters | Epochs | Final Training Loss | Final Val Accuracy | Time/Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 64x64 | **5,418** | 5 | **0.7445** | **74.31%** | ~14s |

---

#### **2. Hybrid Quantum-Classical Metrics (HQCNN)**

*Pending execution of PennyLane/Qiskit circuits*

| Qubits | Quantum Layers | Image Size | Total Parameters | Final Val Accuracy | Sim-to-Real Discrepancy |
| :---: | :---: | :---: | :---: | :---: | :---: |
| TBD | TBD | TBD | TBD | TBD | TBD |