### **QML4EO Structured Report: Baselines and Discrepancies**

This doc is to track the empirical results of replicating the QML4EO baselines

(It provides a direct comparison between classical convolutional architectures and hybrid quantum-classical networks)

#### **Experimental Setup & Hyperparameters**

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Dataset** | EuroSAT (RGB) | 27,000 images, 10 classes |
| **Train/Val Split** | 80% / 20% | Seed: 42, 100, 2026 |
| **Batch Size** | 32 | Standardized across all models |
| **Optimizer** | Adam | Learning Rate: 0.001 |
| **Loss Function** | CrossEntropyLoss | |
| **Hardware** | Google Colab (T4 GPU) | |

---

#### **1. Classical Baseline Metrics (CNN)**

*The classical baseline utilizes a lightweight CNN with Adaptive Average Pooling to remain invariant to the heavy downscaling required by quantum circuits*

| Image Size | Total Parameters | Epochs | Final Training Loss | Final Val Accuracy | Time/Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 64x64 | **5,418** | 15 | **0.59** | **79.15% ± 0.63%** | ~13s |

---

#### **2. Hybrid Quantum-Classical Metrics (HQCNN)**

The baseline utilizes a minimal 4-qubit, 1-layer quantum circuit to establish a lower bound for QML performance and verify gradient flow.*

| Qubits | Quantum Layers | Image Size | Total Parameters | Epochs | Final Val Accuracy (Mean ± Std) | Time/Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4 | 1 | 64x64 | **5,274** | 15 | **66.08% ± 2.42%** | ~25s |

