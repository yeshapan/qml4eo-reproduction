### **QML4EO Structured Report: Baselines and Discrepancies**

This document tracks the empirical results of replicating and scaling the QML4EO baselines. It provides a direct comparison between state-of-the-art classical convolutional architectures (ResNet) and hybrid quantum-classical networks using a decoupled feature extraction strategy.

#### **Experimental Setup and Hyperparameters**

| Parameter | Value |
| :--- | :--- |
| **Dataset** | EuroSAT (RGB) |
| **Number of Classes** | 10 |
| **Train-Val Split** | 80% - 20% |
| **Seeds** | 42, 100, 2026 |
| **Batch Size** | 64 |
| **Optimizer** | Adam |
| **Loss Function** | CrossEntropyLoss |
| **Hardware** | Google Colab (T4 GPU) |

---

#### **1. Classical Baseline Metrics (Fine-Tuned ResNet18)**

*The V2 classical baseline replaces the original custom lightweight CNN with a pre-trained ResNet18 backbone. Images are upscaled to 224x224 pixels using bilinear interpolation. This establishes a rigorous state-of-the-art accuracy ceiling for our classical baseline.*

| Image Size | Total Parameters | Epochs | Final Training Loss (Mean) | Final Val Accuracy (Mean) | Standard Deviation (±) | Time/Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 224x224 | **11,181,642** | 10 | **0.033** | **98.15%** | **0.31%** | ~56s |

**→ Classical Baseline Seed Variance**
| Seed | Final Training Loss | Final Val Accuracy |
| :---: | :---: | :---: |
| **42** | 0.0327 | 98.37% |
| **100** | 0.0347 | 97.70% |
| **2026** | 0.0324 | 98.37% |

---

#### **2. Hybrid Quantum-Classical Metrics (HQCNN Baseline)**

*The hybrid baseline utilizes the pre-trained ResNet18 as a frozen classical feature extractor, projecting the 512-dimensional output vector into a 4-qubit, 1-layer quantum circuit using a 'ring' entanglement topology. This highlights the empirical behavior of the severe classical-to-quantum information bottleneck.*

| Qubits | Quantum Layers | Entanglement | Total Parameters | Epochs | Final Val Accuracy (Mean) | Standard Deviation (±) | Time/Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4 | 1 | Ring | **~11.18M (512 Trainable)** | 10 | **74.86%** | **4.97%** | ~60s |

