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
| 64x64 | **5,418** | 5 | **0.7809** | **73.94%** | ~15s |

---

#### **2. Hybrid Quantum-Classical Metrics (HQCNN)**

This baseline utilizes a minimal 4-qubit, 1-layer quantum circuit to establish a lower bound for QML performance and verify gradient flow.*

| Qubits | Quantum Layers | Image Size | Total Parameters | Final Val Accuracy | Time/Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 4 | 1 | 64x64 | **5,274** | **47.56%** | ~26s |

**The Parameter Discrepancy: Why is the Quantum Model Smaller?**
It is counter-intuitive that adding a quantum circuit reduces the total parameter count (5,418 classical vs 5,274 hybrid). This occurs due to the **information bottleneck** required to fit classical data into a small quantum state.

* **Classical CNN Final Layers (330 parameters):**
    * The fully connected layer maps 32 channels directly to 10 classes: (32 x 10 weights) + 10 biases = 330 parameters.

* **Hybrid QCNN Final Layers (186 parameters):**
    * **Bottleneck Layer:** Maps 32 channels down to 4 qubits: (32 x 4 weights) + 4 biases = 132 parameters.
    * **Quantum Ansatz:** 1 layer across 4 qubits requires 4 parameters (one trainable rotation per qubit).
    * **Final Classification Layer:** Maps 4 qubits to 10 classes: (4 x 10 weights) + 10 biases = 50 parameters

* **Math:** 330 - 186 = 144 
    Subtracting 144 from the classical total of 5,418 yields exactly 5,274 trainable parameters. The lower accuracy (47.56%) is a direct result of forcing the model to compress its learned features through this tiny 4-qubit bottleneck in only 5 epochs.