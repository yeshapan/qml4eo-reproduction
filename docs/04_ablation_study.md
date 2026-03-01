### **QML4EO Ablation Study: Circuit Parameters and Stability**

This doc explores how the hyper-parameters of the quantum circuit impact the overall performance of the Hybrid Quantum Convolutional Neural Network (HQCNN) on the EuroSAT dataset.

#### **What is an Ablation Study?**
In ML; an ablation study is a scientific method applied to neural network architectures. 

Think of it like playing a game of **Jenga** or taking a car engine apart piece by piece: you remove, alter or scale down one specific component at a time to see how much the system relies on it. 

If you remove a layer and the model's accuracy crashes, that layer was critical. If you remove it and the accuracy stays the same, that layer was unnecessary bloat.

#### **Why is this critical for Quantum Machine Learning (QML)?**
In classical ML, adding more parameters is cheap. In QML, it is heavily constrained by current hardware limitations:
1. **Simulation Cost:** Simulating quantum states classically requires exponential memory. Adding just one more qubit doubles the mathematical complexity.
2. **NISQ Era Noise:** On physical quantum hardware (Noisy Intermediate-Scale Quantum), longer circuits introduce more decoherence and noise.

So, we must rigorously prove that every single qubit and quantum operation (gate) we add is actually contributing to the model's learning capacity, rather than just slowing down the training time.

---

#### **1. Qubit Scaling Analysis**
*Testing the performance impact of encoding the classical data into larger quantum states. (Baseline: 4 Qubits)*

| Qubits | Circuit Depth | Total Params | Training Time/Epoch | Final Val Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| 2 | 1 | TBD | TBD | TBD |
| 4 | 1 | TBD | TBD | TBD |
| 8 | 1 | TBD | TBD | TBD |

#### **2. Circuit Depth (Ansatz Layers)**
*Testing the expressivity of the quantum layer by repeating the trainable operations. Repeating the ansatz gives the model more parameters to tune; but increases the risk of vanishing gradients (Barren Plateaus)*

| Qubits | Ansatz Layers | Total Params | Training Time/Epoch | Final Val Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| 4 | 1 | TBD | TBD | TBD |
| 4 | 2 | TBD | TBD | TBD |
| 4 | 3 | TBD | TBD | TBD |

#### **3. Entanglement Strategy**
*Assessing the role of quantum entanglement in feature extraction. Entanglement is what theoretically gives QML an edge over classical models*

| Entanglement Type | Qubits | Depth | Final Val Accuracy | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **None** (No CNOTs) | 4 | 1 | TBD | Acts as a classical linear baseline |
| **Basic** (Ring) | 4 | 1 | TBD | Standard tutorial baseline |
| **Full** (All-to-All) | 4 | 1 | TBD | Highest parameter count |

#### **4. Key Takeaways**
*(To be populated after execution)*
* **Optimal Configuration:** ...
* **Diminishing Returns:** ...