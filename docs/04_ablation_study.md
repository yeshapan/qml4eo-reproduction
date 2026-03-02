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

> NOTE: Please refer to `notebooks/03_ablation_study.ipynb` too see results of each experiment (and their analysis)

#### **1. Qubit Scaling Analysis**
*Testing the performance impact of encoding the classical data into larger quantum states. (Baseline: 4 Qubits)*

| Qubits | Circuit Depth | Total Params | Training Time/Epoch | Final Val Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| 2 | 1 | 5,186 | ~19s | 42.46% |
| 4 | 1 | 5,274 | ~26s | 47.56% |
| 6 | 1 | 5,362 | ~31s | 46.70% |
| 8 | 1 | 5,450 | ~38s | 48.02% |

**Analysis:**
* Restricting the bottleneck to 2 qubits physically destroys too much feature data, resulting in the lowest accuracy. 
* Scaling past 4 qubits offers diminishing returns within a short 5-epoch window. The optimizer simply needs more training time to navigate larger quantum spaces.
* Simulation cost scales aggressively. Simulating 8 qubits nearly doubles the training time per epoch compared to 2 qubits.

#### **2. Circuit Depth (Ansatz Layers)**
*Testing the expressivity of the quantum layer by repeating the trainable operations. Repeating the ansatz gives the model more parameters to tune; but it also increases the risk of vanishing gradients (Barren Plateaus)*

| Qubits | Ansatz Layers | Total Params | Training Time/Epoch | Final Val Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| 4 | 1 | 5,274 | ~26s | 47.56% |
| 4 | 2 | 5,278 | ~30s | 48.69% |
| 4 | 3 | 5,282 | ~35s | 63.69% |

**Analysis:**
* Adding just 8 total parameters (jumping to 3 layers) boosted final accuracy by over 16%. Quantum expressivity relies heavily on depth rather than raw parameter count.
* The 2-layer model exhibited high instability, losing validation accuracy in the final epoch despite a dropping training loss. This proves quantum loss landscapes are highly non-convex and prone to overshooting.
* Training time increases linearly. Each new ansatz layer consistently adds 4 to 5 seconds per epoch on the GPU.

#### **3. Entanglement Strategy**
*Assessing the role of quantum entanglement in feature extraction. Entanglement is what theoretically gives QML an edge over classical models*

| Entanglement Type | Qubits | Depth | Final Val Accuracy | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **None** (No CNOTs) | 4 | 1 | 68.22% | Acts as a classical linear baseline |
| **Basic** (Ring) | 4 | 1 | 48.78% | Standard tutorial baseline |
| **Full** (All-to-All) | 4 | 1 | TBD | Skipped to conserve compute resources |

**Analysis:**
* Removing entanglement completely improved short-term accuracy by ~20%. 
* Unentangled circuits possess smoother loss landscapes and train faster. 
* Entangled circuits introduce landscape complexity. They train slower but offer higher expressivity ceilings when scaled in depth.

#### **4. Key Takeaways**
* **Optimal Configuration (Resource Constrained):** For a strict 5-epoch limit; a shallow, unentangled circuit performs best. For maximum expressivity; a 4-qubit, 3-layer entangled circuit provides the best balance of parameter efficiency and accuracy.
* **The Information Bottleneck:** Qubit count must be carefully balanced. 2 qubits physically destroy too much feature data. 8 qubits mathematically overwhelm classical GPU simulators.
* **Diminishing Returns:** Adding quantum operations without scaling training epochs leads to underfitting. The optimizer simply needs more time to navigate the complex Hilbert space.