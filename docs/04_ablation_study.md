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

> NOTE: Please refer to `notebooks/03_ablation.ipynb` too see results of each experiment (and their analysis)

#### **1. Qubit Scaling Analysis**
*Testing the performance impact of encoding the classical data into larger quantum states. (Baseline: 4 Qubits)*

*Evaluated across 3 seeds for 15 epochs. Assesses the tradeoff between quantum state size and classical simulation cost.*

| Qubits | Ansatz Layers | Total Parameters | Final Mean Accuracy | Std Dev | Speed (it/s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 1 | 5,186 | **49.56%** | ± 1.62% | ~33 |
| 4 | 1 | 5,274 | **66.08%** | ± 2.42% | ~26 |
| 6 | 1 | 5,362 | **70.25%** | ± 1.75% | ~21 |
| 8 | 1 | 5,450 | **69.83%** | ± 2.78% | ~16 |

**Key Observation:** Performance plateaus after 6 qubits. The 8-qubit space is too vast for a 1-layer ansatz to navigate effectively within 15 epochs, resulting in identical accuracy but severe simulation penalties.

#### **2. Circuit Depth (Ansatz Layers)**
*Testing the expressivity of the quantum layer by repeating the trainable operations. Repeating the ansatz gives the model more parameters to tune; but it also increases the risk of vanishing gradients (Barren Plateaus)*

*Evaluated across 3 seeds for 15 epochs. Assesses whether deeper circuits yield better expressivity or trigger Barren Plateaus. Bottleneck locked at 4 qubits.*

| Ansatz Layers | Total Parameters | Final Mean Accuracy | Std Dev | Speed (it/s) |
| :---: | :---: | :---: | :---: | :---: |
| 1 (Baseline) | 5,274 | **66.08%** | ± 2.42% | ~26 |
| 2 | 5,278 | **69.69%** | ± 4.03% | ~21 |
| 3 | 5,282 | **72.38%** | ± 3.02% | ~18 |

**Key Observation:** Deepening the circuit from 1 to 3 layers yields a massive ~6% accuracy boost at the cost of only 8 total parameters. While there is no evidence of completely flat Barren Plateaus at this depth; the visible accuracy dips during training indicate a highly rugged, non-convex loss landscape. The classical optimizer is able to learn (but it occasionally struggles to smoothly navigate the complex quantum parameter space).


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