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

**Key Observation:** Deepening the circuit from 1 to 3 layers yields a ~6% accuracy boost at the cost of only 8 total parameters. 
While there is no evidence of completely flat Barren Plateaus at this depth; the visible accuracy dips during training indicate a highly rugged, non-convex loss landscape. 
The classical optimizer is able to learn (but it occasionally struggles to smoothly navigate the complex quantum parameter space).


#### **3. Entanglement Strategy**
*Assessing the role of quantum entanglement in feature extraction. Entanglement is what theoretically gives QML an edge over classical models*

*Evaluated across 3 seeds for 15 epochs. Tests if theoretical quantum expressivity translates to empirical accuracy. Bottleneck locked at 4 qubits, 1 layer.*

| Entanglement Type | Final Mean Accuracy | Std Dev |
| :--- | :---: | :---: |
| None (No CNOTs) | **76.40%** | ± 0.82% |
| Basic (CNOT Ring) | **61.44%** | ± 2.25% |

**Key Observation:** The "Entanglement Paradox" is confirmed. The unentangled circuit outperformed the entangled baseline by ~15%. While entanglement provides higher theoretical expressivity, it creates a highly rugged loss landscape that classical optimizers struggle to navigate efficiently within 15 epochs.

#### **4. Key Takeaways**
* **The Entanglement Paradox:** Unentangled quantum circuits are significantly more effective for near-term QML. They provide a smooth loss landscape, allowing classical optimizers to easily reach ~76.4% accuracy. Entanglement creates a highly rugged, non-convex parameter space that Adam struggles to navigate within 15 epochs.
* **The Expressivity Sweet Spot:** If an entangled circuit is required, deepening it from 1 to 3 layers is highly parameter-efficient. It yields a ~6% accuracy boost at the cost of only 8 total parameters without triggering flat Barren Plateaus.
* **The Simulation Wall & Bottleneck:** Qubit scaling suffers from severe diminishing returns. Squeezing 32 classical features into 2 qubits destroys too much spatial data (~49% accuracy). Performance plateaus at 6 qubits (~70%). Pushing to 8 qubits yields zero additional accuracy but exponentially slows down GPU simulation speed.