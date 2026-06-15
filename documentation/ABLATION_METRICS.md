### **QML4EO Ablation Study: Circuit Parameters and Stability**

This doc explores how the hyper-parameters of the quantum circuit impact the overall performance of the Hybrid Quantum Convolutional Neural Network (HQCNN) on the EuroSAT dataset.

#### **What is an Ablation Study?**
In ML, an ablation study is a scientific method applied to neural network architectures. 

Think of it like playing a game of Jenga or taking a car engine apart piece by piece: you remove, alter or scale down one specific component at a time to see how much the system relies on it. 

If you remove a layer and the model's accuracy crashes, that layer was critical. If you remove it and the accuracy stays the same, that layer was unnecessary bloat.

#### **Why is this critical for Quantum Machine Learning (QML)?**
In classical ML, adding more parameters is cheap. In QML, it is heavily constrained by current hardware limitations:
1. **Simulation Cost:** Simulating quantum states classically requires exponential memory. Adding just one more qubit doubles the mathematical complexity.
2. **NISQ Era Noise:** On physical quantum hardware (Noisy Intermediate-Scale Quantum), longer circuits introduce more decoherence and noise.

So, we must rigorously prove that every single qubit and quantum operation (gate) we add is actually contributing to the model's learning capacity, rather than just slowing down the training time.

---

> NOTE: Please refer to the notebooks in `notebooks/` to see the raw output logs of each experiment and their visual analysis.

#### **1. Qubit Scaling Analysis & The Information Bottleneck**
*Testing the performance impact of encoding the 512 classical features extracted by the ResNet backbone into quantum states of varying sizes*

*Evaluated across 3 seeds for 10 epochs. Assesses the tradeoff between the dimensionality bottleneck and mathematical stability.*

| Qubits | Ansatz Layers | Total Parameters | Final Mean Accuracy | Std Dev (±) | Time/Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 1 | ~11.18M | **41.81%** | 1.79% | ~58s |
| 4 (Baseline) | 1 | ~11.18M | **74.86%** | 4.97% | ~60s |
| 6 | 1 | ~11.18M | **80.17%** | 20.80% | ~65s |

**Key Observation:** The 2-qubit circuit creates a severe information bottleneck, destroying too much spatial data and capping accuracy at ~41%. Scaling to 6 qubits unlocks massive capacity (hitting ~95% on select seeds) but exponentially increases the jaggedness of the loss landscape, causing a massive ±20.80% variance where some seeds completely fail. 4 Qubits is the empirical sweet spot.

#### **2. Circuit Depth (Ansatz Layers)**
*Testing the expressivity of the quantum layer by repeating the trainable operations. Repeating the ansatz gives the model more parameters to tune, but it also increases the risk of barren plateaus and gradient collapse.*

*Evaluated across 3 seeds for 10 epochs. Bottleneck locked at 4 qubits with 'ring' entanglement.*

| Ansatz Layers | Total Parameters | Final Mean Accuracy | Std Dev (±) | Performance Profile |
| :---: | :---: | :---: | :---: | :--- |
| 1 (Baseline) | ~11.18M | **74.86%** | 4.97% | Stable convergence |
| 2 | ~11.18M | **73.96%** | 9.80% | Barren Plateau Onset (Seed 42 flatlined ~61%) |
| 3 | ~11.18M | **87.17%** | 7.30% | Peak 95.31%, but suffers catastrophic forgetting |

**Key Observation:** Deepening the circuit from 1 to 2 layers actually *decreased* average performance and doubled the variance (± 9.80%) due to the onset of barren plateaus. While 3 layers possessed the expressibility to perfectly map the dataset (95.31%), it was highly unstable (with gradients frequently overshooting optimal minimums). 

#### **3. Entanglement Strategy**
*Assessing the role of quantum entanglement in feature extraction. Entanglement is what theoretically gives QML an edge over classical models.*

*Evaluated across 3 seeds for 10 epochs. Tests if theoretical quantum expressivity translates to empirical accuracy. Bottleneck locked at 4 qubits, 1 layer.*

| Entanglement Type | Final Mean Accuracy | Std Dev (±) | Time/Epoch |
| :--- | :---: | :---: | :---: |
| None (0 CNOTs) | **96.38%** | 1.09% | ~66s |
| Ring (Sequential) | **74.86%** | 4.97% | ~66s |
| Full (All-to-All) | **92.15%** | 5.59% | ~66s |

**Key Observation:** The "Entanglement Paradox" is confirmed. Because the frozen classical ResNet18 is already a highly proficient spatial feature extractor $\rightarrow$ removing CNOT gates completely (None) allowed the qubits to act as independent, parallel classifiers. This eliminated cross-qubit noise $\implies$ resulted in an incredibly smooth optimization landscape and the highest overall accuracy (96.38%).

#### **4. Key Takeaways**
* **The Entanglement Paradox:** Unentangled quantum circuits are significantly more effective when paired with powerful classical feature extractors. They provide a smooth loss landscape $\implies$ allow classical optimizers to easily reach ~96.38% accuracy. Entanglement creates a highly rugged, non-convex parameter space that standard optimizers struggle to navigate within 10 epochs.
* **Overparameterization & Barren Plateaus:** Increasing circuit depth does not inherently improve a hybrid model. Deepening to 2 layers flattened the loss landscape (Barren Plateaus). Deepening to 3 layers caused high-risk/high-reward gradient volatility. 1 Layer remains the most computationally efficient and reliable configuration.
* **The Information Bottleneck:** Qubit scaling must be perfectly balanced. Compressing 512 classical features into 2 qubits destroys critical spatial data (~41% accuracy). While 6 qubits increases capacity, it introduces extreme variance (±20.80%). 4 Qubits provides the necessary equilibrium between information retention and mathematical stability.