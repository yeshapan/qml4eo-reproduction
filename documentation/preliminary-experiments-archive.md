# **Historical Archive: Preliminary QML4EO Experiments and Baselines**

*Notebooks with executed code for preliminary version - v1 are stored in `notebooks/v1-preliminary/` folder in this repository*

## **Experimental Setup + Data Specifications**

### **Hardware Specifications**
(These are the specifications I used)
* **Primary Compute:** Google Colab Environment
* **Accelerator:** NVIDIA T4 Tensor Core GPU
* **Local Development:** macOS (Apple Silicon M1) - *Used for code modularization and version control*

### **Software Environment**
* Quantum Backend:
    * `pennylane>=0.37.0`
    * `qiskit>=1.1.0,<2.0.0`
    * `qiskit-machine-learning>=0.8.0,<0.9.0`

* Core Classical Deep Learning
    * `torch>=2.0.0`
    * `torchvision>=0.15.0`

### **EuroSAT Dataset (RGB)**
* Task: Land Cover Classification (10 Classes)
* Total Images: 27,000 (Native resolution: $64 \times 64 \times 3$)
* Data Splits (via Seed=42): 
    * 80% Train (21,600)
    * 20% Val (5,400) 
* Preprocessing Pipeline (V1.0): 
    * Pixel values converted to PyTorch tensors and normalized via ImageNet standards:
        * Mean: [0.485, 0.456, 0.406]
        * Std: [0.229, 0.224, 0.225]
    * **No spatial data augmentation (random flips or rotations) was used in these preliminary runs**
* Evaluation Protocol: All metrics represent the mean and standard deviation across three independent runs ($15$ epochs, $Seeds = 42, 100, 2026$).

## **Architecture Used**
Preliminary experiments used "Late Hybrid Scheme" $\rightarrow$ substituted final fully-connected layers of classical CNN with VQC (Variational Quantum Circuit)

### **Classical-to-Quantum Bottleneck**
Current NISQ simulators suffer from exponential memory scaling ($O(2^N)$) for $N$ qubits $\implies$ the $12,288$ raw pixel values cannot be directly encoded.
* **Feature Extractor**: A custom 2-layer Classical CNN extracts spatial textures and condenses them using an `AdaptiveAvgPool2D` layer to a flat $1D$ vector of $32$ features.
* **Dimensionality Reduction**: A classical `nn.Linear(32, N)` layer acts as a bottleneck, strictly compressing the $32$ features down to $N$ variables to match the available qubits.
* **The Scaling Function**: To prevent periodic gradient wrap-around (where rotations of $0$ and $2\pi$ result in identical states) $\rightarrow$ the bottleneck vector $x$ is scaled mathematically into radians:

$$x_{quantum} = \tanh(x_{classical}) \times \pi$$

This rigorously bounds all classical inputs into the safe rotational range of $[-\pi, \pi]$.

### **State Preparation (Angle Encoding)**
* Classical tensors are translated into complex probability amplitudes via Angle Encoding 
* We apply independent $Y$-axis rotation gates ($R_y$) to initialize each qubit
* $R_y$ explicitly alters the amplitude along the real plane without introducing complex phase shifts $\implies$ optimizes the embedding of real-valued pixel data.

### **The Ansatz (Parameterized Quantum Layer)** 
It is the trainable sequence of quantum operations defined by:
* Rotation: Parameterized $R_y(\theta)$ gates updated via the classical optimizer to minimize the Cross-Entropy loss.
* Entanglement: CNOT gates acting on pairs of qubits to inextricably link their states $\leftarrow$ attempt to map the data into a highly correlated, high-dimensional Hilbert space.

### **Readout (Measurement)**
* To map the complex quantum state back to classical PyTorch float arrays $\rightarrow$ we measure the Pauli-Z expectation value ($\langle \sigma_z \rangle$) of each qubit.
* This destructive measurement collapses the superposition into continuous logits ranging exactly from $-1$ to $1$.

## **Initial Baselines: Classical vs. Hybrid**
*Evaluated across 3 seeds. Used Adam optimizer*

### **1. Classical Baseline Metrics (CNN)**

*The classical baseline utilizes a lightweight CNN with Adaptive Average Pooling to remain invariant to the heavy downscaling required by quantum circuits*

| Image Size | Total Parameters | Epochs | Final Training Loss | Final Val Accuracy | Time/Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 64x64 | **5,418** | 15 | **0.59** | **79.15% ± 0.63%** | ~13s |

---

### **2. Hybrid Quantum-Classical Metrics (HQCNN)**

The baseline utilizes a minimal 4-qubit, 1-layer quantum circuit to establish a lower bound for QML performance and verify gradient flow.*

| Qubits | Quantum Layers | Image Size | Total Parameters | Epochs | Final Val Accuracy (Mean ± Std) | Time/Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4 | 1 | 64x64 | **5,274** | 15 | **66.08% ± 2.42%** | ~25s |

## **4. Ablation Studies**

### **1. Qubit Scaling Analysis**
*Testing the performance impact of encoding the classical data into larger quantum states. (Baseline: 4 Qubits)*

*Evaluated across 3 seeds for 15 epochs. Assesses the tradeoff between quantum state size and classical simulation cost.*

| Qubits | Ansatz Layers | Total Parameters | Final Mean Accuracy | Std Dev | Speed (it/s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 1 | 5,186 | **49.56%** | ± 1.62% | ~33 |
| 4 | 1 | 5,274 | **66.08%** | ± 2.42% | ~26 |
| 6 | 1 | 5,362 | **70.25%** | ± 1.75% | ~21 |
| 8 | 1 | 5,450 | **69.83%** | ± 2.78% | ~16 |

**Key Observation:** Performance plateaus after 6 qubits. The 8-qubit space is too vast for a 1-layer ansatz to navigate effectively within 15 epochs, resulting in identical accuracy but severe simulation penalties.

### **2. Circuit Depth (Ansatz Layers)**
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


### **3. Entanglement Strategy**
*Assessing the role of quantum entanglement in feature extraction. Entanglement is what theoretically gives QML an edge over classical models*

*Evaluated across 3 seeds for 15 epochs. Tests if theoretical quantum expressivity translates to empirical accuracy. Bottleneck locked at 4 qubits, 1 layer.*

| Entanglement Type | Final Mean Accuracy | Std Dev |
| :--- | :---: | :---: |
| None (No CNOTs) | **76.40%** | ± 0.82% |
| Basic (CNOT Ring) | **61.44%** | ± 2.25% |

**Key Observation:** The "Entanglement Paradox" is confirmed. The unentangled circuit outperformed the entangled baseline by ~15%. While entanglement provides higher theoretical expressivity, it creates a highly rugged loss landscape that classical optimizers struggle to navigate efficiently within 15 epochs.

## **Architectural Flaws Motivating v2 Redesign**
* **The "Lazy Quantum" Hypothesis**: 
    * Unentangled (linear) circuit hit 76.40% accuracy $\leftarrow$ close to the classical baseline (79.15%)
    * This proves that the classical CNN is doing the vast majority of the analytical work. The quantum layer acts merely as a rotational filter.
* **The Shallow Semantic Bottleneck**: 
    * The custom 2-layer CNN cannot extract deep, non-linear latent features
    * Inputs hitting the quantum layer lack semantic richness
    * So, the architecture requires replacing the custom CNN with a deep Transfer Learning backbone (ResNet18).
* **Coupled Optimizer Instability**: 
    * Adam ($LR=0.001$) overshoots the narrow quantum minima
    * Classical backpropagation and the quantum parameter-shift gradients operate at completely different scales
    * So, the optimiser needs to have Decoupled Learning Rates ($10^{-3}$ for classical, $10^{-4}$ for quantum) in future pipelines.