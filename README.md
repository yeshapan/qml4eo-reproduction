## **QML4EO: Reproducing and Stress-Testing Hybrid Quantum Networks**

Rebuilding and stress-testing a Hybrid Quantum Convolutional Neural Network (HQCNN) for Earth Observation. This repo benchmarks quantum models against classical baselines. It breaks down circuit performance on the EuroSAT dataset through three core experiments: qubit scaling, circuit depth and the entanglement paradox.

### **Experimental Results Summary**
All metrics represent the mean and standard deviation across three independent runs ($15$ epochs, $Seed=42, 100, 2026$).

**Core Baselines**
| Model | Parameters | Mean Accuracy | Std Dev (±) |
| :--- | :---: | :---: | :---: |
| **Classical CNN Baseline** | 5,418 | **79.15%** | 0.63% |
| **Hybrid QCNN (4-Qubit Baseline)** | 5,274 | **66.08%** | 2.42% |

**Ablation Study Highlights**
| Experiment | Configuration | Parameters | Mean Accuracy | Std Dev (±) |
| :--- | :--- | :---: | :---: | :---: |
| **Qubit Scaling** | 2 Qubits | 5,186 | 49.56% | 1.62% |
| | 6 Qubits | 5,362 | 70.25% | 1.75% |
| | 8 Qubits | 5,450 | 69.83% | 2.78% |
| **Circuit Depth** | 2-Layer Ansatz | 5,278 | 69.69% | 4.03% |
| | 3-Layer Ansatz | 5,282 | **72.38%** | 3.02% |
| **Entanglement** | None (No CNOTs) | 5,186 | **76.40%** | 0.82% |
| | Basic (Ring) | 5,274 | 61.44% | 2.25% |

### **Open Questions**
* **The "Lazy Quantum" Hypothesis:** 
    * Our unentangled 4-qubit model hit 76.40% accuracy - which is close to our purely classical CNN baseline (79.15%). 
    * Since an unentangled quantum layer essentially just acts as a series of independent rotations; is the classical PyTorch feature extractor doing all the heavy lifting? 
    * We need a mathematical way to isolate the *actual* value the quantum layer is adding in these early hybrid schemes.

* **Is there a "sweet spot" for width vs. depth?** 
    * In Experiment 1, bumping the model from 6 to 8 qubits actually caused a slight drop in accuracy (70.25% down to 69.83%), even though it had more parameters 
    * This makes us wonder: if we widen the quantum register to 8+ qubits, do we strictly *need* a deeper circuit (more than 1 layer) to properly entangle that larger space? 
    * Finding the optimal ratio between qubit width and circuit depth is a critical next step.

* **The 2-Layer Variance Anomaly:** 
    * During Experiment 2, the 2-layer setup showed weirdly high instability across our different seeds (±4.03% standard deviation) compared to the 1-layer (±2.42%) and 3-layer (±3.02%) models. 
    * Does the quantum loss landscape go through an extremely rugged "awkward phase" at intermediate depths before settling into a more stable, trainable structure?

* **Do classical and quantum gradients need to be decoupled?** 
    * We noticed temporary accuracy "dips" during training, likely because the Adam optimizer was overshooting narrow quantum minima
    * In an end-to-end hybrid model, the classical CNN weights update at the exact same time as the quantum parameters. 
    * Since classical backpropagation and quantum parameter-shift gradients are fundamentally different, does training them together inherently cause instability? 
    * It might be worth testing entirely independent learning rates for the two halves of the network.

### **Repository Structure**
The codebase strictly separates classical benchmarks, pure quantum math and the hybrid models that glue them together. All training logs and mathematical proofs are saved in the Jupyter notebooks.

```
QML4EO-reproduction/
├── data/                       # local storage for EuroSAT dataset
├── docs/                       # formal experimental reports and findings
│   ├── 03_baseline_cnn_vs_hqcnn.md 
│   ├── 04_ablation_study.md       
│   └── 05_quantum_hybrid_architecture.md
├── notebooks/                  # executable training logs and proofs
│   ├── 01_classical_cnn_baseline.ipynb
│   ├── 02_hybrid_qcnn_baseline.ipynb
│   └── 03_ablation_study.ipynb
├── src/                        # core source code modules
│   ├── baselines/              # classical neural network architectures (cnn.py)
│   ├── models/                 # hybrid architectures glueing PyTorch & PennyLane (hqcnn.py)
│   ├── quantum/                # pure PennyLane math (feature_map, ansatz, qnode)
│   └── utils/                  # data loaders, transformations and metrics
├── requirements.txt            # python dependencies (PyTorch, PennyLane)
└── README.md
```

### **Acknowledgments & References**
This project builds directly on the research and open-source tutorials from Prof. Silvia Liberata Ullo, @alessandrosebastianelli and their collaboratos at the ESA Φ-lab and University of Sannio.

Specifically, this repository references the concepts demonstrated in:
* [AI4EO](https://github.com/alessandrosebastianelli/AI4EO)
* [awesome-QC4EO](https://github.com/alessandrosebastianelli/awesome-QC4EO)
* [QML4EO-tutorial](https://github.com/alessandrosebastianelli/QML4EO-tutorial)