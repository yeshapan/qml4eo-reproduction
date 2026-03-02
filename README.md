## **QML4EO: Rebuilding and Stress-Testing Hybrid Quantum Networks**

Rebuilding and stress-testing a Hybrid Quantum Convolutional Neural Network (HQCNN) for Earth Observation. This repository benchmarks quantum models against classical baselines. It breaks down circuit performance on the EuroSAT dataset through three core experiments: qubit scaling, circuit depth and the entanglement paradox.

### **Core Achievements & Experiments**
* **The Classical Baseline:** Built a lightweight PyTorch CNN. This set a strict performance floor of 73.94% accuracy on 64x64 RGB EuroSAT images.
* **Quantum Integration:** Successfully bridged PyTorch and PennyLane. Proved that classical optimizers can push gradients through a quantum simulator.
* **Experiment 1 (Qubit Scaling):** Tested the quantum information bottleneck. Found that widening the bottleneck (from 2 to 8 qubits) raises the learning ceiling but exponentially chokes GPU simulation time.
* **Experiment 2 (Circuit Depth):** Tested quantum expressivity. Proved that adding just 8 trainable parameters (jumping from 1 to 3 ansatz layers) boosts accuracy by over 16%.
* **Experiment 3 (Entanglement Strategy):** Physically removed CNOT gates to test quantum advantage. Discovered that unentangled circuits train much faster in short 5-epoch windows. Entangled circuits have higher ultimate capacity but suffer from highly unstable, non-convex loss landscapes.

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
This project builds directly on the research and open-source tutorials from Prof. Silvia Liberata Ullo, Alessandro Sebastianelli and the ESA Φ-lab. Their foundational work at the University of Sannio heavily inspired the quantum baselines and PennyLane integration used here.

Specifically, this repository references the concepts demonstrated in:
* [AI4EO](https://github.com/alessandrosebastianelli/AI4EO)
* [awesome-QC4EO](https://github.com/alessandrosebastianelli/awesome-QC4EO)
* [QML4EO-tutorial](https://github.com/alessandrosebastianelli/QML4EO-tutorial)