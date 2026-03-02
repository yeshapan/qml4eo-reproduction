## **QML4EO: Rebuilding and Stress-Testing Hybrid Quantum Networks**

Rebuilding and stress-testing a Hybrid Quantum Convolutional Neural Network (HQCNN) for Earth Observation. This repository benchmarks quantum models against classical baselines. It breaks down circuit performance on the EuroSAT dataset through three core experiments: qubit scaling, circuit depth and the entanglement paradox.

### **Core Achievements & Experiments**
* **The Classical Baseline:** Established a PyTorch CNN baseline, achieving 73.94% accuracy on a 64x64 RGB EuroSAT subset.
* **Quantum Integration:** Demonstrated successful gradient flow across a hybrid PyTorch-PennyLane architecture using standard classical optimizers.
* **Experiment 1 (Qubit Scaling):** Investigated the quantum information bottleneck. Observations suggest that scaling from 2 to 8 qubits lifts the learning ceiling but incurs exponential computational cost in classical simulation.
* **Experiment 2 (Circuit Depth):** Evaluated quantum expressivity. Empirical results indicate that adding just 8 parameters (scaling from 1 to 3 ansatz layers) improved validation accuracy by over 16% under a strict 5-epoch limit.
* **Experiment 3 (Entanglement Strategy):** Ablated CNOT gates to assess entanglement utility. Preliminary findings show unentangled circuits converge faster in early epochs (smoother loss landscape), whereas entangled circuits suggest higher expressivity but exhibit severe non-convex instability during early training.

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