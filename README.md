# **QML4EO: Reproducing and Stress-Testing Hybrid Quantum Networks**

Rebuilding and stress-testing a Hybrid Quantum Convolutional Neural Network (HQCNN) for Earth Observation. This repo benchmarks quantum models against classical baselines. It breaks down circuit performance on the EuroSAT dataset through three core experiments: qubit scaling, circuit depth and the entanglement paradox.

### **Repository Structure**
The codebase strictly separates classical benchmarks, pure quantum math and the hybrid models that glue them together. All training logs and mathematical proofs are saved in the Jupyter notebooks.

```
QML4EO-reproduction/
├── data/                       # Local storage for EuroSAT dataset
├── docs/                       # Formal experimental reports and findings
├── notebooks/                  # Executable training logs and proofs
├── src/                        # Core source code modules
│   ├── baselines/              # Classical neural network architectures (cnn.py)
│   ├── models/                 # Hybrid architectures glueing PyTorch & PennyLane (hqcnn.py)
│   ├── quantum/                # Pure PennyLane math (feature_map, ansatz, qnode)
│   └── utils/                  # Data loaders, transformations and metrics
├── requirements.txt            # Python dependencies (PyTorch, PennyLane)
└── README.md
```

### **Acknowledgments & References**
This project builds directly on the research and open-source tutorials from Prof. Silvia Liberata Ullo, @alessandrosebastianelli and their collaboratos at the ESA Φ-lab and University of Sannio.

Specifically, this repository references the concepts demonstrated in:
* [AI4EO](https://github.com/alessandrosebastianelli/AI4EO)
* [awesome-QC4EO](https://github.com/alessandrosebastianelli/awesome-QC4EO)
* [QML4EO-tutorial](https://github.com/alessandrosebastianelli/QML4EO-tutorial)