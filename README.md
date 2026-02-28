## **QML4EO: Empirical Reproduction and Baseline Analysis**
A structured exploration, reproduction and analysis of **Quantum Machine Learning models for Earth Observation (QML4EO)**. Includes classical baseline comparisons, ablation studies, and parameter scaling metrics.

### **Objectives**

1. **Reproduce Results:** Execute existing hybrid QML models and replicate baseline metrics on Earth Observation datasets.
2. **Confirm & Document Discrepancies:** Objectively log differences between reported literature values and observed local metrics (including sim-to-real gaps).
3. **Establish Classical Baselines:** Train standard classical models (e.g., CNNs) on the same datasets for direct parameter and accuracy comparisons.
4. **Conduct Ablation Studies:** Isolate and test specific quantum layers to analyze their impact on training stability and convergence.
5. **Analyze Parameter Scaling:** Evaluate how altering the number of qubits, quantum layers, and classical parameters affects overall performance and computational overhead.
6. **Dataset Scaling:** Test models on smaller, optimized subsets of EO data (e.g., binary classification slices of EuroSAT) to enable faster iteration and testing.

### **Probable Repository Structure**

```
QML4EO-reproduction
├── data/                       #raw + pre-processed EO datasets (e.g., EuroSAT subset)
├── docs/                       #reports and documentation
│   ├── setup_and_prereqs.md
│   ├── dataset_details.md
│   ├── structured_report.md    #hyperparameters, observed metrics, and discrepancies
│   └── ablation_study.md       #parameter scaling and stability analysis
├── experiments/                #executable scripts for training and testing
│   ├── train_hybrid.py         
│   ├── train_classical.py      
│   └── run_ablation.py         
├── notebooks/                  #jupyter notebooks demonstrating core Qiskit/PennyLane concepts
├── src/                        #core source code modules
│   ├── baselines/              #classical neural network architectures (PyTorch)
│   ├── quantum/                #quantum feature maps and parameterized circuits (PennyLane/Qiskit)
│   └── utils/                  #data loaders, metric calculations, and logging
├── requirements.txt            #python dependencies
└── README.md
```

### **Acknowledgments & References**
This project is deeply inspired by and builds upon the foundational research, methodologies and open-source tutorials developed by Prof. Silvia Liberata Ullo, Alessandro Sebastianelli and their collaborators at the ESA Φ-lab and the University of Sannio. 

Specifically, this repository references the architectural approaches and baselines demonstrated in:
* [AI4EO](https://github.com/alessandrosebastianelli/AI4EO)
* [awesome-QC4EO](https://github.com/alessandrosebastianelli/awesome-QC4EO)
* [QML4EO-tutorial](https://github.com/alessandrosebastianelli/QML4EO-tutorial)