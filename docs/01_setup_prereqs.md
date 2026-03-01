### **Experimental Setup and Prerequisites**

This document outlines the hardware and software environment required to reproduce the classical and quantum baselines in this repository.

#### **Hardware Specifications**
(These are the specifications I used)
* **Primary Compute:** Google Colab Environment
* **Accelerator:** NVIDIA T4 Tensor Core GPU
* **Local Development:** macOS (Apple Silicon M1) - *Used for code modularization and version control*

#### **Software Dependencies**
The environment is strictly version-controlled to prevent conflicts between classical deep learning frameworks and quantum simulators 

Key libraries include:
* `torch` & `torchvision`: For classical CNN architecture and dataset loading.
* `pennylane` (v0.37.0+): For constructing hybrid quantum-classical nodes (QNodes).
* `qiskit` (v1.1.0+): Backend for quantum circuit execution.

For the exact environment reproduction, initialize a virtual environment and run:
```bash
pip install -r requirements.txt
```

#### **Execution Workflow**
To ensure reproducibility, all heavy computations are executed via cloud GPUs. 
1.  Core logic is modularized in the `src/` directory.
2.  Execution is handled via Jupyter Notebooks (`notebooks/`) running on Google Colab.
3.  The Colab environment clones this repository and dynamically imports the `src/` modules for execution.