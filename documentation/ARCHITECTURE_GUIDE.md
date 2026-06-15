### **Quantum & Hybrid Architecture (Guide)**

*This guide breaks down exactly what we've built in the `src/quantum/` and `src/models/` directories*

#### **1. The Classical-to-Quantum Bottleneck**
To understand why we need a "Hybrid" model $\leftarrow$ we first need to understand the limitations of modern quantum hardware (the NISQ era).

The native resolution of a EuroSAT image is $64 \times 64$ pixels. However, to leverage the massive feature-extraction power of a state-of-the-art classical network, we utilize a pre-trained ResNet18 backbone, which strictly expects input dimensions of $224 \times 224$. 
* **The Upscaling Process:** We stretch the original $64 \times 64$ images to $224 \times 224$ using **Bilinear Interpolation**. This mathematical algorithm smoothly estimates and fills in the new pixels based on their neighbors, preserving the underlying geographical shapes and textures without corrupting the original image features.
* **The Bottleneck:** A $224 \times 224$ image with 3 color channels contains 150,528 total values. Current quantum computers do not have nearly enough stable qubits to encode this much raw data directly.
* **The Solution:** We use the frozen ResNet18 as a classical "feature extractor". The ResNet takes the upscaled $224 \times 224$ image, extracts the most important deep spatial patterns, and compresses it into a 512-dimensional vector. 
* A trainable classical linear layer then compresses those 512 features down to a tiny vector of numbers (e.g., 4 values). 
* We finally feed these 4 highly concentrated values into a 4-qubit quantum circuit.

#### **2. The Feature Map (`src/quantum/feature_map.py`)**
**Goal:** To translate classical data (numbers) into quantum data (states).

In classical ML, we just pass numbers directly into a node. In quantum computing, information is stored in the physical state of a qubit. To put data *into* a qubit, we have to rotate it.

For this, we use **Angle Encoding**. Imagine a qubit as a sphere. Angle encoding takes our classical number (the compressed feature from the ResNet) and uses it as the angle to rotate the qubit: 
* If the classical output is a $0$ $\rightarrow$ the qubit doesn't rotate
* If the classical output is a $\pi$ $\rightarrow$ the qubit flips completely upside down

We use an $R_y$ rotation gate to do this.

#### **3. The Ansatz (`src/quantum/ansatz.py`)**
**Goal:** The trainable "neural network" layer of the quantum circuit.

Once the data is encoded into the qubits, we need to process it. The Ansatz (German for "approach" or "guess") is a sequence of quantum gates with variable settings (weights). 

It has two main jobs:
1. **Rotation:** 
    * Applying parameterized gates (like $R_x, R_y, R_z$).
    * During training, PyTorch's optimizer will adjust the angles of these gates to minimize the loss function (exactly like updating weights in a classical network).
2. **Entanglement:** 
    * Applying CNOT gates to link the qubits together.
    * This is where the "quantum advantage" theoretically lives. By entangling the qubits $\rightarrow$ they process the features simultaneously in a highly correlated way that classical computers struggle to simulate.

##### **(Note:) What is a CNOT Gate?**
In classical computing, we have standard logic gates like AND, OR and NOT. In quantum computing, the **CNOT (Controlled-NOT)** gate is the fundamental building block for multi-qubit operations. 

It always operates on exactly two qubits at a time:
1.  **The Control Qubit:** The trigger
2.  **The Target Qubit:** The one that gets flipped

How it works: 
* If the Control qubit is in state `1` $\rightarrow$ it applies a NOT operation (flips) to the Target qubit
* If the Control qubit is in state `0` $\rightarrow$ it leaves the Target qubit completely alone

**Why it matters in QML:** CNOT gates are the primary way we generate **entanglement**. By linking the state of one qubit directly to the state of another $\rightarrow$ the quantum neural network can learn deep correlations between the different classical features we encoded. 

However, entanglement is a double-edged sword. As we proved in our Ablation Studies, forcing high-level classical features to entangle can sometimes introduce so much mathematical noise that the optimizer fails. Sometimes, removing CNOTs entirely and letting the qubits act as independent classifiers yields much higher accuracy!

#### **4. The QNode (`src/quantum/qnode.py`)**
**Goal:** Bridge the classical and quantum worlds.

A QNode (Quantum Node) is PennyLane's way of packaging the Feature Map and the Ansatz into a single, executable function. 

Crucially, the QNode handles **Measurement**. Quantum superposition collapses when you look at it. To get usable data back out of the quantum circuit and into PyTorch, we measure the **Expectation Value** (specifically, the Pauli-Z expectation, $\langle \sigma_z \rangle$) of each qubit. 
* This converts the complex quantum state back into a simple classical float array of numbers between $-1$ and $1$.

#### **5. The Hybrid Model (`src/models/hqcnn.py`)**
**Goal:** To glue everything together.

This is the final architectural step. The Hybrid Quantum Convolutional Neural Network (HQCNN) orchestrates the entire pipeline from end to end:

1. **Input:** Takes the upscaled 224x224 EuroSAT image.
2. **Classical Forward Pass:** The frozen ResNet18 backbone pulls out the edges textures, and colors, condensing them into a 512-dimensional vector. A trainable linear layer compresses this down to an $N$-dimensional vector.
3. **Quantum Forward Pass:** The vector is handed to the QNode. The QNode maps it to qubits, entangles them (if specified), rotates them, and measures them.
4. **Final Classification:** The measured values from the quantum circuit are passed into a final classical fully-connected (`Linear`) layer $\rightarrow$ which outputs the final 10 probabilities for the land-cover classes.