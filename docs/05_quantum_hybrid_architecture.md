### **Quantum & Hybrid Architecture (Guide)**

This guide breaks down exactly what we've built in the `src/quantum/` and `src/models/` directories (starting from the absolute basics).

#### **1. The Classical-to-Quantum Bottleneck**
To understand why we need a "Hybrid" model we first need to understand the limitations of modern quantum hardware (the NISQ era). 

A standard EuroSAT image is 64x64 pixels with 3 color channels (12,288 total values). Current quantum computers don't have enough stable qubits to encode an entire image directly. 
* **The Solution:** We use a classical Convolutional Neural Network (CNN) as a "feature extractor". The CNN takes the massive 64x64 image, compresses it and learns the most important patterns; reducing it down to a tiny vector of numbers (e.g., 4 values). 
* We then feed these 4 highly concentrated values into a 4-qubit quantum circuit.


#### **2. The Feature Map (`src/quantum/feature_map.py`)**
**Goal**: To translate classical data (numbers) into quantum data (states)

In classical ML, we just pass numbers directly into a node. In quantum computing, information is stored in the physical state of a qubit. To put data *into* a qubit we have to rotate it.

For this, we use **Angle Encoding**. Imagine a qubit as a sphere. Angle encoding takes our classical number (the compressed feature from the CNN) and uses it as the angle to rotate the qubit: 
* If the CNN outputs a $0$, the qubit doesn't rotate 
* If the CNN outputs a $\pi$, the qubit flips completely upside down 
We use an $R_y$ rotation gate to do this.


#### **3. The Ansatz (`src/quantum/ansatz.py`)**
**Goal:** The trainable "neural network" layer of the quantum circuit

Once the data is encoded into the qubits, we need to process it. The Ansatz (German for "approach" or "guess") is a sequence of quantum gates with variable settings (weights). 

It has two main jobs:
1. **Rotation:** Applying parameterized gates (like $R_x, R_y, R_z$). During training, PyTorch's optimizer will adjust the angles of these gates to minimize the loss function (exactly like updating weights in a classical network).
2. **Entanglement:** Applying CNOT gates to link the qubits together. This is where the "quantum advantage" theoretically lives. By entangling the qubits, they process the features simultaneously in a highly correlated way that classical computers struggle to simulate.

##### **(Note:) What is a CNOT Gate?**
In classical computing, we have standard logic gates like AND, OR and NOT. In quantum computing, the **CNOT (Controlled-NOT)** gate is the fundamental building block for multi-qubit operations. 

It always operates on exactly two qubits at a time:
1.  **The Control Qubit:** The trigger
2.  **The Target Qubit:** The one that gets flipped

**How it works:** If the Control qubit is in state `1`, it applies a NOT operation (flips) to the Target qubit. If the Control qubit is in state `0`, it leaves the Target qubit completely alone. 

**Why it matters in QML:** CNOT gates are the primary way we generate **entanglement**. By linking the state of one qubit directly to the state of another, the quantum neural network can learn deep, complex correlations between the different classical features we encoded. 

In fact, if you remove all the CNOT gates from a quantum circuit, the qubits can no longer talk to each other. When that happens, your cutting-edge quantum model chemically degrades into a basic, un-entangled classical linear regression model! This is exactly what we'll prove in our Ablation Study.


#### **4. The QNode (`src/quantum/qnode.py`)**
**Goal:** Bridge the classical and quantum worlds.

A QNode (Quantum Node) is PennyLane's way of packaging the Feature Map and the Ansatz into a single, executable function. 

Crucially, the QNode handles **Measurement**. Quantum superposition collapses when you look at it. To get usable data back out of the quantum circuit and into PyTorch; we measure the **Expectation Value** (specifically, the Pauli-Z expectation, $\langle \sigma_z \rangle$) of each qubit. 
* This converts the complex quantum state back into a simple classical float array of numbers between $-1$ and $1$.


#### **5. The Hybrid Model (`src/models/hqcnn.py`)**
**Goal:** To glue everything together

This is the final architectural step. The Hybrid Quantum Convolutional Neural Network (HQCNN) orchestrates the entire pipeline from end to end:

1. **Input:** Takes the raw 64x64 EuroSAT image.
2. **Classical Forward Pass:** The PyTorch CNN layers pull out the edges, textures, and colors, condensing them into an $N$-dimensional vector.
3. **Quantum Forward Pass:** The vector is handed to the QNode. The QNode maps it to qubits, entangles them, rotates them, and measures them.
4. **Final Classification:** The measured values from the quantum circuit are passed into a final classical fully-connected (`Linear`) layer, which outputs the final 10 probabilities for the land-cover classes.