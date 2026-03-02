import pennylane as qml

def get_ansatz(num_qubits, num_layers):
    """
    creates a trainable quantum circuit (ansatz) using basic entangling layers.
    this allows the qubits to interact (entangle) and learn features.
    """
    def ansatz(weights):
        #weights will be a tensor of shape (num_layers, num_qubits) provided dynamically by pytorch during training
        qml.BasicEntanglerLayers(weights=weights, wires=range(num_qubits))
    
    return ansatz