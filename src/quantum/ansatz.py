import pennylane as qml

def get_ansatz(num_qubits, num_layers, entanglement_type="none"):
    """
    Creates the trainable quantum circuit
    Supports three distinct topologies: 'none', 'ring', and 'full'.
    """
    def ansatz(weights):
        if entanglement_type == "none":
            # Independent Qubits topology
            # Applies independent Y-rotations
            for layer in range(num_layers):
                for wire in range(num_qubits):
                    qml.RY(weights[layer][wire], wires=wire)
                    
        elif entanglement_type == "ring":
            # Ring topology
            # PennyLane's BasicEntanglerLayers uses Rx rotations and a ring of CNOTs
            qml.BasicEntanglerLayers(weights=weights, wires=range(num_qubits))
            
        elif entanglement_type == "full":
            # All-to-All (fully connected) topology
            for layer in range(num_layers):
                # 1. Apply the trainable rotations
                for wire in range(num_qubits):
                    qml.RY(weights[layer][wire], wires=wire)
                    
                # 2. Apply all-to-all CNOT entanglement
                for i in range(num_qubits):
                    for j in range(i + 1, num_qubits):
                        qml.CNOT(wires=[i, j])
                        
        else:
            raise ValueError("entanglement_type must be 'none', 'ring', or 'full'")
            
    return ansatz