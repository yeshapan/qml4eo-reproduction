import pennylane as qml
from .feature_map import get_feature_map
from .ansatz import get_ansatz

def create_qnode(num_qubits, num_layers, device_name="default.qubit", entanglement_type="none"):
    """
    Fuses the feature map and ansatz into an executable quantum node.
    Acts as the bridge passing the exact topology request down to the ansatz
    """
    # Define the simulated quantum hardware
    dev = qml.device(device_name, wires=num_qubits)
    
    feature_map = get_feature_map(num_qubits)

    ansatz = get_ansatz(num_qubits, num_layers, entanglement_type)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        # Encode classical data into qubits
        feature_map(inputs)
        
        # Apply trainable quantum gates
        ansatz(weights)
        
        # Measure the expectation value of each qubit (the output)
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    return qnode