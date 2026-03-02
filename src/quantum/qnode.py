import pennylane as qml
from .feature_map import get_feature_map
from .ansatz import get_ansatz

def create_qnode(num_qubits, num_layers, device_name="default.qubit"):
    """
    fuses the feature map and ansatz into a single quantum node (qnode).
    the qnode acts as a bridge between classical data and quantum operations.
    """
    #define the simulated quantum hardware
    dev = qml.device(device_name, wires=num_qubits)
    
    feature_map = get_feature_map(num_qubits)
    ansatz = get_ansatz(num_qubits, num_layers)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        #encode classical data into qubits
        feature_map(inputs)
        
        #apply trainable quantum gates
        ansatz(weights)
        
        #measure the expectation value of each qubit (the output)
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    return qnode