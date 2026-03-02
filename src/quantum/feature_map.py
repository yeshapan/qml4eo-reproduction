import pennylane as qml

def get_feature_map(num_qubits):
    """
    returns a function that applies angle encoding to classical inputs.
    each classical feature (pixel value) becomes the rotation angle of a qubit.
    """
    def feature_map(inputs):
        #angle embedding takes our flattened classical tensor and rotates the qubits
        #we use 'Y' rotation bcoz it's standard for real-valued image data
        qml.AngleEmbedding(features=inputs, wires=range(num_qubits), rotation='Y')
    
    return feature_map