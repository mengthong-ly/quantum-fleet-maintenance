import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# Number of qubits = number of features
N_QUBITS = 4

# Trainable parameters (theta)
theta = ParameterVector("θ", N_QUBITS)

def build_vqc(feature_vector):
    """
    Build a Variational Quantum Circuit for Fleet Maintenance
    feature_vector: array of 4 scaled features
    """
    qc = QuantumCircuit(N_QUBITS)

    # 1️⃣ Create superposition
    for i in range(N_QUBITS):
        qc.h(i)

    # 2️⃣ Encode classical data (feature encoding)
    for i, value in enumerate(feature_vector):
        qc.ry(value * np.pi, i)

    # 3️⃣ Variational (trainable) layers
    for i in range(N_QUBITS):
        qc.ry(theta[i], i)

    # 4️⃣ Entanglement (fleet correlations)
    for i in range(N_QUBITS - 1):
        qc.cx(i, i + 1)

    # 5️⃣ Measurement
    qc.measure_all()

    return qc


# 🔍 Test build (sanity check)
if __name__ == "__main__":
    sample_features = np.array([0.2, 0.6, 0.4, 0.8])
    circuit = build_vqc(sample_features)
    print(circuit.draw())