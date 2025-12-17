from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import numpy as np

# ===============================
# Parameters
# ===============================
num_qubits = 4

# Example encoded feature angles (scaled features)
x = [
    np.pi / 5,
    3 * np.pi / 5,
    2 * np.pi / 5,
    4 * np.pi / 5
]

# Trainable parameters (θ)
theta = [f"θ[{i}]" for i in range(num_qubits)]

# ===============================
# Build lecture-style VQC
# ===============================
qc = QuantumCircuit(num_qubits)

# 1️⃣ Hadamard → superposition
for i in range(num_qubits):
    qc.h(i)

# 2️⃣ Feature encoding Ry(x)
for i in range(num_qubits):
    qc.ry(x[i], i)

# 3️⃣ Trainable Ry(θ)
for i in range(num_qubits):
    qc.ry(0, i)   # placeholder angle for visualization
    qc.data[-1][0].params[0] = theta[i]  # label parameter

# 4️⃣ Entanglement (linear CNOT chain)
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)

# 5️⃣ Measurement
qc.measure_all()

# ===============================
# Export circuit
# ===============================
circuit_drawer(
    qc,
    output="mpl",
    filename="lecture_style_vqc.png",
    fold=-1
)

print("✅ lecture_style_vqc.png created")
