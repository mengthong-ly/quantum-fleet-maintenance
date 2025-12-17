import pandas as pd
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_algorithms.optimizers import COBYLA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================
# Load prepared quantum data
# ============================
X_train = pd.read_csv("X_quantum_train.csv").values
X_test  = pd.read_csv("X_quantum_test.csv").values
y_train = pd.read_csv("y_quantum_train.csv").values.ravel()
y_test  = pd.read_csv("y_quantum_test.csv").values.ravel()

num_qubits = X_train.shape[1]
print("Number of qubits:", num_qubits)

# ============================
# Quantum Feature Map
# ============================
feature_map = ZZFeatureMap(
    feature_dimension=num_qubits,
    reps=1
)

# ============================
# Variational Ansatz
# ============================
ansatz = TwoLocal(
    num_qubits=num_qubits,
    rotation_blocks="ry",
    entanglement_blocks="cz",
    reps=2
)

# ============================
# Optimizer
# ============================
optimizer = COBYLA(maxiter=50)

# ============================
# Build VQC (NO estimator arg)
# ============================
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer
)

# ============================
# Train
# ============================
print("⏳ Training Quantum VQC...")
vqc.fit(X_train, y_train)
print("✅ Quantum training finished")

# ============================
# Predict
# ============================
y_pred = vqc.predict(X_test)

# ============================
# Evaluation
# ============================
print("\nQuantum Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
