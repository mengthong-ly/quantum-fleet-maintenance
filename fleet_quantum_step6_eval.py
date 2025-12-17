import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_aer.primitives import Sampler


# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv("fleet_selected_features.csv")

X = df.drop("Maintenance_Required", axis=1).values
y = df["Maintenance_Required"].values

# ===============================
# 2. Small test subset (QML rule)
# ===============================
_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=300,
    stratify=y,
    random_state=42
)

print("Evaluating on", len(X_test), "samples")

# ===============================
# 3. Load trained quantum weights
# ===============================
with open("trained_vqc_weights.pkl", "rb") as f:
    model_data = pickle.load(f)

weights = model_data["weights"]
num_qubits = model_data["num_qubits"]

# ===============================
# 4. Rebuild VQC architecture
# ===============================
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1)

ansatz = RealAmplitudes(
    num_qubits=num_qubits,
    reps=1,
    entanglement="linear"
)

optimizer = COBYLA(maxiter=1)
sampler = Sampler()

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler
)

# ===============================
# 5. Inject trained weights
# ===============================
vqc._fit_result = type("", (), {})()
vqc._fit_result.x = weights

# ===============================
# 6. Quantum inference
# ===============================
raw_pred = np.array(vqc.predict(X_test))

# FIX: convert to class labels correctly
if raw_pred.ndim == 2:
    y_pred = np.argmax(raw_pred, axis=1)
else:
    y_pred = raw_pred.astype(int)

# ===============================
# 7. Confusion matrix
# ===============================
cm = confusion_matrix(y_test, y_pred)

np.save("quantum_confusion_matrix.npy", cm)

print("Saved quantum_confusion_matrix.npy")
print("Confusion Matrix:")
print(cm)
