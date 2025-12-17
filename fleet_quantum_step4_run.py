import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
# 2. Small subset (QML rule)
# ===============================
X_q, _, y_q, _ = train_test_split(
    X, y,
    train_size=300,
    stratify=y,
    random_state=42
)

print("Quantum training samples:", len(X_q))

# ===============================
# 3. Build quantum model
# ===============================
num_qubits = X_q.shape[1]

feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1)
ansatz = RealAmplitudes(num_qubits=num_qubits, reps=1, entanglement="linear")
optimizer = COBYLA(maxiter=20)
sampler = Sampler()

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler
)

# ===============================
# 4. Train
# ===============================
print("⏳ Training VQC...")
vqc.fit(X_q, y_q)
print("✅ Training finished")

# ===============================
# 5. Evaluate
# ===============================
y_pred = vqc.predict(X_q)
acc = accuracy_score(y_q, y_pred)
print("Quantum Training Accuracy:", round(acc, 4))

# ===============================
# 6. SAVE ONLY TRAINED PARAMETERS (IMPORTANT)
# ===============================
model_data = {
    "weights": vqc._fit_result.x,     # trained parameters
    "num_qubits": num_qubits
}

with open("trained_vqc_weights.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("✅ Trained quantum weights saved (trained_vqc_weights.pkl)")
