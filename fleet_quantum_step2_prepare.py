import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load encoded dataset
df = pd.read_csv("fleet_encoded.csv")

# Quantum features (4 qubits)
quantum_features = [
    "Engine_Temperature",
    "Vibration_Levels",
    "Fuel_Consumption",
    "Downtime_Maintenance"
]

X = df[quantum_features]
y = df["Maintenance_Required"]

# Normalize to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Small subset for quantum
X_small, _, y_small, _ = train_test_split(
    X_scaled,
    y,
    train_size=400,
    stratify=y,
    random_state=42
)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_small,
    y_small,
    test_size=0.25,
    stratify=y_small,
    random_state=42
)

# 🔹 SAVE FILES (THIS WAS MISSING)
pd.DataFrame(X_train, columns=quantum_features).to_csv("X_quantum_train.csv", index=False)
pd.DataFrame(X_test, columns=quantum_features).to_csv("X_quantum_test.csv", index=False)
pd.DataFrame(y_train, columns=["Maintenance_Required"]).to_csv("y_quantum_train.csv", index=False)
pd.DataFrame(y_test, columns=["Maintenance_Required"]).to_csv("y_quantum_test.csv", index=False)

print("Quantum Train Shape:", X_train.shape)
print("Quantum Test Shape:", X_test.shape)
print("\nTrain Label Distribution:")
print(y_train.value_counts())
