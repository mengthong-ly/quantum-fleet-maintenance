# import pandas as pd

# # Load ENCODED dataset
# df = pd.read_csv("fleet_encoded.csv")
# head =df.head()

# # Target
# y = df["Maintenance_Required"]

# # Selected quantum features (4 qubits)
# quantum_features = [
#     "Engine_Temperature",
#     "Vibration_Levels",
#     "Fuel_Consumption",
#     "Downtime_Maintenance"
# ]
# print(df[quantum_features].describe())
# # Select features
# X_quantum = df[quantum_features]

# print("Quantum Feature Matrix Shape:", X_quantum.shape)
# print("Selected Features:", quantum_features)
# print("\nTarget Distribution:")
# print(y.value_counts())


# # ===============================
# # Save selected quantum features
# # ===============================

# output_df = df[[
#     "Engine_Temperature",
#     "Vibration_Levels",
#     "Fuel_Consumption",
#     "Downtime_Maintenance",
#     "Maintenance_Required"
# ]]

# output_df.to_csv("fleet_selected_features.csv", index=False)

# print("Saved fleet_selected_features.csv")



import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ===============================
# 1. Load encoded dataset
# ===============================
df = pd.read_csv("fleet_encoded.csv")

print("Dataset loaded successfully")
print("Total samples:", len(df))
print("Total columns:", len(df.columns))
print()

# ===============================
# 2. Define target
# ===============================
target_col = "Maintenance_Required"
y = df[target_col]

print("Target distribution:")
print(y.value_counts())
print()

# ===============================
# 3. Select quantum features (CQ-QML)
# ===============================
# 4 classical features → 4 qubits
quantum_features = [
    "Engine_Temperature",
    "Vibration_Levels",
    "Fuel_Consumption",
    "Downtime_Maintenance"
]

# Safety check
assert all(f in df.columns for f in quantum_features), "Missing quantum features"

X_quantum = df[quantum_features]

print("Selected quantum features:")
print(quantum_features)
print()

# ===============================
# 4. Feature statistics (before scaling)
# ===============================
print("Quantum Feature Statistics (Before Scaling):")
print(X_quantum.describe())
print()

print("Quantum Feature Matrix Shape:", X_quantum.shape)
print()

# ===============================
# 5. Normalize features (important for QML)
# ===============================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_quantum)

X_quantum_scaled = pd.DataFrame(
    X_scaled,
    columns=quantum_features
)

print("Features normalized using MinMaxScaler (0–1 range)")
print()

# ===============================
# 6. Save quantum-ready dataset
# ===============================
output_df = X_quantum_scaled.copy()
output_df[target_col] = y.values

output_file = "fleet_selected_features.csv"
output_df.to_csv(output_file, index=False)

print(f"Saved quantum-ready dataset: {output_file}")
print("Final dataset shape:", output_df.shape)

# ===============================
# 7. Summary (for logging / presentation)
# ===============================
print("\n=== SUMMARY ===")
print("QML Type: CQ (Classical Data → Quantum Algorithm)")
print("Samples:", output_df.shape[0])
print("Quantum Features:", len(quantum_features))
print("Target:", target_col)
print("Ready for VQC / Visualization")
