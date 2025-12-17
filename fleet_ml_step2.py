import pandas as pd

# Load raw data
df = pd.read_csv("Dataset/logistics_dataset_with_maintenance_required.csv")

# Target
y = df["Maintenance_Required"]

# Drop non-ML columns
drop_cols = [
    "Vehicle_ID",
    "Make_and_Model",
    "Last_Maintenance_Date"
]

X = df.drop(columns=drop_cols + ["Maintenance_Required"])

# One-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# Combine X + y
fleet_encoded = X_encoded.copy()
fleet_encoded["Maintenance_Required"] = y.values

# SAVE for next steps ✅
fleet_encoded.to_csv("fleet_encoded.csv", index=False)

print("Encoded feature shape:", X_encoded.shape)
print("Target distribution:")
print(y.value_counts())
