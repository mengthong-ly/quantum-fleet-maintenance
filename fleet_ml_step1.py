import pandas as pd

# Load dataset
df = pd.read_csv("Dataset/logistics_dataset_with_maintenance_required.csv")

# Basic inspection
print(df.head())
print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset shape:", df.shape)

