import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ============================
# 1. Load dataset
# ============================
df = pd.read_csv("Dataset/logistics_dataset_with_maintenance_required.csv")

# ============================
# 2. Encode categorical columns
# ============================
cat_cols = df.select_dtypes(include=["object"]).columns

encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# ============================
# 3. Split features & target
# ============================
X = df.drop("Need_Maintenance", axis=1)
y = df["Need_Maintenance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# 4. Train Random Forest
# ============================
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ============================
# 5. Evaluation
# ============================
y_pred = rf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================
# 6. Feature Importance
# ============================
importances = rf.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10))
