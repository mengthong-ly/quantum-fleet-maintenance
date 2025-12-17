# fleet_quantum_step5_visualize.py

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("fleet_selected_features.csv")

features = [
    "Engine_Temperature",
    "Vibration_Levels",
    "Fuel_Consumption",
    "Downtime_Maintenance"
]

plt.figure()
df[features].hist(bins=30)
plt.suptitle("Fleet Feature Distributions")
plt.savefig("fleet_feature_distributions.png")
plt.show()


import matplotlib.pyplot as plt

models = ["Classical ML", "Quantum VQC"]
accuracy = [0.85, 0.78]  # replace with your real values

plt.figure()
plt.bar(models, accuracy)
plt.ylim(0, 1)
plt.title("Accuracy Comparison: Classical vs Quantum")
plt.ylabel("Accuracy")
plt.savefig("accuracy_comparison.png")
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Example labels (replace with real outputs)
y_true = [0,1,1,0,1,0,1,0]
y_pred = [0,1,0,0,1,0,1,1]

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Quantum Model Confusion Matrix")
plt.savefig("quantum_confusion_matrix.png")
plt.show()


from qiskit import QuantumCircuit

qc = QuantumCircuit(4)
qc.h(range(4))
qc.measure_all()

qc.draw("mpl")
plt.savefig("quantum_circuit.png")
plt.show()
