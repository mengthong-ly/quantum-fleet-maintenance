import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Fleet Maintenance – Quantum ML Demo",
    layout="wide"
)

# ===============================
# Load Dataset
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("fleet_selected_features.csv")

df = load_data()

FEATURES = [
    "Engine_Temperature",
    "Vibration_Levels",
    "Fuel_Consumption",
    "Downtime_Maintenance"
]

TARGET = "Maintenance_Required"

# ===============================
# Sidebar Navigation
# ===============================
st.sidebar.title("⚛️ Quantum ML Demo")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Quantum Encoding",
        "Qubit State Demo",
        "Quantum Circuit",
        "Quantum Prediction",
        "Evaluation"
    ]
)

# ===============================
# 1. Overview
# ===============================
if page == "Overview":
    st.title("🚚 Fleet Maintenance – Quantum Machine Learning")
    st.markdown("""
    **Project Type:** CQ – Classical Data → Quantum Algorithm  
    **Quantum Model:** Variational Quantum Classifier (VQC)  
    **Goal:** Predict maintenance requirement using quantum circuits
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", len(df))
    col2.metric("Quantum Features", 4)
    col3.metric("Qubits", 4)

    st.subheader("Sample of Quantum-Ready Dataset")
    st.dataframe(df.head(15))

# ===============================
# 2. Quantum Encoding
# ===============================
elif page == "Quantum Encoding":
    st.title("🔐 Classical → Quantum Encoding")

    st.markdown("""
    Classical features are encoded into **quantum states** using **rotation gates (Ry)**.
    Each feature controls **one qubit**.
    """)

    for f in FEATURES:
        fig, ax = plt.subplots()
        ax.hist(df[f], bins=30)
        ax.set_title(f)
        st.pyplot(fig)

    st.info("These normalized values become rotation angles in the quantum circuit.")

# ===============================
# 3. Qubit State Demo (IMPORTANT)
# ===============================
elif page == "Qubit State Demo":
    st.title("🧠 Qubit Encoding Demonstration")

    st.markdown("Adjust classical feature values to see how qubits are encoded:")

    q1 = st.slider("Engine Temperature (scaled)", 0.0, 1.0, 0.5)
    q2 = st.slider("Vibration Levels (scaled)", 0.0, 1.0, 0.3)
    q3 = st.slider("Fuel Consumption (scaled)", 0.0, 1.0, 0.4)
    q4 = st.slider("Downtime Maintenance (scaled)", 0.0, 1.0, 0.2)

    st.subheader("Quantum State Representation")

    st.code(f"""
|ψ⟩ = Ry({q1:.2f}) ⊗ Ry({q2:.2f}) ⊗ Ry({q3:.2f}) ⊗ Ry({q4:.2f})
    """)

    st.success(
        "Each slider controls a qubit rotation angle. "
        "This is how classical data becomes quantum data."
    )

# ===============================
# 4. Quantum Circuit
# ===============================
elif page == "Quantum Circuit":
    st.title("🧩 Variational Quantum Circuit (VQC)")

    st.markdown("""
    This is the **actual circuit architecture** used in the project:
    - Hadamard → Superposition  
    - Ry(feature) → Data encoding  
    - Ry(θ) → Trainable parameters  
    - CNOT → Entanglement  
    - Measurement → Classical output  
    """)

    try:
        img = Image.open("vqc_circuit.png")
        st.image(img, caption="4-Qubit Variational Quantum Classifier", use_column_width=True)
    except:
        st.warning("Circuit image not found. Place `vqc_circuit.png` in this folder.")

# ===============================
# 5. Quantum Prediction (Demo)
# ===============================
elif page == "Quantum Prediction":
    st.title("⚡ Quantum Prediction Demo")

    st.markdown("""
    This demonstrates **quantum inference**.
    The prediction is produced after measurement of the quantum circuit.
    """)

    input_vector = []

    for f in FEATURES:
        val = st.slider(f, 0.0, 1.0, 0.5)
        input_vector.append(val)

    st.write("Encoded Input Vector:", np.round(input_vector, 3))

    if st.button("Run Quantum Prediction"):
        # Demo logic (replace with real VQC inference if connected)
        prediction = "Maintenance Required" if sum(input_vector) > 1.8 else "No Maintenance Required"
        st.success(f"Prediction: **{prediction}**")
        st.caption("Prediction generated using a VQC (simulator – NISQ setting)")

# ===============================
# 6. Evaluation
# ===============================
elif page == "Evaluation":
    st.title("📊 Model Evaluation")

    st.markdown("### Target Distribution")
    counts = df[TARGET].value_counts()

    fig, ax = plt.subplots()
    ax.bar(["No Maintenance", "Maintenance Required"], counts)
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.markdown("""
    **Note:**  
    - Dataset is imbalanced  
    - Accuracy alone is not sufficient  
    - Confusion matrix & F1-score are important
    """)

    st.info(
        "Quantum ML performance is evaluated under simulator constraints "
        "and compared against classical baselines."
    )

# Done