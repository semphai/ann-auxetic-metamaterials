import numpy as np
import streamlit as st

# Sayfa Yapılandırması ve Başlık
st.set_page_config(
    page_title="Auxetic Honeycomb Property Predictor", layout="centered"
)

st.title("3D Re-entrant Auxetic Honeycomb Surrogate Model")
st.write(
    "Predict the homogenized Poisson's ratio (ν) and Elastic Modulus (E) "
    "using the representative Bayesian Regularized ANN model (Seed 123, Fold 1)."
)

try:
    # --- 1. KULLANICI GİRDİLERİ ---
    st.header("Input Parameters")

    Dc = st.number_input(
        "Diagonal Bar Diameter (Dc, mm)",
        min_value=0.5,
        max_value=2.0,
        value=0.8,
        step=0.1,
    )
    Dd = st.number_input(
        "Vertical Bar Diameter (Dd, mm)",
        min_value=0.5,
        max_value=2.0,
        value=0.8,
        step=0.1,
    )
    theta = st.slider(
        "Diagonal Angle (θ, degrees)",
        min_value=30.0,
        max_value=80.0,
        value=45.0,
        step=1.0,
    )
    E_mat_raw = st.number_input(
        "Parent Material Elastic Modulus (E_mat, MPa)",
        min_value=1000.0,
        max_value=300000.0,
        value=70000.0,
        step=1000.0,
    )

    # --- 2. ARKA PLAN DÖNÜŞÜMÜ ---
    log_E_mat = np.log(E_mat_raw)

    # --- 3. MODEL AĞIRLIKLARI VE BİASLARI ---
    W1 = np.array([
        [-0.3352, 0.3073, -0.4689, 0.1003],
        [-0.6075, 0.0042, -2.3060, 0.1805],
        [-4.0915, -6.0534, -4.8353, 0.7314],
        [-1.8252, 1.2540, 1.7344, -0.1570],
    ])

    b1 = np.array([0.2005, 0.0972, -2.2471, 0.5844])

    W2 = np.array([
        [-1.9238, 1.3427, -3.4984, -2.8834],
        [2.1866, -1.8956, 0.4871, -1.8641],
    ])

    b2 = np.array([-0.1901, 0.9835])

    # --- 4. GİRDİ VEKTÖRÜ ---
    input_vector = np.array([Dc, Dd, theta, log_E_mat])

    # --- 5. TAHMİN ---
    if st.button("Predict Properties", type="primary"):
        hidden_output = np.tanh(np.dot(W1, input_vector) + b1)
        predictions = np.dot(W2, hidden_output) + b2

        pred_poisson = predictions[0]
        pred_elastic = predictions[1]

        st.success("Prediction Completed Successfully!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Predicted Poisson's Ratio (ν)",
                value=f"{pred_poisson:.4f}",
            )
        with col2:
            st.metric(
                label="Predicted Elastic Modulus (E)",
                value=f"{pred_elastic:.2f} MPa",
            )

except Exception as e:
    st.error(f"An error occurred during execution: {e}")
# ==========================
# DISPLAY FOOTER / NOTE
# ==========================
st.markdown("""
---
**Note:**  
This code performs predictions using the best model from the study titled "A Neural Network Surrogate Model for 3D Re-entrant Auxetic Metamaterials".  
The study was conducted at Atatürk University, and the authors are:  
- Mehmet Özyazıcıoğlu¹  
- Bilal Usanmaz¹  
- Ayşe Gül¹  
- Süleyman N. Orhan²  

¹ Faculty of Engineering, Atatürk University, Türkiye  
² Faculty of Engineering, Erzurum Technical University, Türkiye
""")

