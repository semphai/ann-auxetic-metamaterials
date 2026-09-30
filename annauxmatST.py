import numpy as np
import streamlit as st

# Sayfa Yapılandırması ve Başlık
st.set_page_config(
    page_title="Auxetic Honeycomb Property Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("3D Re-entrant Auxetic Honeycomb Surrogate Model")
st.markdown(
    "Predict the homogenized **Poisson's ratio ($\nu$)** and **Elastic Modulus ($E$)** "
    "using the representative Bayesian Regularized ANN model (Seed 123, Fold 1)."
)
st.markdown("---")

# --- 1. KULLANICI GİRDİLERİ (Sidebar) ---
st.sidebar.header("Input Parameters")

Dc = st.sidebar.number_input(
    "Diagonal Bar Diameter (Dc, mm)",
    min_value=0.5,
    max_value=2.0,
    value=0.8,
    step=0.1,
    format="%.2f",
)
Dd = st.sidebar.number_input(
    "Vertical Bar Diameter (Dd, mm)",
    min_value=0.5,
    max_value=2.0,
    value=0.8,
    step=0.1,
    format="%.2f",
)
theta = st.sidebar.slider(
    "Diagonal Angle (θ, degrees)",
    min_value=30.0,
    max_value=80.0,
    value=45.0,
    step=1.0,
)
E_mat_raw = st.sidebar.number_input(
    "Parent Material Elastic Modulus (E_mat, MPa)",
    min_value=1000.0,
    max_value=300000.0,
    value=70000.0,
    step=1000.0,
    format="%.1f",
)

# Predict Butonu Sidebar'a taşındı
predict_button = st.sidebar.button(
    "Predict Properties", type="primary", use_container_width=True
)

# --- 2. MODEL AĞIRLIKLARI VE BİASLARI ---
# Not: Eğer modelinizi eğitirken input/output normalizasyonu (mapstd/mapminmax) 
# kullandıysanız, buraya ilgili ortalama/standart sapma değerlerini eklemeniz gerekebilir.
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

# --- 3. ANA SAYFA GÖSTERİMİ VE HESAPLAMA ---
col_main1, col_main2 = st.columns([1, 1])

with col_main1:
    st.subheader("Selected Configuration Summary")
    st.write(f"- **Diagonal Bar Diameter ($D_c$):** {Dc} mm")
    st.write(f"- **Vertical Bar Diameter ($D_d$):** {Dd} mm")
    st.write(f"- **Diagonal Angle ($\theta$):** {theta}°")
    st.write(f"- **Parent Material Modulus ($E_{{mat}}$):** {E_mat_raw:,.1f} MPa")

with col_main2:
    st.subheader("Prediction Results")
    
    if predict_button:
        try:
            # Arka plan dönüşümü
            log_E_mat = np.log(E_mat_raw)
            input_vector = np.array([Dc, Dd, theta, log_E_mat])

            # İleri besleme (Feedforward hesaplama)
            hidden_output = np.tanh(np.dot(W1, input_vector) + b1)
            predictions = np.dot(W2, hidden_output) + b2

            pred_poisson = predictions[0]
            pred_elastic = predictions[1]

            st.success("Prediction Completed Successfully!")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(
                    label="Poisson's Ratio (ν)",
                    value=f"{pred_poisson:.4f}",
                )
            with res_col2:
                st.metric(
                    label="Elastic Modulus (E)",
                    value=f"{pred_elastic:.2f} MPa",
                )
                
        except Exception as e:
            st.error(f"An error occurred during calculation: {e}")
    else:
        st.info("Adjust the parameters in the sidebar and click **'Predict Properties'** to see the results.")

# ==========================
# FOOTER / BİLGİ ALANI
# ==========================
st.markdown("""
---
**Note:**  
This application performs predictions using the surrogate model from the study titled *"A Neural Network Surrogate Model for 3D Re-entrant Auxetic Metamaterials"*.  
**Research Institutions:**  
- ¹ Faculty of Engineering, Atatürk University, Türkiye  
- ² Faculty of Engineering, Erzurum Technical University, Türkiye  
""")
