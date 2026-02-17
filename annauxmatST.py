import streamlit as st
import numpy as np
from scipy.io import loadmat

# ==========================================
# LOAD MODEL
# ==========================================
data = loadmat("R_seed123.mat", struct_as_record=False, squeeze_me=True)
R = data["RR"]

# ==========================================
# NETWORK WEIGHTS & BIAS
# ==========================================
for w in R.net.IW:
    if w.size > 0:
        IW = np.array(w)
        break

for w in R.net.LW.flatten():
    if w.size > 0:
        LW = np.array(w)
        break

b1 = np.array(R.net.b[0]).reshape(-1)
b2 = np.array(R.net.b[1]).reshape(-1)

# ==========================================
# OTHER PARAMETERS
# ==========================================
muX = np.array(R.muX)
sigX = np.array(R.sigX)

muY = np.array(R.muY)
sigY = np.array(R.sigY)

best_transform = int(R.best_transform)

nMat = R.X_train_raw_clean.shape[1] - 7  # number of material dummy vars

# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================
st.title("ANN Model Prediction")

P = st.number_input("1. Point Load P [kN]", value=0.61)
theta = st.number_input("2. Diagonal Angle θ [°]", value=75)
material = st.selectbox("3. Material Type m", options=[1,2,3,4], index=1)
A = st.number_input("4. Total Cross-Sectional Area A [mm²]", value=5971.271)
a = st.number_input("5. Vertical Bar Area a [mm²]", value=0.1583)
V = st.number_input("6. Volume V [mm³]", value=594595.288)
D_c = st.number_input("7. Diagonal Bar Diameter D_c [mm]", value=0.8)
D_d = st.number_input("8. Vertical Bar Diameter D_d [mm]", value=0.9)

if st.button("Predict"):
    # Log-transform
    i05 = np.log(A)
    i07 = np.log(V)

    # Material one-hot
    mat_dummy = np.zeros(nMat)
    mat_dummy[material-1] = 1

    # Input vector
    X_raw = np.concatenate(([D_c, D_d, a, P, theta, i05, i07], mat_dummy))

    # Normalization
    Xn = (X_raw - muX) / sigX

    # Forward pass
    def tansig(x):
        return 2/(1+np.exp(-2*x)) - 1

    z1 = IW @ Xn + b1
    h = tansig(z1)
    Y_n = LW @ h + b2

    # Denormalization
    Y_t = Y_n * sigY + muY
    Y_real = Y_t.copy()

    if best_transform == 2:
        Y_real[1] = np.exp(Y_t[1])
    elif best_transform == 3:
        Y_real[1] = Y_t[1]**2
    elif best_transform == 4:
        Y_real[1] = Y_t[1]**3

    st.success("Prediction Complete")
    st.write(f"Poisson’s Ratio($\nu$) = {Y_real[0]:.6f}")
    st.write(f"Elasticity Modulus (E)  = {Y_real[1]:.3f}")
    
    
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

