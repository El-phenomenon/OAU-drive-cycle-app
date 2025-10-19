import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Local module imports
from drivecycle.io import load_cycle
from drivecycle.simulation import integrate_energy_for_cycle
from drivecycle.models import predict_pce, predict_dnn, _load_dnn
from drivecycle.sensitivity import run_sobol, plot_sobol

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(page_title="OAU Drive Cycle Energy Simulator", layout="wide")

# ------------------------------------------------------------
# Custom CSS styling
# ------------------------------------------------------------
st.markdown("""
<style>
h1, h2, h3 {
    text-align: center;
    color: #004aad;
}
[data-testid="stMetricValue"] {
    font-size: 26px;
    color: #004aad;
}
section[data-testid="stSidebar"] {
    background-color: #f7f9fc;
}
.stTabs [data-baseweb="tab-list"] {
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Header with logos
# ------------------------------------------------------------
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.image("photos/OAU_logo.png", width=70)
with col2:
    st.markdown(
        "<h2 style='text-align:center; color:#004aad;'>OAU Drive Cycle Energy Simulator</h2>",
        unsafe_allow_html=True
    )
with col3:
    st.image("photos/Mech.png", width=70)

st.markdown("<hr>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Sidebar branding (logos side-by-side)
# ------------------------------------------------------------
scol1, scol2 = st.sidebar.columns([1, 1])
with scol1:
    st.image("photos/OAU_logo.png", width=50)
with scol2:
    st.image("photos/Mech.png", width=50)

st.sidebar.markdown(
    """
    <div style='text-align:center; font-size:13px; line-height:1.3; color:#004aad;'>
    <b>Obafemi Awolowo University</b><br>
    Department of Mechanical Engineering
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# About / Description
# ------------------------------------------------------------
with st.expander("ℹ About this App", expanded=False):
    st.markdown("""
    OAU Drive Cycle Energy Simulator

    Developed from the research project “Evolution of a Novel Drive Cycle for Energy Prediction of EV Vehicles”  
    at Obafemi Awolowo University (OAU), Ile-Ife, Nigeria.

    ---
    ### 🔍 What it does
    - Physics-Based Model (ECB):  
      Estimates energy consumption and regeneration from real drive cycles.
    - Surrogate Models (Polynomial Chaos Expansion (PCE) & Deep Neural Network (DNN)):  
      Provide rapid energy and regeneration predictions using 10 calibrated factors.
    - Sensitivity Analysis:  
      Identifies the most influential input factors using Sobol indices.

    ---
    ### 🧩 Key Features
    - Upload custom drive cycles  
    - Compare ECB, PCE, and DNN models  
    - Visualize top 4 most influential factors  
    - Designed for EV researchers, engineers, and students  

    ---
    Authors: Blessing Babatope, Gabriel Oke, Prof. B.O. Malomo  
    Institution: Department of Mechanical Engineering, OAU, Ile-Ife, Nigeria  
    """)

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tabs = st.tabs([
    "🧮 Physics Model",
    "🤖 Surrogate Models (PCE & DNN)",
    "📊 Sensitivity Analysis"
])

# ============================================================
# TAB 1 — PHYSICS MODEL
# ============================================================
with tabs[0]:
    st.header("🧮 Physics-Based Simulation (ECB Model)")

    st.sidebar.header("Simulation Inputs")

    # 10 major parameters
    mass = st.sidebar.number_input("Vehicle Mass (kg)", 2000, 6000, 3300, step=50)
    hw = st.sidebar.number_input("Headwind, HW (m/s)", -5.0, 5.0, 0.0, step=0.5)
    rrc = st.sidebar.number_input("Rolling Resistance (RRC)", 0.005, 0.02, 0.010, step=0.001)
    ta = st.sidebar.number_input("Ambient Temperature (Ta) (°C)", 10, 40, 25, step=1)
    tb = st.sidebar.number_input("Battery Temperature (Tb) (°C)", 15, 40, 25, step=1)
    soc = st.sidebar.number_input("State of Charge (soc_pct) (%)", 50, 100, 80, step=5)
    bage = st.sidebar.number_input("Battery Age (Bage_pct) (%)", 0, 20, 5, step=1)
    mr = st.sidebar.number_input("Motor Resistance, mr (mΩ)", 40.0, 70.0, 55.0, step=0.5)
    aux = st.sidebar.number_input("Auxiliary Load (Aux_kW)", 0.0, 10.0, 1.0, step=0.1)
    br = st.sidebar.number_input("Battery Resistance Growth (BR_pct) (%)", 80, 150, 100, step=5)

    uploaded_file = st.file_uploader("Upload Drive Cycle CSV", type=["csv"])

    if uploaded_file is not None:
        df = load_cycle(uploaded_file)
        st.success(f"✅ Drive cycle loaded successfully ({len(df)} samples).")

        with st.expander("📊 View Drive Cycle Data"):
            st.dataframe(df.head())

        if st.button("Run Simulation"):
            st.info("Running physics-based ECB simulation...")

            params = {
                "MASS": mass, "HW": hw, "RRC": rrc, "Ta": ta, "Tb": tb,
                "SoC_pct": soc, "BAge_pct": bage, "MR_mOhm": mr,
                "AUX_kW": aux, "BR_pct": br,
            }

            result = integrate_energy_for_cycle(df, params)

            st.subheader("📈 Results Summary")
            cols = st.columns(3)
            cols[0].metric("Energy (kWh/km)", f"{result['energy_kwh_per_km']:.3f}")
            cols[1].metric("Regeneration (%)", f"{result['regen_pct']:.2f}")
            cols[2].metric("Distance (km)", f"{result['distance_km']:.2f}")

            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(df["time_s"], df["speed_m_s"], color="blue", label="Speed (m/s)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Speed (m/s)")
            ax.set_title("Drive Cycle Speed Profile")
            ax.grid(True)
            st.pyplot(fig)

            st.success("✅ Simulation complete.")
    else:
        st.warning("Please upload a drive cycle CSV to begin.")

# ============================================================
# TAB 2 — SURROGATE MODELS
# ============================================================
with tabs[1]:
    st.header("🤖 Surrogate Predictions (PCE & DNN)")

    cols = st.columns(2)
    with cols[0]:
        mass = st.number_input("MASS (kg)", 2500.0, 6000.0, 3300.0)
        hw = st.number_input("HW (m/s)", -5.0, 5.0, 0.0)
        rrc = st.number_input("RRC", 0.006, 0.012, 0.010)
        ta = st.number_input("Ta (°C)", 10.0, 35.0, 25.0)
        tb = st.number_input("Tb (°C)", 18.0, 35.0, 25.0)
    with cols[1]:
        soc = st.number_input("SoC_pct (%)", 50.0, 100.0, 80.0)
        bage = st.number_input("BAge_pct (%)", 0.0, 20.0, 0.0)
        mr = st.number_input("MR_mOhm", 50.0, 58.0, 55.0)
        aux = st.number_input("AUX_kW", 1.0, 5.0, 1.0)
        br = st.number_input("BR_pct (%)", 100.0, 150.0, 100.0)

    input_df = pd.DataFrame([{
        "MASS": mass, "HW": hw, "RRC": rrc, "Ta": ta, "Tb": tb,
        "SoC_pct": soc, "BAge_pct": bage, "MR_mOhm": mr,
        "AUX_kW": aux, "BR_pct": br
    }])

    dnn_available = _load_dnn() is not None

    if st.button("Run Surrogate Predictions"):
        with st.spinner("Running PCE and DNN predictions..."):
            pce_out = predict_pce(input_df)
            dnn_out = predict_dnn(input_df) if dnn_available else None

        st.subheader("Predicted Outputs")
        st.write("PCE Output:", pce_out.to_dict(orient="records")[0])
        if dnn_available:
            st.write("DNN Output:", dnn_out.to_dict(orient="records")[0])
        else:
            st.warning("⚠ DNN model unavailable on this environment. Showing only PCE results.")

# ============================================================
# ============================================================
# TAB 3 — SENSITIVITY ANALYSIS
# ============================================================
with tabs[2]:
    st.header("📊 Global Sensitivity Analysis")

    from drivecycle.models import _load_dnn

    # Try to load DNN model quietly
    dnn_available = _load_dnn() is not None

    if dnn_available:
        model_choice = st.selectbox("Select Model", ["PCE", "DNN"])
    else:
        model_choice = st.selectbox("Select Model", ["PCE"])
        st.info("ℹ DNN model not available. Only PCE sensitivity analysis can be run.")

    N = st.slider("Number of Sobol Samples (power of 2)", 128, 1024, 512, step=128)

    if st.button("Run Sensitivity Analysis"):
        with st.spinner(f"Running Sobol sensitivity for {model_choice}..."):
            energy_df, regen_df = run_sobol(model_choice.lower(), N)

        st.subheader("🔹 Energy Consumption — Top 4 Factors")
        fig1 = plot_sobol(energy_df, f"{model_choice} - Energy Consumption", top_n=4)
        st.pyplot(fig1)

        st.subheader("🔹 Regeneration Efficiency — Top 4 Factors")
        fig2 = plot_sobol(regen_df, f"{model_choice} - Regeneration Efficiency", top_n=4)
        st.pyplot(fig2)

        st.success("✅ Sensitivity analysis complete.")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("""
---
<div style='text-align:center; font-size: small; color: grey;'>
Developed by Prof. B.O. Malomo, Blessing Babatope and Gabriel Oke | Department of Mechanical Engineering, OAU, Ile-Ife
</div>
""", unsafe_allow_html=True)