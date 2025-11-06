import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local modules
from drivecycle.io import load_cycle
from drivecycle.simulation import integrate_energy_for_cycle
from drivecycle.models import predict_pce, predict_dnn
from drivecycle.sensitivity import run_sobol, plot_sobol

# ------------------------------------------------------------
# Page Config
# ------------------------------------------------------------
st.set_page_config(page_title="OAU Drive Cycle Energy Simulator", layout="wide")

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    try:
        st.image("photos/OAU_logo.png", width=70)
    except:
        st.write("")
with col2:
    st.markdown("<h2 style='text-align:center; color:#004aad;'>OAU Drive Cycle Energy Simulator</h2>", unsafe_allow_html=True)
with col3:
    try:
        st.image("photos/Mech.png", width=70)
    except:
        st.write("")
st.markdown("---")

# Sidebar logos
scol1, scol2 = st.sidebar.columns([1, 1])
with scol1:
    st.image("photos/OAU_logo.png", width=50)
with scol2:
    st.image("photos/Mech.png", width=50)
st.sidebar.markdown("""
<div style='text-align:center; font-size:13px; color:#004aad;'>
<b>Obafemi Awolowo University</b><br>Dept. of Mechanical Engineering
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load reference drive cycle
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(_file_))
DRIVE_CYCLE_PATH = os.path.join(BASE_DIR, "data", "final_drive_cycle.csv")

try:
    cycle_df = load_cycle(DRIVE_CYCLE_PATH)
    st.session_state["reference_cycle_loaded"] = True
except Exception as e:
    cycle_df = None
    st.session_state["reference_cycle_loaded"] = False
    st.error(f"Could not load drive cycle: {e}")

# ------------------------------------------------------------
# Sidebar: Select fuel type
# ------------------------------------------------------------
st.sidebar.header("Simulation Setup")
fuel_type = st.sidebar.selectbox("Select Fuel Type", ["Select...", "Electric Vehicle (EV)", "Internal Combustion Engine (ICE)"])
st.session_state["fuel_type"] = fuel_type

# ------------------------------------------------------------
# Helper: ICE Physics Model
# ------------------------------------------------------------
def integrate_fuel_for_cycle(cycle_df, params):
    rho_air = 1.225
    g = 9.81
    LHV_J_per_L = 34.2e6
    CO2_g_per_L = 2392.0
    t = cycle_df["time_s"].values
    v = cycle_df["speed_m_s"].values
    dt = np.diff(t, append=t[-1] + (t[-1]-t[-2] if len(t)>1 else 1.0))

    mass = params["MASS"]; RRC = params["RRC"]; Cd = params["Cd"]
    hw = params["HW"]; aux_w = params["AUX_kW"] * 1000
    eff = max(0.01, params["Engine_Eff"]/100); idle = params["Idle_Fuel_Lph"]

    total_l, dist_m = 0, 0
    for i in range(len(v)):
        vi = v[i]; ai = 0 if i==0 else (v[i]-v[i-1])/(dt[i] if dt[i]>0 else 1)
        F = mass*ai + mass*g*RRC + 0.5*rho_air*Cd*2.2*(vi+hw)**2
        Pw = F*vi
        if Pw >= 0:
            total_l += (Pw/eff/LHV_J_per_L)*dt[i]
        else:
            total_l += (idle/3600)*dt[i]
        dist_m += vi*dt[i]
    dist_km = dist_m/1000
    l_per_100 = (total_l/dist_km*100) if dist_km>0 else np.nan
    co2 = (total_l*CO2_g_per_L/dist_km) if dist_km>0 else np.nan
    return total_l, l_per_100, co2, dist_km

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tabs = st.tabs(["🧮 Physics Model", "🤖 Surrogate Models", "📊 Sensitivity Analysis"])

# ------------------------------------------------------------
# TAB 1 - Physics Model
# ------------------------------------------------------------
with tabs[0]:
    st.header("🧮 Physics Model (OAU Reference Drive Cycle)")

    if fuel_type == "Select...":
        st.info("Please select your fuel type from the sidebar to continue.")
    elif not st.session_state["reference_cycle_loaded"]:
        st.error("Reference drive cycle not loaded.")
    else:
        st.markdown("Using the fixed OAU reference drive cycle for all calculations.")

        # Dynamic form for input fields
        with st.form("physics_inputs"):
            st.subheader(f"{fuel_type} Parameters")
            cols = st.columns(2)
            with cols[0]:
                MASS = st.number_input("Vehicle Mass (kg)", min_value=500.0, max_value=8000.0, value=None, step=10.0)
                RRC = st.number_input("Rolling Resistance (RRC)", min_value=0.005, max_value=0.020, value=None, step=0.001)
                HW = st.number_input("Headwind (m/s)", min_value=-5.0, max_value=5.0, value=None, step=0.1)
                AUX_kW = st.number_input("Auxiliary Load (kW)", min_value=0.0, max_value=10.0, value=None, step=0.1)
            with cols[1]:
                if fuel_type == "Electric Vehicle (EV)":
                    Tb = st.number_input("Battery Temperature (°C)", 0.0, 60.0, value=None)
                    SoC_pct = st.number_input("State of Charge (%)", 0.0, 100.0, value=None)
                    BAge_pct = st.number_input("Battery Age (%)", 0.0, 50.0, value=None)
                    MR_mOhm = st.number_input("Motor Resistance (mΩ)", 10.0, 200.0, value=None)
                    BR_pct = st.number_input("Battery Resistance Growth (%)", 50.0, 200.0, value=None)
                else:
                    Cd = st.number_input("Drag Coefficient (Cd)", 0.2, 0.6, value=None)
                    Engine_Eff = st.number_input("Engine Efficiency (%)", 5.0, 60.0, value=None)
                    Idle_Fuel_Lph = st.number_input("Idle Fuel (L/h)", 0.0, 5.0, value=None)
            submitted = st.form_submit_button("Run Physics Simulation")

        if submitted:
            if fuel_type == "Electric Vehicle (EV)":
                if None in [MASS, RRC, HW, AUX_kW, Tb, SoC_pct, BAge_pct, MR_mOhm, BR_pct]:
                    st.warning("Please fill in all EV parameters before running the simulation.")
                else:
                    params = {
                        "MASS": MASS, "RRC": RRC, "HW": HW, "AUX_kW": AUX_kW,
                        "Tb": Tb, "SoC_pct": SoC_pct, "BAge_pct": BAge_pct,
                        "MR_mOhm": MR_mOhm, "BR_pct": BR_pct
                    }
                    result = integrate_energy_for_cycle(cycle_df, params)
                    st.success("EV Physics Simulation Complete.")
                    st.metric("Energy (kWh/km)", f"{result['energy_kwh_per_km']:.3f}")
                    st.metric("Regen (%)", f"{result['regen_pct']:.2f}")
                    st.metric("Distance (km)", f"{result['distance_km']:.2f}")

            elif fuel_type == "Internal Combustion Engine (ICE)":
                if None in [MASS, RRC, HW, AUX_kW, Cd, Engine_Eff, Idle_Fuel_Lph]:
                    st.warning("Please fill in all ICE parameters before running the simulation.")
                else:
                    params = {
                        "MASS": MASS, "RRC": RRC, "HW": HW, "AUX_kW": AUX_kW,
                        "Cd": Cd, "Engine_Eff": Engine_Eff, "Idle_Fuel_Lph": Idle_Fuel_Lph
                    }
                    total_fuel_l, fuel_l_per_100km, co2_g_per_km, distance_km = integrate_fuel_for_cycle(cycle_df, params)
                    st.success("ICE Physics Simulation Complete.")
                    st.metric("Fuel (L/100 km)", f"{fuel_l_per_100km:.3f}")
                    st.metric("CO₂ (g/km)", f"{co2_g_per_km:.1f}")
                    st.metric("Distance (km)", f"{distance_km:.2f}")

        # Always show the reference cycle
        st.subheader("Reference Drive Cycle")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(cycle_df["time_s"], cycle_df["speed_m_s"], color="blue")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.grid(True)
        st.pyplot(fig)

# ------------------------------------------------------------
# TAB 2 - Surrogates
# ------------------------------------------------------------
with tabs[1]:
    st.header("🤖 Surrogate Models (EV only)")
    if fuel_type != "Electric Vehicle (EV)":
        st.info("Surrogate models are available only for EV.")
    else:
        st.write("Run trained PCE or DNN surrogate models for EV predictions.")
        if st.button("Run EV Surrogates"):
            try:
                st.spinner("Predicting...")
                dummy_input = pd.DataFrame([{"MASS": MASS, "HW": HW, "RRC": RRC, "AUX_kW": AUX_kW,
                                             "Tb": Tb, "SoC_pct": SoC_pct, "BAge_pct": BAge_pct,
                                             "MR_mOhm": MR_mOhm, "BR_pct": BR_pct}])
                pce_out = predict_pce(dummy_input)
                dnn_out = predict_dnn(dummy_input)
                st.write("*PCE Prediction:*", pce_out.to_dict(orient="records")[0])
                st.write("*DNN Prediction:*", dnn_out.to_dict(orient="records")[0])
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ------------------------------------------------------------
# TAB 3 - Sensitivity
# ------------------------------------------------------------
with tabs[2]:
    st.header("📊 Sensitivity Analysis (EV only)")
    if fuel_type != "Electric Vehicle (EV)":
        st.info("Sensitivity analysis is available only for EV models.")
    else:
        model_choice = st.selectbox("Select Model Type", ["PCE", "DNN"])
        N = st.slider("Sobol Sample Size", 128, 2048, 512, step=128)
        if st.button("Run Sensitivity Analysis"):
            try:
                energy_df, regen_df = run_sobol(model_choice.lower(), N)
                st.subheader("Energy Sensitivity (Top 4)")
                fig1 = plot_sobol(energy_df, "Energy Consumption")
                st.pyplot(fig1)
                st.subheader("Regeneration Sensitivity (Top 4)")
                fig2 = plot_sobol(regen_df, "Regeneration Efficiency")
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Sensitivity failed: {e}")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("""
---
<div style='text-align:center; font-size: small; color: grey;'>
Developed by Prof. B.O. Malomo, Blessing Babatope, and Gabriel Oke | Dept. of Mechanical Engineering, OAU, Ile-Ife
</div>
""", unsafe_allow_html=True)