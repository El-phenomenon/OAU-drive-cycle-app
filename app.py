"""
OAU Drive Cycle Energy Simulator (single reference drive cycle)
- Uses drivecycle/data/final_drive_cycle.csv as the reference cycle (no upload).
- Supports EV (physics + EV surrogates) and ICE (physics + placeholder surrogates).
- Estimator tab computes reasonable defaults for users who don't know parameters.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local modules (must exist)
from drivecycle.io import load_cycle
from drivecycle.simulation import integrate_energy_for_cycle
from drivecycle.models import predict_pce, predict_dnn
from drivecycle.sensitivity import run_sobol, plot_sobol

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(page_title="OAU Drive Cycle Energy Simulator", layout="wide")

# ------------------------------------------------------------
# Branding header
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

# Sidebar logos + info
scol1, scol2 = st.sidebar.columns([1, 1])
with scol1:
    st.image("photos/OAU_logo.png", width=50)
with scol2:
    st.image("photos/Mech.png", width=50)

st.sidebar.markdown("""
<div style='text-align:center; font-size:13px; line-height:1.3; color:#004aad;'>
<b>Obafemi Awolowo University</b><br>
Department of Mechanical Engineering
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load reference drive cycle
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVE_CYCLE_PATH = os.path.join(BASE_DIR, "data", "final_drive_cycle.csv")

try:
    cycle_df = load_cycle(DRIVE_CYCLE_PATH)
    st.session_state["reference_cycle_loaded"] = True
except Exception as e:
    cycle_df = None
    st.session_state["reference_cycle_loaded"] = False
    st.error(f"Could not load drive cycle: {e}")

# ------------------------------------------------------------
# Helper: ICE physics model
# ------------------------------------------------------------
def integrate_fuel_for_cycle(cycle_df, params):
    rho_air = 1.225
    g = 9.81
    LHV_J_per_L = 34.2e6
    CO2_g_per_L = 2392.0

    t = cycle_df["time_s"].values
    v = cycle_df["speed_m_s"].values
    dt = np.diff(t, append=t[-1] + (t[-1]-t[-2] if len(t)>1 else 1.0))
    mass = params.get("MASS", 1500)
    RRC = params.get("RRC", 0.01)
    Cd = params.get("Cd", 0.35)
    A = 2.2
    hw = params.get("HW", 0.0)
    aux_w = params.get("AUX_kW", 0.0) * 1000
    engine_eff = max(0.01, params.get("Engine_Eff", 30.0) / 100)
    idle_fuel_lph = params.get("Idle_Fuel_Lph", 0.8)

    total_fuel_l, distance_m = 0, 0
    for i in range(len(v)):
        vi = max(0.0, v[i])
        ai = 0 if i == 0 else (vi - v[i-1]) / (dt[i] if dt[i] > 0 else 1)
        v_air = max(0.0, vi + hw)

        F_inertia = mass * ai
        F_roll = mass * g * RRC
        F_aero = 0.5 * rho_air * Cd * A * (v_air ** 2)
        F_trac = F_inertia + F_roll + F_aero
        P_wheel = F_trac * vi

        if P_wheel >= 0:
            driveline_eff = 0.9
            P_engine = (P_wheel / driveline_eff) + aux_w
            fuel_power_Js = P_engine / engine_eff
            total_fuel_l += (fuel_power_Js / LHV_J_per_L) * dt[i]
        else:
            total_fuel_l += (idle_fuel_lph / 3600) * dt[i]

        distance_m += vi * dt[i]

    distance_km = distance_m / 1000
    fuel_l_per_100km = (total_fuel_l / distance_km * 100) if distance_km > 0 else np.nan
    co2_g_per_km = (total_fuel_l * CO2_g_per_L / distance_km) if distance_km > 0 else np.nan
    return total_fuel_l, fuel_l_per_100km, co2_g_per_km, distance_km

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tabs = st.tabs(["🧮 Physics Model", "🤖 Surrogate Models", "📊 Sensitivity Analysis"])

# ------------------------------------------------------------
# TAB 1 - Physics model
# ------------------------------------------------------------
with tabs[0]:
    st.header("🧮 Physics Model (OAU reference drive cycle)")
    fuel_type = st.selectbox("Select Fuel Type", ["Electric Vehicle (EV)", "Internal Combustion Engine (ICE)"])

    if not st.session_state["reference_cycle_loaded"]:
        st.error("Drive cycle not loaded.")
    else:
        st.markdown("Using reference drive cycle from OAU dataset")

        cols = st.columns(2)
        with cols[0]:
            MASS = st.number_input("Mass (kg)", 500.0, 8000.0, 3300.0)
            RRC = st.number_input("Rolling Resistance", 0.005, 0.02, 0.01)
            HW = st.number_input("Headwind (m/s)", -5.0, 5.0, 0.0)
            AUX_kW = st.number_input("Aux Load (kW)", 0.0, 10.0, 1.0)
        with cols[1]:
            if fuel_type == "Electric Vehicle (EV)":
                Tb = st.number_input("Battery Temp (°C)", 0.0, 60.0, 25.0)
                SoC_pct = st.number_input("State of Charge (%)", 0.0, 100.0, 80.0)
                BAge_pct = st.number_input("Battery Age (%)", 0.0, 50.0, 5.0)
                MR_mOhm = st.number_input("Motor Resistance (mΩ)", 10.0, 100.0, 55.0)
                BR_pct = st.number_input("Battery Resistance Growth (%)", 50.0, 150.0, 100.0)
            else:
                Cd = st.number_input("Drag Coefficient (Cd)", 0.2, 0.6, 0.35)
                Engine_Eff = st.number_input("Engine Efficiency (%)", 5.0, 60.0, 30.0)
                Idle_Fuel_Lph = st.number_input("Idle Fuel (L/h)", 0.0, 5.0, 0.8)

        if st.button("Run Physics Simulation"):
            if fuel_type == "Electric Vehicle (EV)":
                params = {"MASS": MASS, "RRC": RRC, "HW": HW, "AUX_kW": AUX_kW,
                          "Tb": Tb, "SoC_pct": SoC_pct, "BAge_pct": BAge_pct,
                          "MR_mOhm": MR_mOhm, "BR_pct": BR_pct}
                result = integrate_energy_for_cycle(cycle_df, params)
                energy_kwh_per_km = result["energy_kwh_per_km"]
                regen_pct = result["regen_pct"]
                distance_km = result["distance_km"]
                st.success("EV Physics simulation complete.")
                st.metric("Energy (kWh/km)", f"{energy_kwh_per_km:.3f}")
                st.metric("Regeneration (%)", f"{regen_pct:.2f}")
                st.metric("Distance (km)", f"{distance_km:.2f}")
            else:
                params = {"MASS": MASS, "RRC": RRC, "HW": HW, "AUX_kW": AUX_kW,
                          "Cd": Cd, "Engine_Eff": Engine_Eff, "Idle_Fuel_Lph": Idle_Fuel_Lph}
                total_fuel_l, fuel_l_per_100km, co2_g_per_km, distance_km = integrate_fuel_for_cycle(cycle_df, params)
                st.success("ICE Physics simulation complete.")
                st.metric("Fuel (L/100 km)", f"{fuel_l_per_100km:.3f}")
                st.metric("CO₂ (g/km)", f"{co2_g_per_km:.1f}")
                st.metric("Distance (km)", f"{distance_km:.2f}")

        # Plot reference cycle
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
    if fuel_type == "Electric Vehicle (EV)":
        input_df = pd.DataFrame([{
            "MASS": MASS, "HW": HW, "RRC": RRC,
            "Ta": 25.0, "Tb": Tb, "SoC_pct": SoC_pct,
            "BAge_pct": BAge_pct, "MR_mOhm": MR_mOhm,
            "AUX_kW": AUX_kW, "BR_pct": BR_pct
        }])
        if st.button("Run EV Surrogates (PCE & DNN)"):
            try:
                pce_out = predict_pce(input_df)
                dnn_out = predict_dnn(input_df)
                st.write("*PCE Output:*", pce_out.to_dict(orient="records")[0])
                st.write("*DNN Output:*", dnn_out.to_dict(orient="records")[0])
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.info("ICE surrogate models are not yet trained.")

# ------------------------------------------------------------
# TAB 3 - Sensitivity
# ------------------------------------------------------------
with tabs[2]:
    st.header("📊 Sensitivity Analysis (EV only)")
    model_choice = st.selectbox("Select surrogate model", ["PCE", "DNN"])
    N = st.slider("Number of Sobol Samples", 128, 1024, 512, step=128)

    if st.button("Run Sensitivity Analysis"):
        try:
            energy_df, regen_df = run_sobol(model_choice.lower(), N)
            st.subheader("Energy Consumption (Top 4 Factors)")
            fig1 = plot_sobol(energy_df.head(4), f"{model_choice} - Energy")
            st.pyplot(fig1)

            st.subheader("Regeneration Efficiency (Top 4 Factors)")
            fig2 = plot_sobol(regen_df.head(4), f"{model_choice} - Regen")
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