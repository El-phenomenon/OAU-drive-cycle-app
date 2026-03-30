import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from drivecycle.io import load_cycle
from drivecycle.simulation import integrate_energy_for_cycle
from drivecycle.models import predict_pce, predict_pce_ice
from drivecycle.sensitivity import run_sobol, plot_sobol

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="OAU Drive Cycle Energy Simulator", layout="wide")

# ------------------------------------------------------------
# HEADER AND LOGOS
# ------------------------------------------------------------
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.image("photos/OAU_logo.png", width=70)
with col2:
    st.markdown(
        "<h2 style='text-align:center; color:#004aad;'>OAU Drive Cycle Energy Simulator</h2>",
        unsafe_allow_html=True,
    )
with col3:
    st.image("photos/Mech.png", width=70)
st.markdown("---")

# ------------------------------------------------------------
# APP INTRODUCTION
# ------------------------------------------------------------
with st.expander("📘 About This App", expanded=True):
    st.markdown("""
    ### Purpose of the Application
    This app simulates and analyzes *energy or fuel consumption* of vehicles 
    using a *representative drive cycle* obtained within OAU campus.

    ### What the App Does
    - EV energy prediction
    - ICE fuel consumption & emissions
    - Physics + PCE models
    - Sobol sensitivity analysis
    """)

# Sidebar branding
scol1, scol2 = st.sidebar.columns(2)
scol1.image("photos/OAU_logo.png", width=45)
scol2.image("photos/Mech.png", width=45)

# ------------------------------------------------------------
# LOAD DRIVE CYCLE
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVE_CYCLE_PATH = os.path.join(BASE_DIR, "data", "final_drive_cycle.csv")

try:
    cycle_df = load_cycle(DRIVE_CYCLE_PATH)
    st.session_state["reference_cycle_loaded"] = True
except Exception as e:
    st.error(f"Could not load drive cycle: {e}")
    cycle_df = None

# ------------------------------------------------------------
# ICE MODEL
# ------------------------------------------------------------
def integrate_fuel_for_cycle(cycle_df, params):
    rho_air = 1.225
    g = 9.81
    LHV_J_per_L = 34.2e6
    CO2_g_per_L = 2392.0

    t = cycle_df["time_s"].values
    v = cycle_df["speed_m_s"].values
    dt = np.diff(t, append=t[-1] + (t[-1] - t[-2] if len(t) > 1 else 1))

    mass = params["MASS"]
    RRC = params["RRC"]
    Cd = params["Cd"]
    A = 2.2
    hw = params["HW"]
    aux_w = params["AUX_kW"] * 1000
    eff = max(0.01, params["Engine_Eff"] / 100)
    idle_lph = params["Idle_Fuel_Lph"]

    total_fuel_l, dist_m = 0, 0

    for i in range(len(v)):
        vi = v[i]
        ai = 0 if i == 0 else (v[i] - v[i - 1]) / (dt[i] if dt[i] > 0 else 1)
        v_air = vi + hw

        F_inertia = mass * ai
        F_roll = mass * g * RRC
        F_aero = 0.5 * rho_air * Cd * A * (v_air ** 2)

        F_trac = F_inertia + F_roll + F_aero
        P_wheel = F_trac * vi

        if P_wheel >= 0:
            P_engine = (P_wheel / 0.9) + aux_w
            fuel_l = (P_engine / eff / LHV_J_per_L) * dt[i]
        else:
            fuel_l = (idle_lph / 3600) * dt[i]

        total_fuel_l += fuel_l
        dist_m += vi * dt[i]

    dist_km = dist_m / 1000
    fuel_100 = (total_fuel_l / dist_km * 100) if dist_km > 0 else np.nan
    co2_km = (total_fuel_l * CO2_g_per_L / dist_km) if dist_km > 0 else np.nan

    return total_fuel_l, fuel_100, co2_km, dist_km

# ------------------------------------------------------------
# DECISION SUPPORT
# ------------------------------------------------------------
def generate_recommendations(main_df, model_type):
    recommendations = []
    top_factors = main_df.sort_values(by="S1", ascending=False).head(3)

    for _, row in top_factors.iterrows():
        factor = row["Parameter"]
        impact = row["S1"]

        if factor == "MASS":
            recommendations.append(f"Reduce vehicle mass (~{impact*100:.1f}%).")
        elif factor == "HW":
            recommendations.append(f"Reduce speed to minimize drag (~{impact*100:.1f}%).")
        elif factor == "RRC":
            recommendations.append(f"Maintain tyre pressure (~{impact*100:.1f}%).")
        elif factor == "Cd":
            recommendations.append(f"Improve aerodynamics (~{impact*100:.1f}%).")
        elif factor == "AUX_kW":
            recommendations.append(f"Reduce AC usage (~{impact*100:.1f}%).")
        elif factor == "Engine_Eff":
            recommendations.append(f"Improve engine efficiency (~{impact*100:.1f}%).")

    return recommendations

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tabs = st.tabs([
    "🚗 Vehicle Setup",
    "🧮 Physics Model",
    "🤖 Surrogate Models",
    "📊 Sensitivity Analysis"
])

# ------------------------------------------------------------
# TAB 1
# ------------------------------------------------------------
with tabs[0]:
    st.header("Vehicle Setup")

    fuel_type = st.selectbox("Fuel Type", ["EV", "ICE"])
    mass = st.number_input("Mass", 500, 8000, 2000)

    if st.button("Estimate"):
        params = {
            "Total mass of the vehicle (kg)": mass,
            "Rolling resistance coefficient, RRC": 0.01,
            "Headwind, HW": 1.0,
            "Auxiliary Power, AUX_kW": 1.0,
            "Type": fuel_type,
        }
        st.session_state["vehicle_params"] = params

# ------------------------------------------------------------
# TAB 4 (FIXED SECTION)
# ------------------------------------------------------------
with tabs[3]:
    st.header("📊 Sensitivity Analysis")

    if "vehicle_params" not in st.session_state:
        st.warning("Configure vehicle first")
    else:
        params = st.session_state["vehicle_params"]
        model_type = params["Type"]

        N = st.slider("Samples", 128, 1024, 512, step=128)

        base_params = {
            "MASS": params["Total mass of the vehicle (kg)"]
        }

        if st.button("Run Sensitivity Analysis"):
            try:
                main_df, aux_df = run_sobol(model_type, N, base_params)

                st.session_state["sensitivity_main"] = main_df

                if model_type == "EV":
                    st.pyplot(plot_sobol(main_df, "EV - Energy"))
                else:
                    st.pyplot(plot_sobol(main_df, "ICE - Fuel"))

            except Exception as e:
                st.error(f"Sensitivity failed: {e}")

        # ---------------- DECISION SUPPORT ----------------
        if "sensitivity_main" in st.session_state:
            st.markdown("---")
            st.subheader("💡 Decision Support")

            recs = generate_recommendations(
                st.session_state["sensitivity_main"],
                model_type
            )

            for rec in recs:
                st.write(f"✅ {rec}")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("""
---
<div style='text-align:center; font-size: small; color: grey;'>
Developed by Prof. B.O. Malomo, Blessing Babatope & Gabriel Oke | Dept. of Mechanical Engineering, OAU, Ile-Ife
</div>
""", unsafe_allow_html=True)