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
# DECISION SUPPORT FUNCTION
# ------------------------------------------------------------
def generate_recommendations(model_type, performance, sensitivity_df):
    recommendations = []

    if sensitivity_df is not None and not sensitivity_df.empty:
        top_factors = sensitivity_df.sort_values(by="ST", ascending=False)["Parameter"].head(3).tolist()

        for factor in top_factors:
            if factor == "MASS":
                recommendations.append("Reduce vehicle mass to improve efficiency.")
            elif factor == "HW":
                recommendations.append("Reduce exposure to headwinds.")
            elif factor == "RRC":
                recommendations.append("Use low rolling resistance tyres.")
            elif factor == "AUX_kW":
                recommendations.append("Reduce auxiliary loads (e.g., AC usage).")
            elif factor == "Cd":
                recommendations.append("Improve vehicle aerodynamics.")
            elif factor == "Engine_Eff":
                recommendations.append("Improve engine efficiency.")
            elif factor == "BAge_pct":
                recommendations.append("Battery degradation is significant — monitor health.")
            elif factor == "MR_mOhm":
                recommendations.append("Motor resistance impacts performance.")

    # Performance rule
    if model_type == "EV":
        if performance.get("energy_kwh_per_km", 0) > 0.25:
            recommendations.append("High energy consumption detected — optimize conditions.")
    else:
        if performance.get("fuel_100", 0) > 12:
            recommendations.append("High fuel consumption detected — improve efficiency factors.")

    return recommendations


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
# ABOUT
# ------------------------------------------------------------
with st.expander("📘 About This App", expanded=True):
    st.markdown("""Your existing description here...""")

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
    dt = np.diff(t, append=t[-1])

    total_fuel_l, dist_m = 0, 0

    for i in range(len(v)):
        vi = v[i]
        dt_i = dt[i] if dt[i] > 0 else 1
        ai = 0 if i == 0 else (v[i] - v[i - 1]) / dt_i

        F = params["MASS"] * ai
        P = F * vi

        fuel_l = max(0, P) * dt_i / (params["Engine_Eff"] / 100 * LHV_J_per_L)
        total_fuel_l += fuel_l
        dist_m += vi * dt_i

    dist_km = dist_m / 1000
    fuel_100 = (total_fuel_l / dist_km * 100) if dist_km > 0 else 0
    co2_km = (total_fuel_l * CO2_g_per_L / dist_km) if dist_km > 0 else 0

    return total_fuel_l, fuel_100, co2_km, dist_km


# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tabs = st.tabs(["🚗 Vehicle Setup", "🧮 Physics Model", "🤖 Surrogate Models", "📊 Sensitivity Analysis"])

# ------------------------------------------------------------
# TAB 1
# ------------------------------------------------------------
with tabs[0]:
    st.header("Vehicle Setup")

    fuel_type = st.selectbox("Fuel Type", ["EV", "ICE"])
    mass = st.number_input("Mass", 500, 8000, 2000)

    if st.button("Estimate Parameters"):
        params = {
            "Total mass of the vehicle (kg)": mass,
            "Rolling resistance coefficient, RRC": 0.01,
            "Headwind, HW": 1.0,
            "Auxiliary Power, AUX_kW": 1.0,
            "Type": fuel_type,
        }
        st.session_state["vehicle_params"] = params
        st.success("Done")

# ------------------------------------------------------------
# TAB 2
# ------------------------------------------------------------
with tabs[1]:
    st.header("Physics Model")

    if "vehicle_params" in st.session_state:
        params = st.session_state["vehicle_params"]

        if st.button("Run Simulation"):

            if params["Type"] == "EV":
                result = integrate_energy_for_cycle(cycle_df, {
                    "MASS": params["Total mass of the vehicle (kg)"],
                    "RRC": params["Rolling resistance coefficient, RRC"],
                    "HW": params["Headwind, HW"],
                    "AUX_kW": params["Auxiliary Power, AUX_kW"],
                    "Ta": 25
                })

                st.session_state["performance"] = result
                st.metric("Energy", result["energy_kwh_per_km"])

            else:
                total_fuel_l, fuel_100, co2_km, dist_km = integrate_fuel_for_cycle(cycle_df, {
                    "MASS": params["Total mass of the vehicle (kg)"],
                    "RRC": params["Rolling resistance coefficient, RRC"],
                    "HW": params["Headwind, HW"],
                    "AUX_kW": params["Auxiliary Power, AUX_kW"],
                    "Engine_Eff": 30,
                })

                st.session_state["performance"] = {"fuel_100": fuel_100}
                st.metric("Fuel", fuel_100)

# ------------------------------------------------------------
# TAB 3 (UNCHANGED)
# ------------------------------------------------------------
with tabs[2]:
    st.header("Surrogate Model")

# ------------------------------------------------------------
# TAB 4 - SENSITIVITY + DECISION SUPPORT
# ------------------------------------------------------------
with tabs[3]:
    st.header("Sensitivity + Decision Support")

    if "vehicle_params" in st.session_state:
        params = st.session_state["vehicle_params"]

        if st.button("Run Sensitivity"):
            main_df, aux_df = run_sobol(params["Type"], 512, {"MASS": 2000})

            st.session_state["sensitivity_main"] = main_df

            st.pyplot(plot_sobol(main_df, "Main", top_n=4))

            # ----------------------------
            # DECISION SUPPORT
            # ----------------------------
            st.subheader("🧠 Recommendations")

            if "performance" in st.session_state:
                recs = generate_recommendations(
                    params["Type"],
                    st.session_state["performance"],
                    main_df
                )

                for r in recs:
                    st.write(f"✅ {r}")
            else:
                st.warning("Run simulation first")