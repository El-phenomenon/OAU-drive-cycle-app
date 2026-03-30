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
# HEADER
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
# DECISION SUPPORT FUNCTION
# ------------------------------------------------------------
def generate_recommendations(main_df, model_type):
    recommendations = []
    top_factors = main_df.sort_values(by="S1", ascending=False).head(3)

    for _, row in top_factors.iterrows():
        factor = row["Parameter"]
        impact = row["S1"]

        if factor == "MASS":
            recommendations.append(f"Reduce vehicle load (~{impact*100:.1f}% influence).")
        elif factor == "HW":
            recommendations.append(f"Reduce speed to limit aerodynamic drag (~{impact*100:.1f}%).")
        elif factor == "RRC":
            recommendations.append(f"Maintain tyre pressure (~{impact*100:.1f}%).")
        elif factor == "Cd":
            recommendations.append(f"Improve aerodynamics (~{impact*100:.1f}%).")
        elif factor == "AUX_kW":
            recommendations.append(f"Reduce AC/electrical loads (~{impact*100:.1f}%).")
        elif factor == "Engine_Eff":
            recommendations.append(f"Improve engine efficiency (~{impact*100:.1f}%).")
        elif factor == "SoC_pct":
            recommendations.append(f"Avoid low battery charge (~{impact*100:.1f}%).")

    return recommendations

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tabs = st.tabs(["Vehicle Setup", "Physics Model", "Surrogate Models", "Sensitivity"])

# ------------------------------------------------------------
# TAB 1
# ------------------------------------------------------------
with tabs[0]:
    st.header("Vehicle Setup")

    fuel_type = st.selectbox("Fuel Type", ["EV", "ICE"])
    mass = st.number_input("Vehicle Mass (kg)", 500, 8000, 2000)
    passengers = st.number_input("Passengers", 0, 10, 2)

    if st.button("Generate Parameters"):
        total_mass = mass + passengers * 70

        if fuel_type == "EV":
            params = {
                "Total mass of the vehicle (kg)": total_mass,
                "Rolling resistance coefficient, RRC": 0.01,
                "Headwind, HW": 1.0,
                "Auxiliary Power, AUX_kW": 1.5,
                "State of charge of battery, SoC_pct": 80.0,
                "Type": "EV",
            }
        else:
            params = {
                "Total mass of the vehicle (kg)": total_mass,
                "Rolling Resistance coefficient, RRC": 0.01,
                "Headwind, HW": 1.0,
                "Auxiliary Power, AUX_kW": 1.5,
                "Drag Coefficient, Cd": 0.35,
                "Engine_Eff": 30.0,
                "Idle_Fuel_Lph": 0.8,
                "Type": "ICE",
            }

        st.session_state["vehicle_params"] = params
        st.success("Parameters Generated")

# ------------------------------------------------------------
# TAB 4 - SENSITIVITY + DECISION SUPPORT
# ------------------------------------------------------------
with tabs[3]:
    st.header("Sensitivity Analysis")

    if "vehicle_params" not in st.session_state:
        st.warning("Set up vehicle first.")
    else:
        params = st.session_state["vehicle_params"]
        model_type = params["Type"]

        N = st.slider("Samples", 128, 1024, 512)

        base_params = {"MASS": params["Total mass of the vehicle (kg)"]}

        if st.button("Run Sensitivity Analysis"):
            try:
                main_df, aux_df = run_sobol(model_type, N, base_params)

                st.session_state["sensitivity_main"] = main_df

                st.pyplot(plot_sobol(main_df, "Main Output"))

            except Exception as e:
                st.error(str(e))

        # ---------------- DECISION SUPPORT ----------------
        if "sensitivity_main" in st.session_state:
            st.subheader("Decision Support")

            recs = generate_recommendations(
                st.session_state["sensitivity_main"], model_type
            )

            for rec in recs:
                st.write(f"✅ {rec}")