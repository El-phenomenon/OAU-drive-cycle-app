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
    st.markdown("<h2 style='text-align:center; color:#004aad;'>OAU Drive Cycle Energy Simulator</h2>", unsafe_allow_html=True)
with col3:
    st.image("photos/Mech.png", width=70)
st.markdown("---")

# ------------------------------------------------------------
# APP INTRODUCTION / README SECTION
# ------------------------------------------------------------
with st.expander("📘 About This App", expanded=True):
    st.markdown("""
    ###  *Purpose of the Application*
    This application simulates and analyzes the *energy or fuel consumption* of vehicles 
    using a *representative drive cycle* collected within the Obafemi Awolowo University (OAU) campus which reflects typical Nigerian driving patterns.

    ### *What the App Does*
    - Predicts *energy use* and *regeneration efficiency* for *Electric Vehicles (EVs)*.
    - Estimates *fuel consumption* and *CO₂ emissions* for *Internal Combustion Engine (ICE)* vehicles.
    - Uses both *physics-based models* and *Polynomial Chaos Expansion (PCE)* surrogate models.
    - Performs *global sensitivity analysis* (Sobol method) to identify the top factors 
      affecting energy or fuel performance.

    ### *How It Works*
    1. In the *Vehicle Setup* tab, specify your fuel type, vehicle details, and driving conditions.  
       The app automatically estimates technical parameters like rolling resistance, headwind, and efficiency.  
    2. Run the *Physics Model* to compute actual energy or fuel use based on the OAU drive cycle.  
    3. The *Surrogate Models* section predicts performance instantly using trained PCE models.  
    4. Finally, the *Sensitivity Analysis* tab identifies the *top 4 most influential factors* 
       affecting energy or fuel use for your vehicle.

    ---
    *Developed by:*  
    Prof. B.O. Malomo · Blessing Babatope · Gabriel Oke  
    Department of Mechanical Engineering, Obafemi Awolowo University, Ile-Ife.
    """)

# Sidebar logos
scol1, scol2 = st.sidebar.columns(2)
scol1.image("photos/OAU_logo.png", width=45)
scol2.image("photos/Mech.png", width=45)
st.sidebar.markdown(
    "<div style='text-align:center; font-size:13px; color:#004aad;'>"
    "<b>Obafemi Awolowo University</b><br>Dept. of Mechanical Engineering</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# LOAD REFERENCE DRIVE CYCLE
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
# ICE MODEL HELPER
# ------------------------------------------------------------
def integrate_fuel_for_cycle(cycle_df, params):
    rho_air = 1.225
    g = 9.81
    LHV_J_per_L = 34.2e6
    CO2_g_per_L = 2392.0
    t = cycle_df["time_s"].values
    v = cycle_df["speed_m_s"].values
    dt = np.diff(t, append=t[-1] + (t[-1]-t[-2] if len(t) > 1 else 1))
    mass = params["MASS"]; RRC = params["RRC"]; Cd = params["Cd"]; A = 2.2
    hw = params["HW"]; aux_w = params["AUX_kW"] * 1000
    eff = max(0.01, params["Engine_Eff"] / 100); idle_lph = params["Idle_Fuel_Lph"]
    total_fuel_l, dist_m = 0, 0
    for i in range(len(v)):
        vi = v[i]; ai = 0 if i == 0 else (v[i] - v[i-1]) / (dt[i] if dt[i] > 0 else 1)
        v_air = vi + hw
        F_inertia = mass * ai; F_roll = mass * g * RRC
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
# TABS
# ------------------------------------------------------------
tabs = st.tabs(["🚗 Vehicle Setup", "🧮 Physics Model", "🤖 Surrogate Models", "📊 Sensitivity Analysis"])

# ------------------------------------------------------------
# TAB 1 - Vehicle Setup
# ------------------------------------------------------------
with tabs[0]:
    st.header("🚗 Vehicle Setup & Parameter Estimation")
    fuel_type = st.selectbox("Select Fuel Type", ["Electric Vehicle (EV)", "Internal Combustion Engine (ICE)"])
    mass = st.number_input("Vehicle Mass (kg)", 500, 8000, 2000, step=50)
    passengers = st.number_input("Passengers", 0, 100, 2)
    terrain = st.selectbox("Road Type", ["City", "Urban Arterial", "Highway", "Rough"])
    ac_on = st.checkbox("Air conditioner/heavy auxiliary load", value=False)
    tyre_type = st.selectbox("Tyre Condition", ["Standard", "Low Rolling Resistance", "Worn"])
    engine_size = st.selectbox("Engine/Motor Size", ["Small", "Standard", "Large"])

    if st.button("Estimate Technical Parameters"):
        total_mass = mass + passengers * 70
        hw = {"City": 0.0, "Urban Arterial": 1.0, "Highway": 2.5, "Rough": -1.0}[terrain]
        rrc = {"Standard": 0.010, "Low Rolling Resistance": 0.007, "Worn": 0.012}[tyre_type]
        aux = 2.0 if ac_on else 0.8

        if fuel_type == "Electric Vehicle (EV)":
            params = {
                "Total mass of the vehicle (kg)": total_mass, "Rolling resistance coefficient, RRC": rrc, "Headwind, HW": hw, "Auxiliary Power, AUX_kW": aux,
                "Temperature of Battery Pack, Tb (celsius)": 25.0, "State of charge of battery, SoC_pct": 80.0, "Battery capacity fade due to aging, BAge_pct": 10.0,
                "Internal resistance of the motor, MR_mOhm": 55.0 if engine_size != "Small" else 50.0,
                "Internal resistance growth of the battery, BR_pct": 100.0 + 5.0, "Type": "EV"
            }
        else:
            eff = 30.0 if engine_size == "Standard" else (25.0 if engine_size == "Large" else 35.0)
            idle = 0.8 if engine_size == "Standard" else (1.0 if engine_size == "Large" else 0.6)
            Cd = 0.32 if engine_size == "Small" else (0.36 if engine_size == "Standard" else 0.40)
            params = {
                "Total mass of the vehicle": total_mass, "Rolling Resistance coefficient, RRC": rrc, "Headwind, HW": hw, "Auxiliary Power, AUX_kW": aux,
                "Drag Coefficient, Cd": Cd, "Engine_Eff": eff, "Idle_Fuel_Lph": idle, "Type": "ICE"
            }

        st.session_state["vehicle_params"] = params
        st.success("✅ Parameters estimated successfully!")
        st.write(pd.DataFrame(params.items(), columns=["Factor", "Value"]))

# ------------------------------------------------------------
# TAB 2 - Physics Model
# ------------------------------------------------------------
with tabs[1]:
    st.header("🧮 Physics-Based Simulation")
    if "vehicle_params" not in st.session_state:
        st.warning("Please configure your vehicle in the Vehicle Setup tab first.")
    elif not st.session_state["reference_cycle_loaded"]:
        st.error("Reference drive cycle not loaded.")
    else:
        params = st.session_state["vehicle_params"]
        st.write("Using parameters:")
        st.write(pd.DataFrame(params.items(), columns=["Factor", "Value"]))

        if st.button("Run Physics Simulation"):
            if params["Type"] == "EV":
                result = integrate_energy_for_cycle(cycle_df, params)
                st.success("EV simulation complete.")
                st.metric("Energy (kWh/km)", f"{result['energy_kwh_per_km']:.3f}")
                st.metric("Regeneration (%)", f"{result['regen_pct']:.2f}")
            else:
                 fuel_params = {
        "MASS": params["Total mass of the vehicle"],
        "RRC": params["Rolling Resistance coefficient, RRC"],
        "HW": params["Headwind, HW"],
        "AUX_kW": params["Auxiliary Power, AUX_kW"],
        "Cd": params["Drag Coefficient, Cd"],
        "Engine_Eff": params["Engine_Eff"],
        "Idle_Fuel_Lph": params["Idle_Fuel_Lph"]
    }#y
                total_fuel_l, fuel_100, co2_km, dist_km = integrate_fuel_for_cycle(cycle_df, params)
                st.success("ICE simulation complete.")
                st.metric("Fuel (L/100 km)", f"{fuel_100:.3f}")
                st.metric("CO₂ emission (g/km)", f"{co2_km:.1f}")
        st.subheader("Reference Drive Cycle (Speed vs Time)")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(cycle_df["time_s"], cycle_df["speed_m_s"], color="blue")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (m/s)"); ax.grid(True)
        st.pyplot(fig)

# ------------------------------------------------------------
# ------------------------------------------------------------
# TAB 3 - Surrogates
# ------------------------------------------------------------
with tabs[2]:
    st.header("🤖 Surrogate Model (PCE)")
    if "vehicle_params" not in st.session_state:
        st.warning("Configure your vehicle first in the Vehicle Setup tab.")
    else:
        p = st.session_state["vehicle_params"]
        if p["Type"] == "EV":
            # map full names to abbreviations for EV
            input_df = pd.DataFrame([{
                "MASS": p["Total mass of the vehicle (kg)"],
                "HW": p["Headwind, HW"],
                "RRC": p["Rolling resistance coefficient, RRC"],
                "Ta": 25.0,
                "Tb": p["Temperature of Battery Pack, Tb (celsius)"],
                "SoC_pct": p["State of charge of battery, SoC_pct"],
                "BAge_pct": p["Battery capacity fade due to aging, BAge_pct"],
                "MR_mOhm": p["Internal resistance of the motor, MR_mOhm"],
                "AUX_kW": p["Auxiliary Power, AUX_kW"],
                "BR_pct": p["Internal resistance growth of the battery, BR_pct"]
            }])
            if st.button("Run EV Surrogate Prediction (PCE)"):
                try:
                    pce_out = predict_pce(input_df)
                    st.success("EV Surrogate Prediction complete.")
                    st.write(pce_out)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            # map full names to abbreviations for ICE
            input_df = pd.DataFrame([{
                "MASS": p["Total mass of the vehicle"],
                "HW": p["Headwind, HW"],
                "RRC": p["Rolling Resistance coefficient, RRC"],
                "Cd": p["Drag Coefficient, Cd"],
                "AUX_kW": p["Auxiliary Power, AUX_kW"],
                "Engine_Eff": p["Engine_Eff"],
                "Idle_Fuel_Lph": p["Idle_Fuel_Lph"]
            }])
            if st.button("Run ICE Surrogate Prediction (PCE)"):
                try:
                    pce_out = predict_pce_ice(input_df)
                    st.success("ICE Surrogate Prediction complete.")
                    st.write(pce_out)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# ------------------------------------------------------------
# TAB 4 - Sensitivity
# ------------------------------------------------------------
with tabs[3]:
    st.header("📊 Sensitivity Analysis")
    if "vehicle_params" not in st.session_state:
        st.warning("Configure your vehicle first in Vehicle Setup.")
    else:
        params = st.session_state["vehicle_params"]
        model_type = params["Type"]
        N = st.slider("Number of Sobol Samples", 128, 1024, 512, step=128)

        # Map full names to abbreviated keys for analysis
        if model_type == "EV":
            base_params = {
                "MASS": params["Total mass of the vehicle (kg)"],
                "RRC": params["Rolling resistance coefficient, RRC"],
                "HW": params["Headwind, HW"],
                "AUX_kW": params["Auxiliary Power, AUX_kW"],
                "Tb": params["Temperature of Battery Pack, Tb (celsius)"],
                "SoC_pct": params["State of charge of battery, SoC_pct"],
                "BAge_pct": params["Battery capacity fade due to aging, BAge_pct"],
                "MR_mOhm": params["Internal resistance of the motor, MR_mOhm"],
                "BR_pct": params["Internal resistance growth of the battery, BR_pct"],
                "Ta": 25.0
            }
        else:
            base_params = {
                "MASS": params["Total mass of the vehicle"],
                "HW": params["Headwind, HW"],
                "RRC": params["Rolling Resistance coefficient, RRC"],
                "Cd": params["Drag Coefficient, Cd"],
                "AUX_kW": params["Auxiliary Power, AUX_kW"],
                "Engine_Eff": params["Engine_Eff"],
                "Idle_Fuel_Lph": params["Idle_Fuel_Lph"]
            }

        if st.button("Run Sensitivity Analysis"):
            try:
                main_df, aux_df = run_sobol(model_type, N, base_params)
                if model_type == "EV":
                    st.subheader("Energy Consumption (Top 4)")
                    st.pyplot(plot_sobol(main_df, "EV - Energy", top_n=4))
                    st.subheader("Regeneration Efficiency (Top 4)")
                    st.pyplot(plot_sobol(aux_df, "EV - Regen", top_n=4))
                else:
                    st.subheader("Fuel Consumption (Top 4)")
                    st.pyplot(plot_sobol(main_df, "ICE - Fuel", top_n=4))
                    st.subheader("CO₂ Emission (Top 4)")
                    st.pyplot(plot_sobol(aux_df, "ICE - CO₂", top_n=4))
            except Exception as e:
                st.error(f"Sensitivity failed: {e}")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("""
---
<div style='text-align:center; font-size: small; color: grey;'>
Developed by Prof. B.O. Malomo, Blessing Babatope & Gabriel Oke | Dept. of Mechanical Engineering, OAU Ile-Ife
</div>
""", unsafe_allow_html=True)