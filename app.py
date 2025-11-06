# app.py
"""
OAU Drive Cycle Energy Simulator (single reference drive cycle)
- Uses drivecycle/data/final_drive_cycle.csv as the reference cycle (no upload).
- Supports EV (physics + EV surrogates) and ICE (physics + placeholder surrogates).
- Estimator tab computes reasonable defaults for users who don't know parameters.
"""

import os
import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local modules (assume these files exist and are importable)
from drivecycle.io import load_cycle
from drivecycle.simulation import integrate_energy_for_cycle  # EV physics integrator you already have
from drivecycle.models import predict_pce, predict_dnn, FEATURE_ORDER  # EV surrogates
from drivecycle.sensitivity import run_sobol, plot_sobol  # EV sensitivity helpers

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
    except Exception:
        st.write("")
with col2:
    st.markdown("<h2 style='text-align:center; color:#004aad;'>OAU Drive Cycle Energy Simulator</h2>", unsafe_allow_html=True)
with col3:
    try:
        st.image("photos/Mech.png", width=70)
    except Exception:
        st.write("")
st.markdown("---")

# Sidebar logos + institution info
scol1, scol2 = st.sidebar.columns([1, 1])
with scol1:
    try:
        st.image("photos/OAU_logo.png", width=50)
    except Exception:
        st.write("")
with scol2:
    try:
        st.image("photos/Mech.png", width=50)
    except Exception:
        st.write("")
st.sidebar.markdown(
    """
    <div style='text-align:center; font-size:13px; line-height:1.3; color:#004aad;'>
    <b>Obafemi Awolowo University</b><br>
    Department of Mechanical Engineering
    </div>
    """,
    unsafe_allow_html=True
)

# Info: drive cycle usage
st.info("⚠ All surrogate predictions and sensitivity analysis are trained/defined for the *OAU representative drive cycle*. "
        "Users can change vehicle/fuel inputs below. Physics results use the same reference cycle.")

# ------------------------------------------------------------
# Constants & internal paths
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVE_CYCLE_PATH = os.path.join(BASE_DIR, "data", "final_drive_cycle.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ------------------------------------------------------------
# Helper: load the fixed drive cycle
# ------------------------------------------------------------
try:
    cycle_df = load_cycle(DRIVE_CYCLE_PATH)
    st.session_state["reference_cycle_loaded"] = True
except Exception as e:
    cycle_df = None
    st.session_state["reference_cycle_loaded"] = False
    st.error(f"Could not load reference drive cycle from {DRIVE_CYCLE_PATH}: {e}")

# ------------------------------------------------------------
# Initialize session state defaults
# ------------------------------------------------------------
def init_session_state_defaults():
    defaults = {
        # EV inputs (10 factors)
        "MASS": 3300,
        "HW": 0.0,
        "RRC": 0.010,
        "Ta": 25.0,
        "Tb": 25.0,
        "SoC_pct": 80.0,
        "BAge_pct": 0.0,
        "MR_mOhm": 55.0,
        "AUX_kW": 1.0,
        "BR_pct": 100.0,
        # ICE additional params
        "Engine_Eff": 30.0,    # %
        "Idle_Fuel_Lph": 0.8,  # L/h at idle
        "Displacement_L": 1.6, # engine size
        "Cd": 0.35,            # drag coefficient for ICE vehicle
        # control
        "last_physics_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state_defaults()

# ------------------------------------------------------------
# Presets
# ------------------------------------------------------------
VEHICLE_PRESETS = {
    "EV - Small Bus": {"MASS": 3500, "RRC": 0.010, "AUX_kW": 1.5, "MR_mOhm": 55, "BR_pct": 100, "SoC_pct": 90},
    "EV - Large Bus": {"MASS": 5000, "RRC": 0.012, "AUX_kW": 3.0, "MR_mOhm": 58, "BR_pct": 100, "SoC_pct": 90},
    "Petrol - Compact": {"MASS": 1300, "RRC": 0.010, "Engine_Eff": 28.0, "Idle_Fuel_Lph": 0.6, "Displacement_L": 1.4, "Cd": 0.30},
    "Petrol - SUV": {"MASS": 1800, "RRC": 0.011, "Engine_Eff": 26.0, "Idle_Fuel_Lph": 0.9, "Displacement_L": 2.4, "Cd": 0.38},
}

# ------------------------------------------------------------
# App flow controls (sidebar)
# ------------------------------------------------------------
st.sidebar.header("Simulation Setup")
fuel_type = st.sidebar.selectbox("Fuel type", ["Electric Vehicle (EV)", "Internal Combustion Engine (ICE)"])
preset_choice = st.sidebar.selectbox("Apply preset", ["None"] + list(VEHICLE_PRESETS.keys()))
if preset_choice != "None" and st.sidebar.button("Apply preset values"):
    preset = VEHICLE_PRESETS[preset_choice]
    for k, v in preset.items():
        st.session_state[k] = v
    st.sidebar.success(f"Preset '{preset_choice}' applied.")

# Main tabs
tabs = st.tabs(["🧠 Parameter Estimator", "🧮 Physics Model", "🤖 Surrogate Models", "📊 Sensitivity Analysis"])

# ------------------------------------------------------------
# Utility: Simple ICE physics integrator (uses same cycle)
# ------------------------------------------------------------
def integrate_fuel_for_cycle(cycle_df, params):
    """
    Simple physics-based fuel consumption estimate (ICE).
    - cycle_df must have 'time_s' and 'speed_m_s' (m/s).
    - params should contain: MASS, RRC, HW, Cd, AUX_kW, Engine_Eff (pct), Idle_Fuel_Lph
    Returns: total_fuel_l (L), fuel_l_per_100km, co2_g_per_km, distance_km
    """
    # constants
    rho_air = 1.225
    g = 9.81
    wheel_radius = 0.30  # not needed explicitly
    LHV_gasoline_MJ_per_L = 34.2  # MJ per liter (approx)
    LHV_J_per_L = LHV_gasoline_MJ_per_L * 1e6

    t = cycle_df["time_s"].values
    v = cycle_df["speed_m_s"].values  # m/s
    dt = np.diff(t, append=t[-1] + (t[-1]-t[-2] if len(t)>1 else 1.0))
    mass = float(params.get("MASS", 1500.0))
    RRC = float(params.get("RRC", 0.01))
    Cd = float(params.get("Cd", 0.35))
    A = 2.2  # frontal area (m^2) - generic
    hw = float(params.get("HW", 0.0))
    aux_w = float(params.get("AUX_kW", 0.0)) * 1000.0  # W
    engine_eff = max(0.01, float(params.get("Engine_Eff", 30.0)) / 100.0)  # convert % to fraction
    idle_fuel_lph = float(params.get("Idle_Fuel_Lph", 0.8))

    total_fuel_j = 0.0
    total_fuel_l = 0.0
    distance_m = 0.0

    for i in range(len(v)):
        vi = max(0.0, v[i])
        ai = 0.0 if i == 0 else (vi - v[i-1]) / (dt[i] if dt[i] > 0 else 1.0)
        v_air = max(0.0, vi + hw)

        # Forces
        F_inertia = mass * ai
        F_roll = mass * g * RRC
        F_aero = 0.5 * rho_air * Cd * A * (v_air ** 2)

        F_trac = F_inertia + F_roll + F_aero
        P_wheel = F_trac * vi  # W (can be negative for braking)

        if P_wheel >= 0:
            # Power the engine must produce (account for driveline losses roughly)
            driveline_eff = 0.9
            P_engine = (P_wheel / max(1e-6, driveline_eff)) + aux_w
            # Fuel chemical energy needed per second:
            fuel_power_needed_Js = P_engine / max(1e-6, engine_eff)
            # Convert energy per second to liters per second
            fuel_l_per_s = fuel_power_needed_Js / LHV_J_per_L
            fuel_l = fuel_l_per_s * dt[i]
            total_fuel_l += fuel_l
            total_fuel_j += fuel_power_needed_Js * dt[i]
        else:
            # braking (regenerative negligible for ICE) -> engine may be idling or off; still aux consumes fuel
            # approximate that during braking the fuel used is idle fuel rate (converted to L per dt)
            idle_l_per_s = idle_fuel_lph / 3600.0
            total_fuel_l += idle_l_per_s * dt[i]
            total_fuel_j += (idle_l_per_s * LHV_J_per_L) * dt[i]

        distance_m += vi * dt[i]

    # add idle fuel baseline for entire trip if engine idles at stops (approx)
    # Already approximated in braking sections above; keep this simple.

    distance_km = distance_m / 1000.0 if distance_m > 0 else 0.0
    fuel_l_per_100km = (total_fuel_l / distance_km * 100.0) if distance_km > 0 else float("nan")

    # CO2: petrol ~2392 g CO2 per liter (approx), diesel different; use petrol default
    CO2_g_per_L = 2392.0
    co2_g_per_km = (total_fuel_l * CO2_g_per_L / distance_km) if distance_km > 0 else float("nan")

    return total_fuel_l, fuel_l_per_100km, co2_g_per_km, distance_km

# ------------------------------------------------------------
# TAB 1: Parameter Estimator
# ------------------------------------------------------------
with tabs[0]:
    st.header("🧠 Parameter Estimator")
    st.write("Provide simple details and click Estimate parameters to fill technical inputs for your vehicle.")
    colA, colB = st.columns(2)
    with colA:
        curb_weight = st.number_input("Curb weight (kg)", 500, 8000, int(st.session_state["MASS"]))
        passengers = st.number_input("Passengers (typical)", 0, 100, 10)
        road_type = st.selectbox("Road type", ["City (stop-go)", "Urban arterial", "Highway", "Rough road"])
        tyre_type = st.selectbox("Tyre type", ["Standard", "Low rolling resistance", "Worn/rough"])
        ac_on = st.checkbox("Air conditioner / heavy aux on?", value=False)
    with colB:
        vehicle_size = st.selectbox("Vehicle size", ["Compact car", "SUV", "Small bus", "Large bus"])
        engine_size = st.selectbox("Engine size / motor type", ["Small", "Standard", "Large"])
        battery_age = st.slider("Battery age (%) (EV)", 0, 50, int(st.session_state["BAge_pct"]))
        soc_guess = st.slider("Typical SoC (%) (EV)", 10, 100, int(st.session_state["SoC_pct"]))

    if st.button("Estimate parameters"):
        mass_est = curb_weight + passengers * 70
        rrc_est = 0.010 if tyre_type == "Standard" else (0.007 if tyre_type == "Low rolling resistance" else 0.012)
        aux_est = 2.0 if ac_on else 0.8
        if vehicle_size == "Small bus":
            aux_est += 1.0
        elif vehicle_size == "Large bus":
            aux_est += 2.0
        hw_map = {"City (stop-go)": 0.0, "Urban arterial": 1.0, "Highway": 2.0, "Rough road": -1.0}
        hw_est = hw_map.get(road_type, 0.0)

        # engine/motor heuristics
        if fuel_type.startswith("Electric"):
            mr_est = 55.0 if vehicle_size.endswith("bus") else 50.0
            br_est = 100.0 + battery_age * 1.5
            soc_est = soc_guess
            st.session_state.update({"MASS": mass_est, "RRC": rrc_est, "AUX_kW": aux_est,
                                     "HW": hw_est, "MR_mOhm": mr_est, "BR_pct": br_est,
                                     "SoC_pct": soc_est, "BAge_pct": battery_age})
        else:
            # ICE
            engine_eff = 24.0 if engine_size == "Large" else 30.0
            idle_fuel = 1.0 if engine_size == "Large" else 0.6
            cd_guess = 0.38 if vehicle_size in ["SUV", "Large bus"] else 0.30
            st.session_state.update({"MASS": mass_est, "RRC": rrc_est, "AUX_kW": aux_est,
                                     "HW": hw_est, "Engine_Eff": engine_eff, "Idle_Fuel_Lph": idle_fuel,
                                     "Displacement_L": 2.0 if engine_size == "Large" else 1.4, "Cd": cd_guess})

        st.success("Parameters estimated and stored (editable in Physics tab).")

    st.markdown("#### Current parameter values (session)")
    param_table = {
        "MASS (kg)": st.session_state["MASS"],
        "RRC": st.session_state["RRC"],
        "AUX_kW (kW)": st.session_state["AUX_kW"],
        "HW (m/s)": st.session_state["HW"],
        "Engine_Eff (%)": st.session_state.get("Engine_Eff", "N/A"),
        "Idle_Fuel (L/h)": st.session_state.get("Idle_Fuel_Lph", "N/A"),
        "SoC_pct (%)": st.session_state.get("SoC_pct", "N/A"),
    }
    st.table(pd.DataFrame(list(param_table.items()), columns=["Parameter", "Value"]))

# ------------------------------------------------------------
# TAB 2: Physics Model (uses the fixed OAU cycle)
# ------------------------------------------------------------
with tabs[1]:
    st.header("🧮 Physics Model (reference cycle applied)")
    if not st.session_state["reference_cycle_loaded"]:
        st.error("Reference drive cycle not loaded. Check file path.")
    else:
        st.markdown("*Reference drive cycle:* drivecycle/data/final_drive_cycle.csv (used for all physics results)")

        # allow editing the parameters before running
        cols = st.columns(2)
        with cols[0]:
            st.session_state["MASS"] = st.number_input("Vehicle mass (kg)", 500.0, 8000.0, value=float(st.session_state["MASS"]), step=10.0)
            st.session_state["RRC"] = st.number_input("Rolling resistance (RRC)", 0.005, 0.020, value=float(st.session_state["RRC"]), step=0.001)
            st.session_state["AUX_kW"] = st.number_input("Auxiliary load (kW)", 0.0, 10.0, value=float(st.session_state["AUX_kW"]), step=0.1)
            st.session_state["HW"] = st.number_input("Headwind (m/s)", -5.0, 5.0, value=float(st.session_state["HW"]), step=0.5)
            st.session_state["Ta"] = st.number_input("Ambient temp (°C)", -10.0, 50.0, value=float(st.session_state["Ta"]))
        with cols[1]:
            if fuel_type.startswith("Electric"):
                st.session_state["Tb"] = st.number_input("Battery temp (°C)", -10.0, 60.0, value=float(st.session_state["Tb"]))
                st.session_state["SoC_pct"] = st.number_input("SoC (%)", 0.0, 100.0, value=float(st.session_state["SoC_pct"]))
                st.session_state["BAge_pct"] = st.number_input("Battery age (%)", 0.0, 100.0, value=float(st.session_state["BAge_pct"]))
                st.session_state["MR_mOhm"] = st.number_input("Motor internal resistance (mΩ)", 10.0, 200.0, value=float(st.session_state["MR_mOhm"]))
                st.session_state["BR_pct"] = st.number_input("Battery resistance growth (%)", 50.0, 300.0, value=float(st.session_state["BR_pct"]))
            else:
                st.session_state["Engine_Eff"] = st.number_input("Engine efficiency (%)", 5.0, 60.0, value=float(st.session_state["Engine_Eff"]))
                st.session_state["Idle_Fuel_Lph"] = st.number_input("Idle fuel (L/h)", 0.0, 5.0, value=float(st.session_state["Idle_Fuel_Lph"]))
                st.session_state["Displacement_L"] = st.number_input("Engine displacement (L)", 0.6, 8.0, value=float(st.session_state["Displacement_L"]))
                st.session_state["Cd"] = st.number_input("Drag coefficient (Cd)", 0.2, 0.6, value=float(st.session_state["Cd"]))

        # Run the appropriate physics simulation
        if st.button("Run physics simulation"):
            if fuel_type.startswith("Electric"):
                params = {
                    "MASS": st.session_state["MASS"],
                    "HW": st.session_state["HW"],
                    "RRC": st.session_state["RRC"],
                    "Ta": st.session_state["Ta"],
                    "Tb": st.session_state["Tb"],
                    "SoC_pct": st.session_state["SoC_pct"],
                    "BAge_pct": st.session_state["BAge_pct"],
                    "MR_mOhm": st.session_state["MR_mOhm"],
                    "AUX_kW": st.session_state["AUX_kW"],
                    "BR_pct": st.session_state["BR_pct"],
                }
                net_wh, energy_kwh_per_km, regen_pct, distance_km = integrate_energy_for_cycle(cycle_df, params)
                st.session_state["last_physics_result"] = {
                    "energy_kwh": net_wh,
                    "energy_kwh_per_km": energy_kwh_per_km,
                    "regen_pct": regen_pct,
                    "distance_km": distance_km
                }
                result = integrate_energy_for_cycle(...)
                st.success("EV physics simulation complete.")
                st.metric("Energy (kWh/km)", f"{result['energy_kwh_per_km']:.3f}")
                st.metric("Regen (%)", f"{result['regen_pct']:.2f}")
                st.metric("Distance (km)", f"{result['distance_km']:.2f}")
            else:
                params = {
                    "MASS": st.session_state["MASS"],
                    "HW": st.session_state["HW"],
                    "RRC": st.session_state["RRC"],
                    "Cd": st.session_state["Cd"],
                    "AUX_kW": st.session_state["AUX_kW"],
                    "Engine_Eff": st.session_state["Engine_Eff"],
                    "Idle_Fuel_Lph": st.session_state["Idle_Fuel_Lph"],
                }
                total_fuel_l, fuel_l_per_100km, co2_g_per_km, distance_km = integrate_fuel_for_cycle(cycle_df, params)
                st.session_state["last_physics_result"] = {
                    "fuel_l_total": total_fuel_l,
                    "fuel_l_per_100km": fuel_l_per_100km,
                    "co2_g_per_km": co2_g_per_km,
                    "distance_km": distance_km
                }
                st.success("ICE physics estimate complete.")
                st.metric("Fuel (L/100 km)", f"{fuel_l_per_100km:.3f}")
                st.metric("CO2 (g/km)", f"{co2_g_per_km:.1f}")
                st.metric("Distance (km)", f"{distance_km:.2f}")

        # Show the reference cycle plot
        st.subheader("Reference drive cycle (speed vs time)")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(cycle_df["time_s"], cycle_df["speed_m_s"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.grid(True)
        st.pyplot(fig)

# ------------------------------------------------------------
# TAB 3: Surrogate Models
# ------------------------------------------------------------
with tabs[2]:
    st.header("🤖 Surrogate Models")

    st.markdown("Surrogates are trained for the *OAU reference drive cycle*. Supply the inputs below (pre-filled from estimator/session).")

    # Build an input dataframe for EV surrogate feature order (we reuse FEATURE_ORDER)
    # If fuel type is ICE, we'll map available fields into the same structure or show placeholder
    if fuel_type.startswith("Electric"):
        input_df = pd.DataFrame([{
            "MASS": st.session_state["MASS"],
            "HW": st.session_state["HW"],
            "RRC": st.session_state["RRC"],
            "Ta": st.session_state["Ta"],
            "Tb": st.session_state["Tb"],
            "SoC_pct": st.session_state["SoC_pct"],
            "BAge_pct": st.session_state["BAge_pct"],
            "MR_mOhm": st.session_state["MR_mOhm"],
            "AUX_kW": st.session_state["AUX_kW"],
            "BR_pct": st.session_state["BR_pct"],
        }])
        if st.button("Run EV surrogates (PCE & DNN)"):
            with st.spinner("Predicting..."):
                try:
                    pce_out = predict_pce(input_df)
                except Exception as e:
                    pce_out = pd.DataFrame([{"energy_kwh_per_km": None, "regen_pct": None}])
                    st.error(f"PCE EV prediction error: {e}")
                try:
                    dnn_out = predict_dnn(input_df)
                except Exception as e:
                    dnn_out = pd.DataFrame([{"energy_kwh_per_km": None, "regen_pct": None}])
                    st.warning(f"DNN EV unavailable or error: {e}")

            st.subheader("Predicted outputs (EV)")
            st.write("PCE:", pce_out.to_dict(orient="records")[0])
            st.write("DNN:", dnn_out.to_dict(orient="records")[0])
    else:
        # ICE surrogate placeholders
        st.info("ICE surrogates are not yet trained in this release. The physics model gives direct estimates above.")
        st.markdown("You can still run sensitivity for ICE (if surrogate available).")

# ------------------------------------------------------------
# TAB 4: Sensitivity Analysis
# ------------------------------------------------------------
with tabs[3]:
    st.header("📊 Global Sensitivity Analysis")
    st.markdown("Sensitivity (Sobol) is computed with surrogates trained on the OAU reference cycle. Select model/fuel type.")

    model_choice = st.selectbox("Model for Sobol (PCE/DNN) — EV only currently", ["PCE (EV)", "DNN (EV)"])
    N = st.slider("Sobol base sample (N, power-of-2 preferred)", 128, 2048, 512, step=128)

    if st.button("Run Sensitivity"):
        # Only EV surrogates are implemented in the repo so we run EV sensitivity
        if "EV" in model_choice:
            with st.spinner("Running Sobol (EV surrogate). This can take a minute..."):
                try:
                    energy_df, regen_df = run_sobol("pce", N)  # run_sobol currently uses PCE or DNN; here we ask for pce
                    # plot top 4 using helper plot_sobol which we assume accepts top_n
                    st.subheader("Top factors for energy consumption (EV)")
                    fig1 = plot_sobol(energy_df, title="EV - Energy", top_n=4)
                    st.pyplot(fig1)
                    st.subheader("Top factors for regeneration (EV)")
                    fig2 = plot_sobol(regen_df, title="EV - Regen", top_n=4)
                    st.pyplot(fig2)
                except Exception as e:
                    st.error(f"Sobol analysis failed: {e}")
        else:
            st.info("ICE sensitivity not implemented yet. Train an ICE surrogate and add it to the sensitivity backend.")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown(
    """
    <div style='text-align:center; font-size: small; color: grey; padding-top:10px;'>
    Developed by Prof. B.O. Malomo, Blessing Babatope and Gabriel Oke | Department of Mechanical Engineering, OAU, Ile-Ife
    </div>
    """,
    unsafe_allow_html=True
)