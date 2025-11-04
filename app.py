# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local modules (assumes these exist as in your project)
from drivecycle.io import load_cycle
from drivecycle.simulation import integrate_energy_for_cycle
from drivecycle.models import predict_pce, predict_dnn, FEATURE_ORDER
from drivecycle.sensitivity import run_sobol, plot_sobol

st.set_page_config(page_title="OAU Drive Cycle Energy Simulator", layout="wide")

# -----------------------
# Helper: session-state defaults
# -----------------------
def init_session_state():
    defaults = {
        # EV defaults (the 10 factors used by your surrogates)
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
        # ICE-specific (example placeholders)
        "Engine_Eff": 30.0,    # %
        "Idle_Fuel": 0.8,      # L/h
        "Displacement_L": 2.0, # L
        "AFR": 14.7,           # air/fuel ratio
        # common control values
        "uploaded_cycle": None,
        "last_sim_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# -----------------------
# Presets for quick fill (can expand)
# -----------------------
VEHICLE_PRESETS = {
    "EV - Small Bus": {
        "MASS": 3500, "RRC": 0.010, "AUX_kW": 1.5, "MR_mOhm": 55, "BR_pct": 100, "SoC_pct": 90
    },
    "EV - Large Bus": {
        "MASS": 5000, "RRC": 0.012, "AUX_kW": 3.0, "MR_mOhm": 58, "BR_pct": 100, "SoC_pct": 90
    },
    "Petrol - Compact": {
        "MASS": 1300, "RRC": 0.010, "Engine_Eff": 28.0, "Idle_Fuel": 0.6, "Displacement_L": 1.4, "AFR": 14.7
    },
    "Petrol - SUV": {
        "MASS": 1800, "RRC": 0.011, "Engine_Eff": 26.0, "Idle_Fuel": 0.9, "Displacement_L": 2.4, "AFR": 14.7
    }
}

# -----------------------
# Top header
# -----------------------
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.image("photos/OAU_logo.png", width=70)
with col2:
    st.markdown("<h2 style='text-align:center; color:#004aad;'>OAU Drive Cycle Energy Simulator</h2>", unsafe_allow_html=True)
with col3:
    st.image("photos/Mech.png", width=70)
st.markdown("---")

# -----------------------
# Left sidebar: vehicle type + quick controls
# -----------------------
st.sidebar.header("Setup & Quick Controls")
vehicle_type = st.sidebar.selectbox("Vehicle Type", ["Electric Vehicle (EV)", "Petrol Engine (ICE)"])
preset_choice = st.sidebar.selectbox("Vehicle Preset (quick fill)", ["None"] + list(VEHICLE_PRESETS.keys()))

if preset_choice != "None":
    if st.sidebar.button("Apply preset"):
        preset = VEHICLE_PRESETS[preset_choice]
        # Update session state with preset values
        for k, v in preset.items():
            st.session_state[k] = v
        st.sidebar.success(f"Preset '{preset_choice}' applied. Values filled.")

# -----------------------
# Tabs: Estimator, Physics, Surrogates, Sensitivity
# -----------------------
tabs = st.tabs(["🧠 Parameter Estimator", "🧮 Physics Model", "🤖 Surrogate Models", "📊 Sensitivity Analysis"])

# -----------------------
# TAB: Parameter Estimator
# -----------------------
with tabs[0]:
    st.header("🧠 Parameter Estimator (help to compute unknown inputs)")
    st.write("If you don't know the technical parameters, enter simple vehicle details below and press Estimate.")
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Basic details")
        curb_weight = st.number_input("Curb weight (kg)", 500, 8000, value=int(st.session_state["MASS"]))
        passengers = st.number_input("Passengers (typical load)", 0, 100, 10)
        road_type = st.selectbox("Road type", ["City (stop-go)", "Urban arterial", "Highway", "Rough road"])
        tyre_type = st.selectbox("Tire type", ["Standard", "Low rolling resistance", "Worn/rough"])
        ac_on = st.checkbox("Air Conditioner / Heavy Aux On?", value=False)

    with colB:
        st.subheader("Vehicle specifics")
        vehicle_size = st.selectbox("Vehicle size", ["Compact car", "SUV", "Small bus", "Large bus"])
        engine_or_motor = st.selectbox("Engine / Motor type", ["Standard motor", "High-efficiency motor", "Small petrol engine", "Large petrol engine"])
        battery_age = st.slider("Battery age (%) (for EVs)", 0, 50, int(st.session_state["BAge_pct"]))
        soc_guess = st.slider("Typical State of Charge (%)", 10, 100, int(st.session_state["SoC_pct"]))

    if st.button("Estimate parameters"):
        # Simple rules (tweak these as you like)
        mass_est = curb_weight + passengers * 70
        # rolling resistance estimator
        if tyre_type == "Low rolling resistance":
            rrc_est = 0.007
        elif tyre_type == "Worn/rough":
            rrc_est = 0.012
        else:
            rrc_est = 0.010

        # aux load estimate
        aux_est = 2.0 if ac_on else 0.8
        if vehicle_size == "Small bus":
            aux_est += 1.0
        elif vehicle_size == "Large bus":
            aux_est += 2.0

        # headwind estimate (simple)
        hw_est = 0.0
        if road_type == "Highway":
            hw_est = 2.0
        elif road_type == "City (stop-go)":
            hw_est = 0.0
        elif road_type == "Rough road":
            hw_est = -1.0

        # motor/engine defaults
        if vehicle_type.startswith("Electric"):
            mr_est = 55.0 if vehicle_size.endswith("bus") else 50.0
            br_est = 100.0 + battery_age * 1.5
            soc_est = soc_guess
        else:
            # ICE estimates
            mr_est = None
            br_est = None
            soc_est = None

        # engine-specific
        if vehicle_type.startswith("Petrol"):
            if engine_or_motor.startswith("Large"):
                engine_eff = 24.0
                idle_fuel = 1.0
            else:
                engine_eff = 30.0
                idle_fuel = 0.6
        else:
            engine_eff = None
            idle_fuel = None

        # Update session_state
        st.session_state["MASS"] = float(mass_est)
        st.session_state["RRC"] = float(rrc_est)
        st.session_state["AUX_kW"] = float(aux_est)
        st.session_state["HW"] = float(hw_est)
        if mr_est is not None:
            st.session_state["MR_mOhm"] = float(mr_est)
        if br_est is not None:
            st.session_state["BR_pct"] = float(br_est)
        if soc_est is not None:
            st.session_state["SoC_pct"] = float(soc_est)
        if engine_eff is not None:
            st.session_state["Engine_Eff"] = float(engine_eff)
            st.session_state["Idle_Fuel"] = float(idle_fuel)

        st.success("Estimated parameters saved to session. Switch to Physics or Surrogates tab and Run simulation.")

    # Show estimated values (read-only)
    st.markdown("### Computed values (session)")
    df_show = {
        "MASS (kg)": st.session_state["MASS"],
        "RRC": st.session_state["RRC"],
        "AUX_kW (kW)": st.session_state["AUX_kW"],
        "HW (m/s)": st.session_state["HW"],
        "MR_mOhm (mΩ)": st.session_state.get("MR_mOhm", "N/A"),
        "BR_pct (%)": st.session_state.get("BR_pct", "N/A"),
        "SoC_pct (%)": st.session_state.get("SoC_pct", "N/A"),
        "Engine_Eff (%)": st.session_state.get("Engine_Eff", "N/A"),
        "Idle_Fuel (L/h)": st.session_state.get("Idle_Fuel", "N/A"),
    }
    st.table(pd.DataFrame(list(df_show.items()), columns=["Parameter", "Value"]))

# -----------------------
# TAB: Physics Model
# -----------------------
with tabs[1]:
    st.header("🧮 Physics Simulation")

    # upload drive cycle
    uploaded_file = st.file_uploader("Upload drive cycle CSV (Time_s & Rep_Speed_mps)", type=["csv"])
    if uploaded_file is not None:
        df_cycle = load_cycle(uploaded_file)
        st.session_state["uploaded_cycle"] = df_cycle
        st.success(f"Drive cycle loaded ({len(df_cycle)} samples).")
        with st.expander("View drive cycle data"):
            st.dataframe(df_cycle.head())

    st.subheader("Simulation Inputs (from session — edit as needed)")
    # Show editable inputs that read from session_state
    cols = st.columns(2)
    with cols[0]:
        st.session_state["MASS"] = st.number_input("Vehicle Mass (kg)", 500.0, 8000.0, value=float(st.session_state["MASS"]), step=10.0)
        st.session_state["RRC"] = st.number_input("Rolling Resistance (RRC)", 0.005, 0.02, value=float(st.session_state["RRC"]), step=0.001)
        st.session_state["AUX_kW"] = st.number_input("Auxiliary load (kW)", 0.0, 10.0, value=float(st.session_state["AUX_kW"]), step=0.1)
        st.session_state["HW"] = st.number_input("Headwind (m/s)", -5.0, 5.0, value=float(st.session_state["HW"]), step=0.5)
        st.session_state["Ta"] = st.number_input("Ambient Temperature (°C)", 0.0, 50.0, value=float(st.session_state["Ta"]), step=1.0)
    with cols[1]:
        if vehicle_type.startswith("Electric"):
            st.session_state["Tb"] = st.number_input("Battery Temperature (°C)", 0.0, 50.0, value=float(st.session_state["Tb"]), step=1.0)
            st.session_state["SoC_pct"] = st.number_input("State of Charge (%)", 0.0, 100.0, value=float(st.session_state["SoC_pct"]), step=1.0)
            st.session_state["BAge_pct"] = st.number_input("Battery Age (%)", 0.0, 100.0, value=float(st.session_state["BAge_pct"]), step=1.0)
            st.session_state["MR_mOhm"] = st.number_input("Motor Resistance (mΩ)", 20.0, 200.0, value=float(st.session_state["MR_mOhm"]), step=0.5)
            st.session_state["BR_pct"] = st.number_input("Battery Resistance Growth (%)", 0.0, 300.0, value=float(st.session_state["BR_pct"]), step=1.0)
        else:
            st.session_state["Engine_Eff"] = st.number_input("Engine Efficiency (%)", 10.0, 50.0, value=float(st.session_state.get("Engine_Eff", 30.0)), step=0.5)
            st.session_state["Idle_Fuel"] = st.number_input("Idle Fuel (L/h)", 0.0, 5.0, value=float(st.session_state.get("Idle_Fuel", 0.8)), step=0.1)
            st.session_state["Displacement_L"] = st.number_input("Engine displacement (L)", 0.6, 8.0, value=float(st.session_state.get("Displacement_L", 1.6)), step=0.1)
            st.session_state["AFR"] = st.number_input("Air-Fuel Ratio", 10.0, 20.0, value=float(st.session_state.get("AFR", 14.7)), step=0.1)

    # Run physics simulation for EV only (we will use a simplified ICE model placeholder)
    if st.button("Run Physics Simulation"):
        if st.session_state["uploaded_cycle"] is None:
            st.warning("Please upload a drive cycle first.")
        else:
            params = {
                "MASS": st.session_state["MASS"],
                "HW": st.session_state["HW"],
                "RRC": st.session_state["RRC"],
                "Ta": st.session_state["Ta"],
                "Tb": st.session_state.get("Tb", st.session_state["Ta"]),
                "SoC_pct": st.session_state.get("SoC_pct", 80.0),
                "BAge_pct": st.session_state.get("BAge_pct", 0.0),
                "MR_mOhm": st.session_state.get("MR_mOhm", 55.0),
                "AUX_kW": st.session_state["AUX_kW"],
                "BR_pct": st.session_state.get("BR_pct", 100.0),
            }
            # Call existing EV physics integrator for EV
            if vehicle_type.startswith("Electric"):
                net_wh, energy_kwh_per_km, regen_pct, distance_km = integrate_energy_for_cycle(st.session_state["uploaded_cycle"], params)
                result = {
                    "energy_kwh": net_wh,
                    "energy_kwh_per_km": energy_kwh_per_km,
                    "regen_pct": regen_pct,
                    "distance_km": distance_km,
                }
                st.session_state["last_sim_result"] = result
                st.success("Physics simulation (EV) complete.")
                st.metric("Energy (kWh/km)", f"{energy_kwh_per_km:.3f}")
                st.metric("Regen (%)", f"{regen_pct:.2f}")
            else:
                # Placeholder: simple physics approx for ICE (fuel L/100km)
                # This is a simple empirical estimate: fuel L/100km ≈ c1 * (mass/1000) + c2 * aux + c3 * (1/engine_eff)
                mass = st.session_state["MASS"]
                aux = st.session_state["AUX_kW"]
                eff = max(1e-3, st.session_state.get("Engine_Eff", 30.0))
                # coarse formula — replace with your physics model later
                fuel_l_per_100km = 0.8 * (mass / 1000.0) * (30.0 / eff) + 0.05 * aux * 100
                co2_g_per_km = fuel_l_per_100km * 2392 / 100.0  # petrol ~2392 g per liter (rough)
                st.session_state["last_sim_result"] = {
                    "fuel_l_per_100km": fuel_l_per_100km,
                    "co2_g_per_km": co2_g_per_km
                }
                st.success("Physics estimate (ICE) complete.")
                st.metric("Fuel (L/100 km)", f"{fuel_l_per_100km:.2f}")
                st.metric("CO2 (g/km)", f"{co2_g_per_km:.0f}")

    # show time-series or last results
    if st.session_state.get("uploaded_cycle") is not None:
        fig, ax = plt.subplots(figsize=(8, 3))
        cdf = st.session_state["uploaded_cycle"]
        ax.plot(cdf["time_s"], cdf["speed_m_s"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Drive cycle speed profile")
        st.pyplot(fig)

# -----------------------
# TAB: Surrogate Models (PCE & DNN)
# -----------------------
with tabs[2]:
    st.header("🤖 Surrogate Predictions")
    st.write("Enter or confirm the 10 input parameters (filled from the estimator or edit manually).")

    feat_cols = st.columns(2)
    with feat_cols[0]:
        MASS = st.number_input("MASS (kg)", 500.0, 8000.0, value=float(st.session_state["MASS"]))
        HW = st.number_input("HW (m/s)", -10.0, 10.0, value=float(st.session_state["HW"]))
        RRC = st.number_input("RRC", 0.005, 0.02, value=float(st.session_state["RRC"]))
        Ta = st.number_input("Ta (°C)", -10.0, 50.0, value=float(st.session_state["Ta"]))
        Tb = st.number_input("Tb (°C)", 0.0, 50.0, value=float(st.session_state.get("Tb", st.session_state["Ta"])))
    with feat_cols[1]:
        SoC_pct = st.number_input("SoC_pct (%)", 0.0, 100.0, value=float(st.session_state.get("SoC_pct", 80.0)))
        BAge_pct = st.number_input("BAge_pct (%)", 0.0, 100.0, value=float(st.session_state.get("BAge_pct", 0.0)))
        MR_mOhm = st.number_input("MR_mOhm", 10.0, 200.0, value=float(st.session_state.get("MR_mOhm", 55.0)))
        AUX_kW = st.number_input("AUX_kW", 0.0, 10.0, value=float(st.session_state["AUX_kW"]))
        BR_pct = st.number_input("BR_pct (%)", 50.0, 300.0, value=float(st.session_state.get("BR_pct", 100.0)))

    input_df = pd.DataFrame([{
        "MASS": MASS, "HW": HW, "RRC": RRC, "Ta": Ta, "Tb": Tb,
        "SoC_pct": SoC_pct, "BAge_pct": BAge_pct, "MR_mOhm": MR_mOhm,
        "AUX_kW": AUX_kW, "BR_pct": BR_pct
    }])

    if st.button("Run Surrogate Predictions"):
        with st.spinner("Predicting..."):
            try:
                pce_out = predict_pce(input_df)
            except Exception as e:
                pce_out = pd.DataFrame([{"energy_kwh_per_km": None, "regen_pct": None}])
                st.error(f"PCE prediction failed: {e}")

            try:
                dnn_out = predict_dnn(input_df)
            except Exception as e:
                dnn_out = pd.DataFrame([{"energy_kwh_per_km": None, "regen_pct": None}])
                st.warning(f"DNN not available: {e}")

        st.subheader("Predictions")
        st.write("*PCE:*", pce_out.to_dict(orient="records")[0])
        st.write("*DNN:*", dnn_out.to_dict(orient="records")[0])

# -----------------------
# TAB: Sensitivity Analysis
# -----------------------
with tabs[3]:
    st.header("📊 Sensitivity Analysis (Sobol)")
    model_choice = st.selectbox("Model for sensitivity", ["PCE", "DNN"])
    N = st.slider("Sobol base samples (N)", 128, 1024, 512, step=128)

    if st.button("Run Sobol"):
        with st.spinner("Running Sobol analysis (this can take a minute)..."):
            energy_df, regen_df = run_sobol(model_choice.lower(), N)
        st.subheader("Top factors for energy consumption")
        # show only top 4
        fig1 = plot_sobol(energy_df, title=f"{model_choice} - Energy (top 4)", top_n=4)
        st.pyplot(fig1)
        st.subheader("Top factors for regeneration (or target)")
        fig2 = plot_sobol(regen_df, title=f"{model_choice} - Regen (top 4)", top_n=4)
        st.pyplot(fig2)

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown("<small>Developed by Ayomide B. — Dept. of Mechanical Engineering, OAU</small>", unsafe_allow_html=True)