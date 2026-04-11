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
with st.expander("### 🚘 About This Application (Click to expand/collapse)", expanded=False):
    st.markdown("""

This application is designed to simulate and analyze how vehicles consume energy or fuel under real driving conditions within Obafemi Awolowo University (OAU) campus.

It uses a **representative drive cycle**, a recorded pattern of speed over time which reflects how vehicles actually move in a typical Nigerian environment (including stops, accelerations, and varying speeds).

---

### 🔍 What This App Does

This tool helps you understand how different factors affect vehicle performance:

#### ⚡ For Electric Vehicles (EVs):
- Estimates **energy consumption (kWh/km)**  
- Calculates **regenerative braking efficiency (%)**

👉 *Regenerative braking* is the process where the vehicle recovers energy during braking and stores it back in the battery instead of wasting it as heat.

---

#### ⛽ For Petrol Vehicles (ICE):
- Estimates **fuel consumption (L/100 km)**  
- Calculates **CO₂ emissions (g/km)**  

---

### 🧠 Modeling Approach

The app combines two powerful approaches:

- **Physics-Based Model**  
  Uses real engineering equations to simulate forces like:
  - Aerodynamic drag  
  - Rolling resistance  
  - Vehicle mass effects  

- **Surrogate Model (PCE)**  
  A fast approximation model that provides instant predictions without running full simulations.

---

### 📊 Sensitivity Analysis

The app performs a **Sobol sensitivity analysis** to determine:

👉 *Which factors matter the most*

Examples include:
- Vehicle mass  
- Tyre resistance  
- Auxiliary loads (e.g., AC)  
- Battery or engine characteristics  

---

### 💡 Decision Support

Based on the analysis, the app provides:

👉 **Simple, actionable recommendations tailored to your vehicle setup**

These help you understand:
- Where energy is being lost  
- What changes can improve efficiency  
- How to operate the vehicle more economically  

---

### How to Use the App

1. **Set up your vehicle** (EV or Petrol)
2. **Run the physics model** to simulate performance  
3. **Use surrogate models** for quick predictions  
4. **Run sensitivity analysis** to identify key drivers  
5. **View decision support recommendations** for optimization  

---

### Why This Matters

This tool can support:
- Better **vehicle design decisions**
- Smarter **driving habits**
- Planning for **EV adoption and charging infrastructure**
- Understanding **energy efficiency in real-world Nigerian conditions**

---
""")
# Sidebar branding
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
# DECISION SUPPORT FUNCTION
# ------------------------------------------------------------
# DECISION SUPPORT FUNCTION (FIXED)
# ------------------------------------------------------------
def generate_recommendations(main_df, base_params):
    recommendations = []

    sens_col = "S1" if "S1" in main_df.columns else main_df.columns[0]

    # Sort top 4
    top_factors = main_df.sort_values(by=sens_col, ascending=False).head(4)

    for _, row in top_factors.iterrows():

        factor = row["factor"]   # ✅ FIXED
        factor_clean = factor.lower()

        # -------------------------------
        # GENERAL PARAMETERS
        # -------------------------------
        if "mass" in factor_clean:
            recommendations.append(
                f"The mass of your vehicle ({base_params.get('MASS', 'N/A'):.0f} kg) is a major factor in its energy demand. "
    "A heavier vehicle requires more energy to accelerate and maintain motion. "
    "Reducing passenger or cargo load can noticeably improve efficiency."
            )

        elif "rrc" in factor_clean:
            recommendations.append(
                f"The rolling resistance coefficient of your tyres ({base_params.get('RRC', 'N/A')}) affects how much energy is lost as the vehicle moves. "
    "Higher resistance means more energy is required to keep the car in motion. "
    "Ensuring proper tyre inflation and using low-resistance tyres can improve efficiency."
            )

        elif "hw" in factor_clean:
            recommendations.append(
                 f"Under your current driving conditions, the effective headwind ({base_params.get('HW', 'N/A')} m/s) contributes to aerodynamic drag. "
    "This increases the effort required to move the vehicle forward. "
    "Driving at steady, moderate speeds can help reduce these aerodynamic losses."
            )

        elif "aux_kw" in factor_clean:
            recommendations.append(
                f"Your auxiliary systems are drawing about {base_params.get('AUX_kW', 'N/A')} kW of power. "
    "This includes air conditioning and onboard electronics. "
    "These loads add directly to total energy consumption, "
    "so minimizing unnecessary usage can improve overall vehicle efficiency."
            )

        # ---------------- EV-SPECIFIC ----------------
        elif "tb" in factor_clean:
            recommendations.append(
                f"Your battery temperature ({base_params.get('Tb', 'N/A')} °C) influences both efficiency and battery health. "
    "Operating within an optimal temperature range ensures better energy delivery "
    "and helps maintain long-term battery performance."
            )

        elif "soc_pct" in factor_clean:
            recommendations.append(
                f"Your battery state of charge is currently around {base_params.get('SoC_pct', 'N/A')}%. "
    "Operating within a moderate charge range helps maintain efficient energy usage "
    "and supports better regenerative braking performance."
            )

        elif "bage_pct" in factor_clean:
            recommendations.append(
                f"The battery has an estimated aging level of {base_params.get('BAge_pct', 'N/A')}%. "
    "As the battery ages, its effective capacity reduces, which impacts overall vehicle range and efficiency. "
    "Monitoring battery health is important for maintaining consistent performance."
            )

        elif "mr_mohm" in factor_clean:
            recommendations.append(
                 f"The motor internal resistance in your system ({base_params.get('MR_mOhm', 'N/A')} mΩ) affects how efficiently electrical energy is converted into motion. "
    "Higher resistance leads to greater energy losses, "
    "so maintaining efficient motor components is key to overall performance."
            )

        elif "br_pct" in factor_clean:
            recommendations.append(
                f"The battery resistance growth factor ({base_params.get('BR_pct', 'N/A')}%) indicates internal degradation effects. "
    "Higher internal resistance reduces efficiency and increases energy losses. "
    "Avoiding excessive heat and overcharging can help slow down this degradation."
            )

        # ---------------- ICE-SPECIFIC ----------------
        elif "cd" in factor_clean:
            recommendations.append(
                f"Your vehicle’s drag coefficient ({base_params.get('Cd', 'N/A')}) affects how it interacts with air at higher speeds. "
    "Higher drag increases fuel consumption, particularly on highways. "
    "Maintaining steady speeds and minimizing unnecessary acceleration helps reduce this effect."
            )

        elif "engine_eff" in factor_clean:
            recommendations.append(
                f"The engine efficiency of your vehicle is estimated at {base_params.get('Engine_Eff', 'N/A')}%. "
    "This determines how effectively fuel is converted into useful motion. "
    "Regular servicing and proper engine tuning can significantly improve fuel economy."
            )

        elif "idle_fuel_lph" in factor_clean:
            recommendations.append(
                 f"Your vehicle consumes approximately {base_params.get('Idle_Fuel_Lph', 'N/A')} L/h while idling. "
    "Extended idling leads to unnecessary fuel consumption without contributing to motion. "
    "Reducing idle time can improve fuel efficiency and lower emissions."
            )

        # ---------------- FALLBACK ----------------
        else:
            recommendations.append(
                f"{factor} plays a noticeable role in vehicle performance. "
                "Optimizing this parameter can contribute to improved efficiency and operation."
            )

    return recommendations
# TABS
# ------------------------------------------------------------
tabs = st.tabs([
    "🚗 Vehicle Setup",
    "🧮 Physics Model",
    "🤖 Surrogate Models",
    "📊 Sensitivity Analysis",
    "💡 Decision Support"
])

# ------------------------------------------------------------
# TAB 1 - Vehicle Setup
# ------------------------------------------------------------
with tabs[0]:
    st.header("🚗 Vehicle Setup & Parameter Estimation")
    fuel_type = st.selectbox("Select Fuel Type", ["Electric Vehicle (EV)", "Internal Combustion Engine Vehicle (ICEV) (Petrol-powered)"])
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
                "Total mass of the vehicle (kg)": total_mass,
                "Rolling resistance coefficient, RRC": rrc,
                "Headwind, HW": hw,
                "Auxiliary Power, AUX_kW": aux,
                "Temperature of Battery Pack, Tb (celsius)": 25.0,
                "State of charge of battery, SoC_pct": 80.0,
                "Battery capacity fade due to aging, BAge_pct": 10.0,
                "Internal resistance of the motor, MR_mOhm": 55.0 if engine_size != "Small" else 50.0,
                "Internal resistance growth of the battery, BR_pct": 100.0 + 5.0,
                "Type": "EV",
            }
        else:
            eff = 30.0 if engine_size == "Standard" else (25.0 if engine_size == "Large" else 35.0)
            idle = 0.8 if engine_size == "Standard" else (1.0 if engine_size == "Large" else 0.6)
            Cd = 0.32 if engine_size == "Small" else (0.36 if engine_size == "Standard" else 0.40)
            params = {
                "Total mass of the vehicle (kg)": total_mass,
                "Rolling Resistance coefficient, RRC": rrc,
                "Headwind, HW": hw,
                "Auxiliary Power, AUX_kW": aux,
                "Drag Coefficient, Cd": Cd,
                "Engine_Eff": eff,
                "Idle_Fuel_Lph": idle,
                "Type": "ICE",
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
                # Map descriptive names to internal short names
                ev_params = {
                    "MASS": params["Total mass of the vehicle (kg)"],
                    "RRC": params["Rolling resistance coefficient, RRC"],
                    "HW": params["Headwind, HW"],
                    "AUX_kW": params["Auxiliary Power, AUX_kW"],
                    "Tb": params["Temperature of Battery Pack, Tb (celsius)"],
                    "SoC_pct": params["State of charge of battery, SoC_pct"],
                    "BAge_pct": params["Battery capacity fade due to aging, BAge_pct"],
                    "MR_mOhm": params["Internal resistance of the motor, MR_mOhm"],
                    "BR_pct": params["Internal resistance growth of the battery, BR_pct"],
                    "Ta": 25.0,
                }
                result = integrate_energy_for_cycle(cycle_df, ev_params)
                st.success("EV simulation complete.")
                st.metric("Energy (kWh/km)", f"{result['energy_kwh_per_km']:.3f}")
                st.metric("Regeneration (%)", f"{result['regen_pct']:.2f}")
            else:
                # Use fuel_params mapping correctly
                fuel_params = {
                    "MASS": params["Total mass of the vehicle (kg)"],
                    "RRC": params["Rolling Resistance coefficient, RRC"],
                    "HW": params["Headwind, HW"],
                    "AUX_kW": params["Auxiliary Power, AUX_kW"],
                    "Cd": params["Drag Coefficient, Cd"],
                    "Engine_Eff": params["Engine_Eff"],
                    "Idle_Fuel_Lph": params["Idle_Fuel_Lph"],
                }
                total_fuel_l, fuel_100, co2_km, dist_km = integrate_fuel_for_cycle(cycle_df, fuel_params)
                st.success("ICE simulation complete.")
                st.metric("Fuel (L/100 km)", f"{fuel_100:.3f}")
                st.metric("CO₂ emission (g/km)", f"{co2_km:.1f}")

        st.subheader("Reference Drive Cycle (Speed vs Time)")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(cycle_df["time_s"], cycle_df["speed_m_s"], color="blue")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.grid(True)
        st.pyplot(fig)

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
                "BR_pct": p["Internal resistance growth of the battery, BR_pct"],
            }])
            if st.button("Run EV Surrogate Prediction (PCE)"):
                try:
                    pce_out = predict_pce(input_df)
                    st.success("EV Surrogate Prediction complete.")
                    st.write(pce_out)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            input_df = pd.DataFrame([{
                "MASS": p["Total mass of the vehicle (kg)"],
                "HW": p["Headwind, HW"],
                "RRC": p["Rolling Resistance coefficient, RRC"],
                "Cd": p["Drag Coefficient, Cd"],
                "AUX_kW": p["Auxiliary Power, AUX_kW"],
                "Engine_Eff": p["Engine_Eff"],
                "Idle_Fuel_Lph": p["Idle_Fuel_Lph"],
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
                "Ta": 25.0,
            }
        else:
            base_params = {
                "MASS": params["Total mass of the vehicle (kg)"],
                "HW": params["Headwind, HW"],
                "RRC": params["Rolling Resistance coefficient, RRC"],
                "Cd": params["Drag Coefficient, Cd"],
                "AUX_kW": params["Auxiliary Power, AUX_kW"],
                "Engine_Eff": params["Engine_Eff"],
                "Idle_Fuel_Lph": params["Idle_Fuel_Lph"],
            }

        if st.button("Run Sensitivity Analysis"):
            try:
                main_df, aux_df = run_sobol(model_type, N, base_params)
                st.session_state["sensitivity_main"] = main_df
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
# TAB 5 - DECISION SUPPORT
# ------------------------------------------------------------
with tabs[4]:
    st.header("💡 Decision Support System")

    if "sensitivity_main" not in st.session_state:
        st.warning("Run Sensitivity Analysis first to generate recommendations.")
    else:
        st.subheader("Actionable Insights")
        recs = generate_recommendations(
    st.session_state["sensitivity_main"],
    base_params   # 🔥 IMPORTANT
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