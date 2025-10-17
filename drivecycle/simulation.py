# drivecycle/simulation.py
import numpy as np

# --- constants you can adjust later ---
NOMINALS = {
    "CdA": 2.2,          # drag area (m²)
    "rho_air": 1.225,    # air density (kg/m³)
    "g": 9.81,           # gravity (m/s²)
    "V_nom": 600.0       # nominal battery voltage (V)
}

def integrate_energy_for_cycle(df_cycle, params):
    """
    Compute energy consumption and regeneration for a drive cycle.
    
    Parameters
    ----------
    df_cycle : DataFrame with 'time_s' and 'speed_m_s'
    params   : dict of physical parameters (mass, RRC, etc.)

    Returns
    -------
    result : dict with energy_kwh_per_km, regen_pct, distance_km, etc.
    """
    # Unpack nominal constants
    g = NOMINALS["g"]
    rho = NOMINALS["rho_air"]
    CdA = NOMINALS["CdA"]

    # Parameters (with defaults)
    mass = float(params.get("MASS", 3300.0))        # kg
    RRC  = float(params.get("RRC", 0.015))          # rolling resistance coeff
    AUX  = float(params.get("AUX_kW", 1.0)) * 1000  # W
    HW   = float(params.get("HW", 0.0))             # headwind (m/s)
    Tb   = float(params.get("Tb", 25.0))            # °C
    MR   = float(params.get("MR_mOhm", 55.0)) / 1000 # motor resistance (Ohm)
    BR   = float(params.get("BR_pct", 100.0)) / 100  # battery resistance multiplier

    # Efficiency and correction factors
    motor_eff = 0.94
    inverter_eff = 0.96
    drivetrain_eff = 0.98
    batt_temp_eff = 1.0 - 0.001 * max(0, 25 - Tb)   # lower temp → less efficient

    v = df_cycle["speed_m_s"].to_numpy()
    t = df_cycle["time_s"].to_numpy()
    dt = np.diff(t, append=t[-1] + (t[-1] - t[-2]))  # same length as v

    energy_wh = 0.0
    regen_wh = 0.0
    distance_m = 0.0
    prev_v = v[0]

    for i in range(len(v)):
        vi = v[i]
        ai = (vi - prev_v) / dt[i] if dt[i] > 0 else 0
        prev_v = vi

        # Forces
        F_inertia = mass * ai
        F_roll = mass * g * RRC
        F_aero = 0.5 * rho * CdA * (vi + HW)**2
        F_total = F_inertia + F_roll + F_aero

        P_wheel = F_total * vi

        if P_wheel >= 0:
            eff_chain = motor_eff * inverter_eff * drivetrain_eff * batt_temp_eff
            P_elec = P_wheel / eff_chain + AUX
            I = P_elec / NOMINALS["V_nom"]
            P_loss = (I**2) * MR * BR
            energy_wh += (P_elec + P_loss) * dt[i] / 3600
        else:
            P_brake = -P_wheel
            regen_eff = 0.6 * batt_temp_eff
            P_recovered = regen_eff * P_brake * motor_eff * inverter_eff * drivetrain_eff
            regen_wh += P_recovered * dt[i] / 3600
            # aux load during braking
            energy_wh += AUX * dt[i] / 3600

        distance_m += vi * dt[i]

    # Results
    distance_km = distance_m / 1000
    net_wh = energy_wh - regen_wh
    energy_kwh_per_km = net_wh / distance_km if distance_km > 0 else np.nan
    regen_pct = 100 * regen_wh / (energy_wh + 1e-9)

    return {
        "energy_kwh_per_km": energy_kwh_per_km,
        "regen_pct": regen_pct,
        "distance_km": distance_km,
        "total_energy_Wh": energy_wh,
        "regen_energy_Wh": regen_wh,
        "net_energy_Wh": net_wh
    }