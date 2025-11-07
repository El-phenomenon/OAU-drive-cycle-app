# train_pce_ice.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from SALib.sample import saltelli
import joblib
import argparse
import math

# -----------------------------
# Simple ICE physics (same logic as app)
# -----------------------------
def integrate_fuel_for_cycle(cycle_df, params):
    rho_air = 1.225
    g = 9.81
    LHV_J_per_L = 34.2e6
    CO2_g_per_L = 2392.0

    t = cycle_df["time_s"].values
    v = cycle_df["speed_m_s"].values
    # safe dt (append last interval)
    dt = np.diff(t, append=t[-1] + (t[-1]-t[-2] if len(t)>1 else 1.0))

    mass = float(params.get("MASS", 1500.0))
    RRC = float(params.get("RRC", 0.01))
    Cd = float(params.get("Cd", 0.35))
    A = float(params.get("A", 2.2))
    hw = float(params.get("HW", 0.0))
    aux_w = float(params.get("AUX_kW", 0.0)) * 1000.0
    engine_eff = max(0.01, float(params.get("Engine_Eff", 30.0)) / 100.0)
    idle_fuel_lph = float(params.get("Idle_Fuel_Lph", 0.8))

    total_fuel_l = 0.0
    distance_m = 0.0

    prev_v = v[0] if len(v)>0 else 0.0
    for i in range(len(v)):
        vi = max(0.0, v[i])
        ai = (vi - prev_v) / (dt[i] if dt[i] > 0 else 1.0) if i>0 else 0.0
        prev_v = vi

        v_air = max(0.0, vi + hw)
        F_inertia = mass * ai
        F_roll = mass * g * RRC
        F_aero = 0.5 * rho_air * Cd * A * (v_air ** 2)
        F_trac = F_inertia + F_roll + F_aero
        P_wheel = F_trac * vi

        if P_wheel >= 0:
            driveline_eff = 0.9
            P_engine = (P_wheel / max(1e-6, driveline_eff)) + aux_w
            fuel_power_needed_Js = P_engine / max(1e-6, engine_eff)
            fuel_l_per_s = fuel_power_needed_Js / LHV_J_per_L
            fuel_l = fuel_l_per_s * dt[i]
        else:
            idle_l_per_s = idle_fuel_lph / 3600.0
            fuel_l = idle_l_per_s * dt[i]

        total_fuel_l += fuel_l
        distance_m += vi * dt[i]

    distance_km = distance_m / 1000.0 if distance_m > 0 else np.nan
    fuel_l_per_100km = (total_fuel_l / distance_km * 100.0) if distance_km > 0 else np.nan
    co2_g_per_km = (total_fuel_l * CO2_g_per_L / distance_km) if distance_km > 0 else np.nan

    return total_fuel_l, fuel_l_per_100km, co2_g_per_km, distance_km

# -----------------------------
# Sampling ranges for ICE factors
# NOTE: match these ranges to what you used elsewhere (sensitivity)
# -----------------------------
FACTORS = [
    ("MASS", 1000.0, 6000.0),        # kg
    ("HW", -5.0, 5.0),              # m/s
    ("RRC", 0.006, 0.012),          # rolling resistance
    ("Cd", 0.25, 0.45),             # drag coefficient
    ("AUX_kW", 0.0, 5.0),           # kW
    ("Engine_Eff", 20.0, 36.0),     # percent
    ("Idle_Fuel_Lph", 0.4, 1.5),    # L/h
]

def main(args):
    # paths
    base = os.path.abspath(os.path.dirname(__file__))
    results_dir = os.path.join(base, "results")
    os.makedirs(results_dir, exist_ok=True)
    sim_csv = os.path.join(results_dir, "pce_ice_training_samples.csv")
    model_path = os.path.join(results_dir, "pce_surrogate_ice.pkl")
    metrics_path = os.path.join(results_dir, "pce_ice_metrics.txt")

    # load cycle
    cycle_path = os.path.join(base, "data", "final_drive_cycle.csv")
    cycle_df = pd.read_csv(cycle_path)
    # confirm required cols
    if not {"time_s", "speed_m_s"}.issubset(set(cycle_df.columns.str.lower())):
        # try with original names if needed
        cycle_df.columns = [c.strip().lower() for c in cycle_df.columns]
    # ensure lowercase names
    cycle_df.columns = [c.strip().lower() for c in cycle_df.columns]
    # unify names
    if "rep_speed_mps" in cycle_df.columns:
        cycle_df = cycle_df.rename(columns={"rep_speed_mps":"speed_m_s"})
    if "time_s" not in cycle_df.columns or "speed_m_s" not in cycle_df.columns:
        raise RuntimeError("Drive cycle CSV must contain 'time_s' and 'speed_m_s' columns.")

    # create SALib Saltelli sample (we will sample N base samples, sample size grows by factor)
    N = args.samples  # base N
    problem = {"num_vars": len(FACTORS), "names": [f[0] for f in FACTORS], "bounds": [[f[1], f[2]] for f in FACTORS]}
    print("Sampling using Saltelli (this may take time)...")
    X = saltelli.sample(problem, N, calc_second_order=False)
    # saltelli returns shape (NB) * (2D+2) — but we are using this for good coverage

    print("Total samples:", X.shape[0])
    # Evaluate ICE physics for each sample
    out_rows = []
    for i, row in enumerate(X):
        params = {name: float(val) for name, val in zip(problem["names"], row)}
        # add default frontal area if needed (integrator uses A=2.2 by default)
        total_fuel_l, fuel_100, co2_km, dist_km = integrate_fuel_for_cycle(cycle_df, params)
        out_rows.append({**params, "fuel_l_per_100km": fuel_100, "co2_g_per_km": co2_km, "distance_km": dist_km})
        if (i+1) % 500 == 0:
            print(f" Evaluated {i+1}/{X.shape[0]} samples...")

    df = pd.DataFrame(out_rows)
    df.to_csv(sim_csv, index=False)
    print("Saved simulation results to", sim_csv)

    # Prepare X and Y for regression
    X_mat = df[[c[0] for c in FACTORS]].values
    Y_mat = df[["fuel_l_per_100km", "co2_g_per_km"]].values

    # Fit polynomial surrogate
    degree = args.degree
    print(f"Fitting Polynomial surrogate (degree {degree}) ...")
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_mat)
    model = LinearRegression()
    model.fit(X_poly, Y_mat)
    Y_pred = model.predict(X_poly)

    # Metrics
    r2_fuel = r2_score(Y_mat[:,0], Y_pred[:,0])
    mae_fuel = mean_absolute_error(Y_mat[:,0], Y_pred[:,0])

    r2_co2 = r2_score(Y_mat[:,1], Y_pred[:,1])
    mae_co2 = mean_absolute_error(Y_mat[:,1], Y_pred[:,1])

    # Save model
    joblib.dump((poly, model), model_path)
    print("Saved PCE surrogate to", model_path)

    # Write metrics
    with open(metrics_path, "w") as f:
        f.write("ICE PCE Surrogate Metrics\n")
        f.write("=========================\n")
        f.write(f"Samples: {X.shape[0]}\n")
        f.write(f"Degree: {degree}\n\n")
        f.write(f"Fuel (L/100km): R2={r2_fuel:.6f}, MAE={mae_fuel:.6f} L/100km\n")
        f.write(f"CO2 (g/km): R2={r2_co2:.6f}, MAE={mae_co2:.6f} g/km\n")
    print("Metrics saved to", metrics_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=256, help="base Saltelli samples (power of two recommended)")
    parser.add_argument("--degree", type=int, default=3, help="polynomial degree")
    args = parser.parse_args()
    main(args)