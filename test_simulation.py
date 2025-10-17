from drivecycle.io import load_cycle
from drivecycle.simulation import integrate_energy_for_cycle

# 1. Load your real drive cycle
df = load_cycle("data/final_drive_cycle.csv")

# 2. Define vehicle parameters
params = {
    "MASS": 3300,
    "RRC": 0.015,
    "AUX_kW": 1.0,
    "HW": 0.0,
    "Tb": 25,
    "MR_mOhm": 55,
    "BR_pct": 100
}

# 3. Run the simulation
result = integrate_energy_for_cycle(df, params)

# 4. Display the results
print("\n--- Physics-based ECB Simulation Results ---")
for k, v in result.items():
    print(f"{k:20s}: {v:.4f}")