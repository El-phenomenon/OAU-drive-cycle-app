# test_sensitivity_pce.py
from drivecycle.sensitivity import run_sobol, plot_sobol
import matplotlib.pyplot as plt

print("Running PCE sensitivity test...")

# Run Sobol for PCE
energy_df, regen_df = run_sobol(model_type="pce", N=256)

# Display tables
print("\nTop Energy Factors:")
print(energy_df.head())

print("\nTop Regen Factors:")
print(regen_df.head())

# Plot both
fig1 = plot_sobol(energy_df, "PCE Sensitivity - Energy Consumption")
fig2 = plot_sobol(regen_df, "PCE Sensitivity - Regeneration Efficiency")

plt.show()  # show both plots together