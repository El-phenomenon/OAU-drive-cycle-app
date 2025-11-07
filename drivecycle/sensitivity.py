# drivecycle/sensitivity.py
import os
import pandas as pd
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
from .models import predict_pce, predict_pce_ice, FEATURE_ORDER, FEATURE_ORDER_ICE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ============================================================
# Generic Sobol runner that works for both EV and ICE
# ============================================================
def run_sobol(model_type: str, N: int = 512, base_params: dict = None):
    """
    Run Sobol sensitivity analysis based on user's input parameters.
    Args:
        model_type (str): "EV" or "ICE"
        N (int): number of Sobol samples
        base_params (dict): user's factor values from app input
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]
        -> (Energy or Fuel factors, Regen or CO2 factors)
    """

    if model_type.upper() == "EV":
        factors = FEATURE_ORDER
        model_fn = predict_pce
    else:
        factors = FEATURE_ORDER_ICE
        model_fn = predict_pce_ice

    # Define problem
    names = factors
    bounds = []
    for f in names:
        val = float(base_params.get(f, 1.0))
        # ±20% variation around user input (reasonable neighborhood)
        delta = abs(val * 0.2) if val != 0 else 0.1
        bounds.append([val - delta, val + delta])

    problem = {"num_vars": len(names), "names": names, "bounds": bounds}

    # Generate Sobol samples
    X = saltelli.sample(problem, N, calc_second_order=False)
    X_df = pd.DataFrame(X, columns=names)

    # Predict outputs
    preds = model_fn(X_df).to_numpy()

    # Separate outputs based on model type
    if model_type.upper() == "EV":
        y_main = preds[:, 0]  # energy (kWh/km)
        y_aux = preds[:, 1]   # regen (%)
    else:
        y_main = preds[:, 0]  # fuel (L/100km)
        y_aux = preds[:, 1]   # CO2 (g/km)

    # Run Sobol analysis
    Si_main = sobol.analyze(problem, y_main, calc_second_order=False)
    Si_aux = sobol.analyze(problem, y_aux, calc_second_order=False)

    # Convert to dataframe
    def to_df(Si):
        return pd.DataFrame({
            "factor": names,
            "S1": Si["S1"],
            "ST": Si["ST"]
        }).sort_values("ST", ascending=False)

    df_main = to_df(Si_main)
    df_aux = to_df(Si_aux)

    return df_main, df_aux


# ============================================================
# Plot function
# ============================================================
def plot_sobol(df, title="Sobol Sensitivity", top_n=4):
    """Plot Sobol indices for top N factors."""
    df_top = df.sort_values("ST", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df_top))
    ax.bar(x - 0.2, df_top["S1"] * 100, width=0.4, label="S1 (First-order)")
    ax.bar(x + 0.2, df_top["ST"] * 100, width=0.4, label="ST (Total)")

    ax.set_xticks(x)
    ax.set_xticklabels(df_top["factor"], rotation=30, ha="right")
    ax.set_ylabel("Contribution (%)")
    ax.legend()
    ax.set_title(f"{title} (Top {top_n} Factors)")
    plt.tight_layout()
    return fig