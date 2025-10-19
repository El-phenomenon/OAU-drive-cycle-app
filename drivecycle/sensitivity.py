# drivecycle/sensitivity.py
import os
import pandas as pd
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt

from .models import _load_pce, _load_dnn, _load_scaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

FACTOR_NAMES = [
    "MASS", "HW", "RRC", "Ta", "Tb",
    "SoC_pct", "BAge_pct", "MR_mOhm", "AUX_kW", "BR_pct"
]
BOUNDS = [
    [2500.0, 6000.0],
    [-5.0, 5.0],
    [0.006, 0.012],
    [10.0, 35.0],
    [18.0, 35.0],
    [50.0, 100.0],
    [0.0, 20.0],
    [50.0, 58.0],
    [1.0, 5.0],
    [100.0, 150.0],
]
PROBLEM = {"num_vars": len(FACTOR_NAMES), "names": FACTOR_NAMES, "bounds": BOUNDS}


def run_sobol(model_type="pce", N=512):
    """
    Run small Sobol analysis on PCE or DNN surrogate.
    Returns two DataFrames: (energy_df, regen_df)
    """
    if model_type.lower() == "pce":
        pce = _load_pce()
        if pce is None:
            raise RuntimeError("PCE model not found.")
        poly, model = pce

        def predict(X):
            return model.predict(poly.transform(X))
    else:
        dnn = _load_dnn()
        if dnn is None:
            raise RuntimeError("DNN model not found.")
        scaler = _load_scaler()

        def predict(X):
            Xs = scaler.transform(X) if scaler is not None else X
            return dnn.predict(Xs)

    # Sobol sampling
    X = saltelli.sample(PROBLEM, N, calc_second_order=False)
    Y = predict(X)
    y_e = Y[:, 0]
    y_r = Y[:, 1]

    Si_e = sobol.analyze(PROBLEM, y_e, calc_second_order=False)
    Si_r = sobol.analyze(PROBLEM, y_r, calc_second_order=False)

    def to_df(Si):
        return pd.DataFrame({
            "factor": FACTOR_NAMES,
            "S1": Si["S1"],
            "ST": Si["ST"]
        }).sort_values("ST", ascending=False)

    return to_df(Si_e), to_df(Si_r)


def plot_sobol(df, title="Sobol Sensitivity", top_n=4):
    """
    Plot Sobol indices for the top N most influential factors.
    """
    # Select top N factors based on Total Effect (ST)
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