# drivecycle/models.py
import os
import joblib
import numpy as np
import pandas as pd

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

PCE_PATH = os.path.join(RESULTS_DIR, "pce_surrogate.pkl")
DNN_PATH = os.path.join(RESULTS_DIR, "dnn_surrogate.h5")

SCALER_CANDIDATES = [
    os.path.join(RESULTS_DIR, "input_scaler.pkl"),
    os.path.join(RESULTS_DIR, "scaler.pkl"),
    os.path.join(RESULTS_DIR, "input_scaler.pkl"),
    os.path.join(RESULTS_DIR, "scaler.pkl"),
]

FEATURE_ORDER = [
    "MASS", "HW", "RRC", "Ta", "Tb",
    "SoC_pct", "BAge_pct", "MR_mOhm", "AUX_kW", "BR_pct"
]

_pce = None
_dnn = None
_scaler = None


def _load_pce():
    global _pce
    if _pce is None and os.path.exists(PCE_PATH):
        _pce = joblib.load(PCE_PATH)
    return _pce


def _load_dnn():
    global _dnn
    if _dnn is None and load_model is not None and os.path.exists(DNN_PATH):
        _dnn = load_model(DNN_PATH, compile=False)
    return _dnn


def _load_scaler():
    global _scaler
    if _scaler is None:
        for p in SCALER_CANDIDATES:
            if os.path.exists(p):
                try:
                    _scaler = joblib.load(p)
                    break
                except Exception:
                    continue
    return _scaler


def _prepare_X(X_df):
    missing = [c for c in FEATURE_ORDER if c not in X_df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return X_df[FEATURE_ORDER].astype(float).values


# ---------- PCE Predictor ----------
def predict_pce(X_df):
    pce = _load_pce()
    X = _prepare_X(X_df)
    if pce is None:
        # fallback (demo)
        mass = X[:, 0]
        aux = X[:, 8]
        energy = 0.001 * mass + 0.1 * aux
        regen = np.clip(10 - 0.001 * mass + 0.05 * aux, 0, 100)
        return pd.DataFrame({"energy_kwh_per_km": energy, "regen_pct": regen})

    poly, model = pce
    X_poly = poly.transform(X)
    preds = model.predict(X_poly)
    preds = np.asarray(preds)
    if preds.ndim == 1:
        preds = np.vstack([preds, np.zeros_like(preds)]).T
    return pd.DataFrame(preds, columns=["energy_kwh_per_km", "regen_pct"])


# ---------- DNN Predictor ----------
def predict_dnn(X_df):
    dnn = _load_dnn()
    scaler = _load_scaler()
    X = _prepare_X(X_df)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass

    if dnn is None:
        return predict_pce(X_df)

    preds = dnn.predict(X)
    preds = np.asarray(preds)
    if preds.ndim == 1:
        preds = np.vstack([preds, np.zeros_like(preds)]).T
    return pd.DataFrame(preds, columns=["energy_kwh_per_km", "regen_pct"])