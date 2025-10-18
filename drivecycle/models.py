import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf   # use TensorFlow CPU runtime (no tflite_runtime)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

PCE_MODEL_PATH = os.path.join(RESULTS_DIR, "pce_surrogate.pkl")
DNN_MODEL_PATH = os.path.join(RESULTS_DIR, "dnn_surrogate.h5")
SCALER_PATH = os.path.join(RESULTS_DIR, "input_scaler.pkl")

FEATURE_ORDER = [
    "MASS", "HW", "RRC", "Ta", "Tb",
    "SoC_pct", "BAge_pct", "MR_mOhm",
    "AUX_kW", "BR_pct"
]

# ------------------------------------------------------------
# Load Models
# ------------------------------------------------------------
def _load_pce():
    """Load the saved Polynomial Chaos Expansion model."""
    if os.path.exists(PCE_MODEL_PATH):
        return joblib.load(PCE_MODEL_PATH)
    return None


def _load_dnn():
    """Load Keras DNN model (.h5) for inference."""
    if not os.path.exists(DNN_MODEL_PATH):
        raise RuntimeError("❌ DNN .h5 model not found in results/")
    try:
        model = tf.keras.models.load_model(DNN_MODEL_PATH, compile=False)
        return model
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load DNN model: {e}")


def _load_scaler():
    """Load the input scaler if available."""
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None

# ------------------------------------------------------------
# PCE Prediction
# ------------------------------------------------------------
def predict_pce(X_df):
    """Predict using Polynomial Chaos Expansion surrogate."""
    pce = _load_pce()
    if pce is None:
        raise RuntimeError("❌ PCE surrogate model not found.")
    poly, model = pce

    X = X_df[FEATURE_ORDER].values
    X_poly = poly.transform(X)
    Y_pred = model.predict(X_poly)

    return pd.DataFrame(
        Y_pred,
        columns=["energy_kwh_per_km", "regen_pct"]
    )

# ------------------------------------------------------------
# DNN Prediction (Keras)
# ------------------------------------------------------------
def predict_dnn(X_df):
    """Predict using the DNN .h5 model."""
    model = _load_dnn()
    scaler = _load_scaler()

    X = X_df[FEATURE_ORDER].values.astype(np.float32)
    if scaler is not None:
        X = scaler.transform(X).astype(np.float32)

    Y_pred = model.predict(X)
    return pd.DataFrame(
        Y_pred,
        columns=["energy_kwh_per_km", "regen_pct"]
    )