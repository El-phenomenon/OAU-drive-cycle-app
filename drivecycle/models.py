import os
import pandas as pd
import numpy as np
import joblib
# For TensorFlow Lite runtime
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

PCE_MODEL_PATH = os.path.join(RESULTS_DIR, "pce_surrogate.pkl")
DNN_TFLITE_PATH = os.path.join(RESULTS_DIR, "dnn_surrogate.tflite")
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
    """Load TensorFlow Lite model (converted from DNN)."""
    if os.path.exists(DNN_TFLITE_PATH):
        interpreter = Interpreter(model_path=DNN_TFLITE_PATH)
        interpreter.allocate_tensors()
        return interpreter
    return None


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
        raise RuntimeError("PCE surrogate model not found.")
    poly, model = pce

    X = X_df[FEATURE_ORDER].values
    X_poly = poly.transform(X)
    Y_pred = model.predict(X_poly)

    return pd.DataFrame(
        Y_pred,
        columns=["energy_kwh_per_km", "regen_pct"]
    )

# ------------------------------------------------------------
# DNN Prediction (TensorFlow Lite)
# ------------------------------------------------------------
def predict_dnn(X_df):
    """Predict using DNN surrogate (TensorFlow Lite)."""
    interpreter = _load_dnn()
    if interpreter is None:
        raise RuntimeError("TFLite DNN model not found.")

    scaler = _load_scaler()
    X = X_df[FEATURE_ORDER].values.astype(np.float32)
    if scaler is not None:
        X = scaler.transform(X).astype(np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], X)
    interpreter.invoke()
    Y_pred = interpreter.get_tensor(output_details[0]["index"])

    return pd.DataFrame(
        Y_pred,
        columns=["energy_kwh_per_km", "regen_pct"]
    )