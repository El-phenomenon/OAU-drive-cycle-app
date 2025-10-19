# drivecycle/models.py
import os
import joblib
import numpy as np
import pandas as pd

# Try lightweight TFLite first; fallback to TensorFlow if available
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        import tensorflow as tf
        from tensorflow.lite.python.interpreter import Interpreter
    except Exception:
        Interpreter = None  # No TensorFlow or TFLite

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

PCE_PATH = os.path.join(RESULTS_DIR, "pce_surrogate.pkl")
DNN_TFLITE_PATH = os.path.join(RESULTS_DIR, "dnn_surrogate.tflite")

SCALER_CANDIDATES = [
    os.path.join(RESULTS_DIR, "input_scaler.pkl"),
    os.path.join(RESULTS_DIR, "scaler.pkl"),
]

FEATURE_ORDER = [
    "MASS", "HW", "RRC", "Ta", "Tb",
    "SoC_pct", "BAge_pct", "MR_mOhm", "AUX_kW", "BR_pct"
]

_pce = None
_scaler = None


# ---------- Loaders ----------
def _load_pce():
    global _pce
    if _pce is None and os.path.exists(PCE_PATH):
        _pce = joblib.load(PCE_PATH)
    return _pce


def _load_dnn():
    """Load TensorFlow Lite model (if runtime available)."""
    if Interpreter is None:
        return None
    if os.path.exists(DNN_TFLITE_PATH):
        try:
            interpreter = Interpreter(model_path=DNN_TFLITE_PATH)
            interpreter.allocate_tensors()
            return interpreter
        except Exception:
            return None
    return None


def _load_scaler():
    global _scaler
    if _scaler is None:
        for path in SCALER_CANDIDATES:
            if os.path.exists(path):
                try:
                    _scaler = joblib.load(path)
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
        # fallback dummy model
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
    interpreter = _load_dnn()
    scaler = _load_scaler()
    X = _prepare_X(X_df).astype(np.float32)
    if scaler is not None:
        try:
            X = scaler.transform(X).astype(np.float32)
        except Exception:
            pass

    if interpreter is None:
        # Fallback to PCE prediction if DNN unavailable
        return predict_pce(X_df)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    try:
        interpreter.set_tensor(input_details[0]["index"], X)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]["index"])
    except Exception:
        return predict_pce(X_df)

    preds = np.asarray(preds)
    if preds.ndim == 1:
        preds = np.vstack([preds, np.zeros_like(preds)]).T

    return pd.DataFrame(preds, columns=["energy_kwh_per_km", "regen_pct"])