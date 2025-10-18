# drivecycle/models.py
import os
import joblib
import numpy as np
import pandas as pd

# --- For TFLite inference ---
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(_file_)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# File paths
PCE_MODEL_PATH = os.path.join(RESULTS_DIR, "pce_surrogate.pkl")
DNN_TFLITE_PATH = os.path.join(RESULTS_DIR, "dnn_surrogate.tflite")
DNN_H5_PATH = os.path.join(RESULTS_DIR, "dnn_surrogate.h5")
SCALER_PATH = os.path.join(RESULTS_DIR, "input_scaler.pkl")

FEATURE_ORDER = [
    "MASS", "HW", "RRC", "Ta", "Tb",
    "SoC_pct", "BAge_pct", "MR_mOhm", "AUX_kW", "BR_pct"
]

# ------------------------------
# LOADERS
# ------------------------------
def _load_scaler():
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None

def _load_pce():
    if os.path.exists(PCE_MODEL_PATH):
        return joblib.load(PCE_MODEL_PATH)
    return None

# ------------------------------
# PCE Prediction
# ------------------------------
def predict_pce(X_df):
    pce = _load_pce()
    if pce is None:
        raise RuntimeError("PCE model not found.")
    poly, model = pce
    X = X_df[FEATURE_ORDER].values
    Y_pred = model.predict(poly.transform(X))
    return pd.DataFrame(Y_pred, columns=["energy_kwh_per_km", "regen_pct"])

# ------------------------------
# DNN Prediction (supports .tflite)
# ------------------------------

   def _load_dnn():
    import os
    import numpy as np
    model_path_tflite = os.path.join("results", "dnn_surrogate.tflite")

    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_path_tflite)
    except Exception:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path_tflite)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def predict(X):
        X = np.array(X, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], X)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])

    return predict
    return pd.DataFrame(Y_pred, columns=["energy_kwh_per_km", "regen_pct"])