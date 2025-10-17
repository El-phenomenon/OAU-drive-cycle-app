from drivecycle.models import predict_pce, predict_dnn, FEATURE_ORDER
import pandas as pd

# Create a sample input row
sample = pd.DataFrame([{
    "MASS": 3300, "HW": 0, "RRC": 0.009, "Ta": 25, "Tb": 25,
    "SoC_pct": 80, "BAge_pct": 5, "MR_mOhm": 55, "AUX_kW": 2, "BR_pct": 120
}])

print("📦 Running PCE prediction...")
pce_out = predict_pce(sample)
print(pce_out)

print("\n🧠 Running DNN prediction...")
dnn_out = predict_dnn(sample)
print(dnn_out)