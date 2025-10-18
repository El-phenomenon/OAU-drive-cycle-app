import tensorflow as tf
import tf2onnx
import os
MODEL_PATH = "results/dnn_surrogate.h5"
OUTPUT_PATH = "results/dnn_surrogate.onnx"
print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Converting to ONNX format...")
onnx_model, _ = tf2onnx.convert.from_keras(model)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"✅ Model successfully exported to {OUTPUT_PATH}")