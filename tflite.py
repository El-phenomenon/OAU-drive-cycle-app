import tensorflow as tf
import os

# Path to your saved DNN model
MODEL_PATH = os.path.join("results", "dnn_surrogate.h5")
TFLITE_PATH = os.path.join("results", "dnn_surrogate.tflite")

# --- Load model safely ---
print("🔍 Loading DNN model from:", MODEL_PATH)

# Use compile=False to avoid metric deserialization problems
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# --- Convert to TensorFlow Lite ---
print("⚙ Converting model to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# --- Save .tflite file ---
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete.")
print("Saved TFLite model to:", TFLITE_PATH)