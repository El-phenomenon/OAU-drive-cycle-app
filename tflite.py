import tensorflow as tf
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "results", "dnn_surrogate.h5")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "dnn_surrogate.tflite")
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False) 
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)
print(f"✅ Conversion complete! Saved to: {OUTPUT_PATH}")