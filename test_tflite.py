import numpy as np
from tflite_runtime.interpreter import Interpreter  # lightweight runtime
# If you only have TensorFlow installed, replace with:
# from tensorflow.lite.python.interpreter import Interpreter

# Path to your converted TFLite model
MODEL_PATH = "results/dnn_surrogate.tflite"

# Load the model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print them (helps debug shape mismatches)
print("Input details:", input_details)
print("Output details:", output_details)

# Create a fake sample input matching the model’s input shape
input_shape = input_details[0]['shape']
print("Expected input shape:", input_shape)

# Replace with realistic scaled values if you know your input scaling
sample_input = np.array([[3300, 0, 0.010, 25, 25, 80, 5, 55, 1, 100]], dtype=np.float32)

# If your model was trained with normalized/scaled inputs,
# you should apply the same scaler.transform() here.

# Run inference
interpreter.set_tensor(input_details[0]['index'], sample_input)
interpreter.invoke()

# Extract output
output_data = interpreter.get_tensor(output_details[0]['index'])
print("✅ Model prediction:", output_data)