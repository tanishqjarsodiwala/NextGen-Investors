import pickle
import numpy as np

# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Example usage (with dummy features)
sample_features = np.array([[10.5, 1.2, 0.8, 12.3, 8.5]])
prediction = model.predict(sample_features)
print("Sample Prediction:", prediction)
