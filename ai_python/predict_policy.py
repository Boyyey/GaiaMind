import joblib
import numpy as np

# Load trained model
model = joblib.load('sustainability_model.pkl')

def predict_sustainability(year, population, energy, pollution, renewable_ratio):
    features = np.array([[year, population, energy, pollution, renewable_ratio]])
    prediction = model.predict(features)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    # Example prediction for year 50
    pred = predict_sustainability(50, 9000000000, 1200, 70, 0.5)
    print(f"Predicted Economy (Sustainability Index): {pred}")

    # Suggest policy
    if pred < 80:
        print("Policy Suggestion: Increase renewable investment to reduce pollution.")
    else:
        print("Policy Suggestion: Maintain current balance.")
