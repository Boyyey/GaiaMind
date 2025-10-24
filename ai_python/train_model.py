import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load simulation data
data = pd.read_csv('../results/world_log.csv')

# Features: Year, Population, Energy, Pollution, RenewableRatio
# Target: Economy (as a proxy for sustainability)
X = data[['Year', 'Population', 'Energy', 'Pollution', 'RenewableRatio']]
y = data['Economy']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, '../ai_python/sustainability_model.pkl')
print("Model trained and saved as sustainability_model.pkl")

# Evaluate
score = model.score(X_test, y_test)
print(f"Model R^2 score: {score}")
