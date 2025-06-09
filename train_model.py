# train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load data
df = pd.read_csv("data/train.csv")

# Keep only selected features
df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]
df.dropna(inplace=True)

X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Save model and scaler
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/house_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Model trained and saved!")
