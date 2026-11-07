# export_model.py - Export the champion model and its predictions for used cars

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

# Step 1: Use the same database that train.py writes to
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Step 2: Load the model that has the alias "champion"
# Format: "models:/<registered model name>@<alias>"
model = mlflow.sklearn.load_model("models:/used-car-price-model@champion")

# Step 3: Save the champion model to a file with joblib
joblib.dump(model, "champion.pkl")
print("Champion model saved to champion.pkl")

# Step 4: Read the used cars data
df = pd.read_csv("used_cars.csv")

# Step 5: Predict a price for every car (same feature columns as train.py)
X = df[["engine_cc", "age_years"]]
df["predicted_price_lakh"] = model.predict(X).round(1)

# Step 6: Show how far off the guesses are, then save everything to a CSV
df["off_by"] = (df["price_lakh"] - df["predicted_price_lakh"]).abs().round(1)
df.to_csv("used_cars_predictions.csv", index=False)
print("Predictions saved to used_cars_predictions.csv")
print(df.head())