# predict.py - Use the saved model to predict a new used car's price

import pandas as pd
import joblib

# Step 1: Load the trained model saved by train.py
model = joblib.load("model.pkl")

# Step 2: Make a DataFrame for the new car (column names must match training)
new_car = pd.DataFrame([[600, 1]], columns=["engine_cc", "age_years"])

# Step 3: Predict the price
predicted_price = model.predict(new_car)[0]

# Step 4: Print the result
print(f"600 cc, 1 years old -> predicted price: {predicted_price:.1f} lakh")