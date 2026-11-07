# train.py - Beginner ML lab: predict used car prices (with MLflow tracking)

import sys
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Step 0: Tell MLflow where to store results and name the experiment
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("used-car-price-project")

# Step 1: Read the CSV file with pandas
df = pd.read_csv("used_cars.csv")

# Step 2: Choose features (inputs) X and the label (answer) y
X = df[["engine_cc", "age_years"]]
y = df["price_lakh"]

# Step 3: Split into training data (80%) and testing data (20%)
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)
print("Cars used for training:", len(X_train))
print("Cars used for testing:", len(X_test))

# Step 4: Choose the model type from the command line
# Run: python train.py linear   (or: tree, forest). Default is linear.
model_type = sys.argv[1] if len(sys.argv) > 1 else "linear"

if model_type == "linear":
    model = LinearRegression()
elif model_type == "tree":
    model = DecisionTreeRegressor(random_state=42)
elif model_type == "forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    print("Unknown model type. Use: linear, tree or forest")
    sys.exit(1)

# Step 5: Start an MLflow run, then train and score inside it
with mlflow.start_run():

    # Step 5a: Train the model
    model.fit(X_train, y_train)

    # Step 5b: Predict the test cars and compare with the real prices
    predictions = model.predict(X_test)

    print("\nengine_cc | age_years | real_price | guess | off_by")
    for i in range(len(X_test)):
        engine = X_test.iloc[i]["engine_cc"]
        age = X_test.iloc[i]["age_years"]
        real = y_test.iloc[i]
        guess = predictions[i]
        off_by = abs(real - guess)
        print(f"{engine:>9} | {age:>9} | {real:>10} | {guess:>5.1f} | {off_by:>6.1f}")

    # Step 6: Calculate the Mean Absolute Error (average mistake size)
    mae = mean_absolute_error(y_test, predictions)
    print(f"\nModel type: {model_type} | MAE: {mae:.2f} lakh")

    # Step 7: Log the settings (parameters) and the result (metric) to MLflow
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("rows_in_dataset", len(df))
    mlflow.log_metric("mae", mae)

    # Step 8: Log the trained model to MLflow and register it
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="used-car-price-model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

# Step 9: Still save the model to a file using joblib
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")