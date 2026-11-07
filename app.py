# app.py - a small web API that answers used-car price questions using champion.pkl
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Used Car Price API")

# Load the model ONCE when the API starts (not on every request)
model = joblib.load("champion.pkl")

# The shape of a request. FastAPI rejects anything that doesn't match.
class Car(BaseModel):
    engine_cc: int = Field(gt=0, description="Engine size in cc")
    age_years: int = Field(ge=0, description="Age of the car in years")

# Door 1: "are you alive?"
@app.get("/health")
def health():
    return {"status": "ok"}

# Door 2: the real job
@app.post("/predict")
def predict(car: Car):
    # Column names must match the ones used in train.py
    data = pd.DataFrame([{"engine_cc": car.engine_cc, "age_years": car.age_years}])
    price = model.predict(data)[0]
    return {"price_lakh": round(float(price), 1)}