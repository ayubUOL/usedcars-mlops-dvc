# Data tests: is houses.csv safe to train on?
import pandas as pd

data = pd.read_csv("used_cars.csv")

def test_columns_are_correct():
    assert list(data.columns) == ["engine_cc", "age_years", "price_lakh"]

def test_no_empty_cells():
    assert data.notna().all().all()

def test_all_values_are_positive():
    assert (data > 0).all().all()

def test_enough_cars():
    assert len(data) >= 20
