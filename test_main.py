
from fastapi.testclient import TestClient

from main import app
from main import model
import pandas as pd

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Rain Prediction API is running!"
    }


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200
    assert "Location" in response.json()["input_columns"]


def test_predict():
    payload = {
        "Location": "MelbourneAirport",
        "MinTemp": 12.0,
        "MaxTemp": 16.6,
        "Rainfall": 10.6,
        "Evaporation": 2.2,
        "Sunshine": 5.0,
        "WindGustDir": "W",
        "WindGustSpeed": 65.0,
        "WindDir9am": "W",
        "WindDir3pm": "WSW",
        "WindSpeed9am": 33.0,
        "WindSpeed3pm": 41.0,
        "Humidity9am": 84.0,
        "Humidity3pm": 47.0,
        "Pressure9am": 1003.1,
        "Pressure3pm": 1009.0,
        "Cloud9am": 8.0,
        "Cloud3pm": 5.0,
        "Temp9am": 13.7,
        "Temp3pm": 15.9,
        "RainYesterday": "Yes",
        "Season": "Autumn",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["rain_prediction"] in ["Yes", "No"]
def test_prediction_matches_model(payload=None):
    sample = {
        "Location": "MelbourneAirport",
        "MinTemp": 12.0,
        "MaxTemp": 16.6,
        "Rainfall": 10.6,
        "Evaporation": 2.2,
        "Sunshine": 5.0,
        "WindGustDir": "W",
        "WindGustSpeed": 65.0,
        "WindDir9am": "W",
        "WindDir3pm": "WSW",
        "WindSpeed9am": 33.0,
        "WindSpeed3pm": 41.0,
        "Humidity9am": 84.0,
        "Humidity3pm": 47.0,
        "Pressure9am": 1003.1,
        "Pressure3pm": 1009.0,
        "Cloud9am": 8.0,
        "Cloud3pm": 5.0,
        "Temp9am": 13.7,
        "Temp3pm": 15.9,
        "RainYesterday": "Yes",
        "Season": "Autumn",
    }

    expected = str(model.predict(pd.DataFrame([sample]))[0])
    response = client.post("/predict", json=sample)

    assert response.status_code == 200
    assert response.json()["rain_prediction"] == expected

def test_predict_rejects_missing_field():
    payload = {
        "Location": "MelbourneAirport"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

def test_prediction_probabilities():
    payload = {
        "Location": "MelbourneAirport",
        "MinTemp": 12.0,
        "MaxTemp": 16.6,
        "Rainfall": 10.6,
        "Evaporation": 2.2,
        "Sunshine": 5.0,
        "WindGustDir": "W",
        "WindGustSpeed": 65.0,
        "WindDir9am": "W",
        "WindDir3pm": "WSW",
        "WindSpeed9am": 33.0,
        "WindSpeed3pm": 41.0,
        "Humidity9am": 84.0,
        "Humidity3pm": 47.0,
        "Pressure9am": 1003.1,
        "Pressure3pm": 1009.0,
        "Cloud9am": 8.0,
        "Cloud3pm": 5.0,
        "Temp9am": 13.7,
        "Temp3pm": 15.9,
        "RainYesterday": "Yes",
        "Season": "Autumn",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()
    assert 0 <= result["rain_probability"] <= 1
    assert 0 <= result["no_rain_probability"] <= 1
    assert abs(
        result["rain_probability"]
        + result["no_rain_probability"] - 1
    ) < 0.001
