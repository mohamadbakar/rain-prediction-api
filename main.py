
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path(__file__).parent / "rain_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Rain Prediction API")


class WeatherInput(BaseModel):
    Location: str
    MinTemp: float
    MaxTemp: float
    Rainfall: float
    Evaporation: float
    Sunshine: float
    WindGustDir: str
    WindGustSpeed: float
    WindDir9am: str
    WindDir3pm: str
    WindSpeed9am: float
    WindSpeed3pm: float
    Humidity9am: float
    Humidity3pm: float
    Pressure9am: float
    Pressure3pm: float
    Cloud9am: float
    Cloud3pm: float
    Temp9am: float
    Temp3pm: float
    RainYesterday: str
    Season: str


@app.get("/")
def home():
    return {"message": "Rain Prediction API is running!"}


@app.get("/model-info")
def model_info():
    return {"input_columns": model.feature_names_in_.tolist()}



@app.post("/predict")
def predict_weather(data: WeatherInput):
    input_df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    classes = model.classes_.tolist()
    probability_by_class = dict(zip(classes, probabilities))

    return {
        "rain_prediction": str(prediction),
        "rain_probability": round(
            float(probability_by_class.get("Yes", 0.0)), 4
        ),
        "no_rain_probability": round(
            float(probability_by_class.get("No", 0.0)), 4
        ),
    }

