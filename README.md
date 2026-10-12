# Rain Prediction API 🌧️

A Machine Learning API that predicts rain using weather data.

## Live API
https://rain-prediction-api-5zen.onrender.com/

## API Documentation
https://rain-prediction-api-5zen.onrender.com/docs

## Features
- FastAPI REST API
- Machine Learning predictions
- Docker containerization
- API testing
- Cloud deployment on Render

## Endpoints
- `GET /` — API home
- `GET /model-info` — Model information
- `POST /predict` — Predict rain probability

## Technologies
Python, FastAPI, scikit-learn, Docker, Git, GitHub, Render

## Example Response
```json
{
  "rain_prediction": "No",
  "rain_probability": 0.0403,
  "no_rain_probability": 0.9597
}
```

Note: This is a sample prediction, not a live weather forecast.

