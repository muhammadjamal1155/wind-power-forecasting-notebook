import os
# Disable GPU and oneDNN optimizations to prevent hangs on Windows
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
import numpy as np
import joblib
from catboost import CatBoostRegressor
from tensorflow.keras.models import load_model

app = FastAPI(title="Wind Power Forecasting API")

@app.get("/")
def read_root():
    return FileResponse("templates/index.html")

# Load models
xgb_model = joblib.load("models/xgboost_model.pkl")
lgb_model = joblib.load("models/lightgbm_model.pkl")

cb_model = CatBoostRegressor()
cb_model.load_model("models/catboost_model.cbm")

lstm_model = load_model("models/lstm_model.keras")

scaler_X = joblib.load("models/scaler_X.pkl")
scaler_y = joblib.load("models/scaler_y.pkl")


def clip_prediction(pred):
    return max(0.0, float(pred))


from typing import List

from pydantic import BaseModel

class PredictionInput(BaseModel):
    features: List[float]

@app.post("/predict")
def predict(data: PredictionInput):
    """
    features: list of input values in the same order as training
    """
    features = data.features

    X = np.array(features).reshape(1, -1)

    # Tree-based models
    pred_xgb = clip_prediction(xgb_model.predict(X)[0])
    pred_lgb = clip_prediction(lgb_model.predict(X)[0])
    pred_cb  = clip_prediction(cb_model.predict(X)[0])

    # LSTM model
    X_scaled = scaler_X.transform(X)
    X_lstm = X_scaled.reshape((1, 1, X_scaled.shape[1]))
    lstm_scaled = lstm_model.predict(X_lstm)[0][0]
    pred_lstm = clip_prediction(
        scaler_y.inverse_transform([[lstm_scaled]])[0][0]
    )

    return {
        "XGBoost": pred_xgb,
        "LightGBM": pred_lgb,
        "CatBoost": pred_cb,
        "LSTM": pred_lstm
    }
