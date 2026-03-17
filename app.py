import os
# Disable GPU and oneDNN optimizations to prevent hangs on Windows
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import joblib
from catboost import CatBoostRegressor
from tensorflow.keras.models import load_model
from feature_engineering import FeatureEngineer
from datetime import datetime

app = FastAPI(title="Wind Power Forecasting API")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("templates/index.html")

# Load models
import xgboost as xgb
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("models/xgboost_model.json")
lgb_model = joblib.load("models/lightgbm_model.pkl")

cb_model = CatBoostRegressor()
cb_model.load_model("models/catboost_model.cbm")

lstm_model = load_model("models/lstm_model.keras")

scaler_X = joblib.load("models/scaler_X.pkl")
scaler_y = joblib.load("models/scaler_y.pkl")

# Initialize Feature Engineer
fe = FeatureEngineer()


def clip_prediction(pred):
    return max(0.0, float(pred))


from typing import List

from pydantic import BaseModel

class PredictionInput(BaseModel):
    features: List[float]

class RawPredictionInput(BaseModel):
    wind_speed_ms: float
    theoretical_power_kwh: float
    wind_direction: float
    timestamp: str = None # Format: YYYY-MM-DD HH:MM:SS

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

@app.post("/predict_smart")
def predict_smart(data: RawPredictionInput):
    """
    Automated feature engineering prediction
    """
    ts = data.timestamp if data.timestamp else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get 13 features from feature engineer
    features = fe.get_features_for_prediction(
        data.wind_speed_ms,
        data.theoretical_power_kwh,
        data.wind_direction,
        ts
    )
    
    # Perform prediction using existing logic
    X = np.array(features).reshape(1, -1)
    
    raw_xgb = xgb_model.predict(X)[0]
    pred_xgb = clip_prediction(raw_xgb)
    
    raw_lgb = lgb_model.predict(X)[0]
    pred_lgb = clip_prediction(raw_lgb)
    
    raw_cb = cb_model.predict(X)[0]
    pred_cb  = clip_prediction(raw_cb)
    
    X_scaled = scaler_X.transform(X)
    X_lstm = X_scaled.reshape((1, 1, X_scaled.shape[1]))
    lstm_scaled = lstm_model.predict(X_lstm)[0][0]
    pred_lstm = clip_prediction(
        scaler_y.inverse_transform([[lstm_scaled]])[0][0]
    )
    
    # Debug logging
    print(f"DEBUG - Timestamp: {ts}")
    print(f"DEBUG - Features: {features}")
    print(f"DEBUG - Raw Predictions: XGB={raw_xgb}, LGB={raw_lgb}, CB={raw_cb}, LSTM={pred_lstm}")
    
    # Update feature engineer with predicted power (ensemble mean as proxy if actual not available)
    ensemble_avg = (pred_xgb + pred_lgb + pred_cb + pred_lstm) / 4
    fe.add_reading(data.wind_speed_ms, data.theoretical_power_kwh, data.wind_direction, ts, actual_power=ensemble_avg)
    
    return {
        "features_generated": features,
        "predictions": {
            "XGBoost": pred_xgb,
            "LightGBM": pred_lgb,
            "CatBoost": pred_cb,
            "LSTM": pred_lstm,
            "Ensemble": ensemble_avg
        }
    }
