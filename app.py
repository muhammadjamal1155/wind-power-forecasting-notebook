import os
import json
import requests
from datetime import datetime
from typing import List
from dotenv import load_dotenv
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tensorflow.keras.models import load_model

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from supabase import create_client, Client

from feature_engineering import FeatureEngineer

load_dotenv() # Load env vars from .env

# Supabase Setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Connected to Supabase")
    except Exception as e:
        print(f"Supabase connection error: {e}")

# Automation Config
AUTO_LAT = float(os.getenv("AUTO_LAT", "52.52"))
AUTO_LON = float(os.getenv("AUTO_LON", "13.41"))
AUTO_THEORETICAL_CAPACITY = float(os.getenv("AUTO_THEORETICAL_CAPACITY", "500"))

# Disable GPU and oneDNN optimizations to prevent hangs on Windows (rebuild v2)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start Scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(automated_prediction_job, 'interval', minutes=10)
    scheduler.start()
    print("Scheduler started: Running automated predictions every 10 minutes.")
    
    # Run once at startup immediately
    # await automated_prediction_job()
    
    yield
    # Shutdown
    scheduler.shutdown()
    print("Scheduler shutdown")

app = FastAPI(title="Wind Power Forecasting API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class WeatherInput(BaseModel):
    lat: float
    lon: float


@app.get("/")
def read_root():
    return FileResponse("templates/index.html")


# Open-Meteo API Proxy (No Key Required)
@app.post("/fetch_weather")
def fetch_weather(data: WeatherInput):
    # Open-Meteo requires no API key for non-commercial use
    url = f"https://api.open-meteo.com/v1/forecast?latitude={data.lat}&longitude={data.lon}&current=wind_speed_10m,wind_direction_10m&wind_speed_unit=ms"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        weather_data = response.json()
        current = weather_data.get("current", {})
        
        return {
            "wind_speed": current.get("wind_speed_10m", 0),
            "wind_deg": current.get("wind_direction_10m", 0),
            "temp": current.get("temperature_2m", 0), # Optional
            "location": "Local Weather (Open-Meteo)"
        }
    except Exception as e:
        return {"error": f"Failed to fetch weather: {str(e)}"}



# Load models
with open("models/xgboost_model.json", "r", encoding="utf-8") as f:
    _xgb_model_metadata = json.load(f)

XGB_MODEL_VERSION = ".".join(str(part) for part in _xgb_model_metadata.get("version", []))
XGB_RUNTIME_VERSION = xgb.__version__

if XGB_MODEL_VERSION and ".".join(XGB_RUNTIME_VERSION.split(".")[:2]) != ".".join(XGB_MODEL_VERSION.split(".")[:2]):
    raise RuntimeError(
        f"XGBoost runtime version {XGB_RUNTIME_VERSION} is incompatible with the "
        f"serialized model version {XGB_MODEL_VERSION}. Install a matching XGBoost "
        f"version before serving predictions."
    )

xgb_model = xgb.XGBRegressor()
xgb_model.load_model("models/xgboost_model.json")
XGB_FEATURE_NAMES = xgb_model.get_booster().feature_names or [
    "wind_speed_ms",
    "Theoretical_Power_Curve (KWh)",
    "Wind Direction (°)",
    "hour_of_day",
    "day_of_week",
    "power_lag_1",
    "wind_speed_ms_lag_1",
    "power_lag_2",
    "wind_speed_ms_lag_2",
    "power_lag_6",
    "wind_speed_ms_lag_6",
    "power_roll_mean_6",
    "wind_speed_ms_roll_mean_6",
]
lgb_model = joblib.load("models/lightgbm_model.pkl")

cb_model = CatBoostRegressor()
cb_model.load_model("models/catboost_model.cbm")

lstm_model = load_model("models/lstm_model.keras")

scaler_X = joblib.load("models/scaler_X.pkl")
scaler_y = joblib.load("models/scaler_y.pkl")

STATEFUL_FEATURE_HISTORY = os.getenv("STATEFUL_FEATURE_HISTORY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Initialize shared Feature Engineer only when stateful mode is enabled
fe = FeatureEngineer() if STATEFUL_FEATURE_HISTORY else None


@app.get("/debug_env")
def debug_env():
    import sys

    import numpy
    import pandas
    import sklearn
    import xgboost

    test_X = np.zeros((1, 13))
    try:
        raw_test = xgb_model.predict(test_X)[0]
    except Exception as e:
        raw_test = str(e)

    return {
        "python": sys.version,
        "xgboost": xgboost.__version__,
        "xgboost_model_version": XGB_MODEL_VERSION,
        "sklearn": sklearn.__version__,
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "stateful_feature_history": STATEFUL_FEATURE_HISTORY,
        "xgb_model_type": str(type(xgb_model)),
        "xgb_feature_names": XGB_FEATURE_NAMES,
        "raw_test_prediction": float(raw_test)
        if isinstance(raw_test, (float, np.float32, np.float64))
        else raw_test,
    }


def clip_prediction(pred):
    return max(0.0, float(pred))


def predict_xgb(features):
    X = np.array(features).reshape(1, -1)
    dmatrix = xgb.DMatrix(X, feature_names=XGB_FEATURE_NAMES)
    raw_pred = float(xgb_model.get_booster().predict(dmatrix)[0])
    return raw_pred, clip_prediction(raw_pred)


def cold_start_xgb_fallback(raw_xgb, pred_lgb, pred_cb, pred_lstm, is_cold_start):
    peer_preds = np.array([pred_lgb, pred_cb, pred_lstm], dtype=float)
    peer_mean = float(np.mean(peer_preds))
    peer_median = float(np.median(peer_preds))

    if not is_cold_start:
        return clip_prediction(raw_xgb), False, "history_available"

    # XGBoost is heavily lag-dependent. In stateless cold-start mode the
    # synthetic lag features can produce values that are directionally wrong
    # even when they are still positive, so guard against strong low outliers.
    severe_low_outlier = raw_xgb < (peer_median * 0.65)
    if raw_xgb > 0 and not severe_low_outlier:
        return clip_prediction(raw_xgb), False, "xgb_ok"

    fallback_reason = "negative_or_zero_xgb" if raw_xgb <= 0 else "cold_start_low_outlier"
    return clip_prediction(peer_mean), True, fallback_reason


class PredictionInput(BaseModel):
    features: List[float]


class RawPredictionInput(BaseModel):
    wind_speed_ms: float
    theoretical_power_kwh: float
    wind_direction: float
    timestamp: str = None  # Format: YYYY-MM-DD HH:MM:SS


@app.post("/predict")
def predict(data: PredictionInput):
    """
    features: list of input values in the same order as training
    """
    features = data.features
    X = np.array(features).reshape(1, -1)

    _, pred_xgb = predict_xgb(features)
    pred_lgb = clip_prediction(lgb_model.predict(X)[0])
    pred_cb = clip_prediction(cb_model.predict(X)[0])

    X_scaled = scaler_X.transform(X)
    X_lstm = X_scaled.reshape((1, 1, X_scaled.shape[1]))
    lstm_scaled = lstm_model.predict(X_lstm)[0][0]
    pred_lstm = clip_prediction(scaler_y.inverse_transform([[lstm_scaled]])[0][0])

    return {
        "XGBoost": pred_xgb,
        "LightGBM": pred_lgb,
        "CatBoost": pred_cb,
        "LSTM": pred_lstm,
    }


@app.post("/predict_smart")
def predict_smart(data: RawPredictionInput):
    """
    Automated feature engineering prediction
    """
    ts = data.timestamp if data.timestamp else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    feature_engineer = fe if STATEFUL_FEATURE_HISTORY else FeatureEngineer()
    is_cold_start = feature_engineer.is_cold_start()

    features = feature_engineer.get_features_for_prediction(
        data.wind_speed_ms,
        data.theoretical_power_kwh,
        data.wind_direction,
        ts,
    )

    X = np.array(features).reshape(1, -1)

    raw_lgb = lgb_model.predict(X)[0]
    pred_lgb = clip_prediction(raw_lgb)

    raw_cb = cb_model.predict(X)[0]
    pred_cb = clip_prediction(raw_cb)

    X_scaled = scaler_X.transform(X)
    X_lstm = X_scaled.reshape((1, 1, X_scaled.shape[1]))
    lstm_scaled = lstm_model.predict(X_lstm)[0][0]
    pred_lstm = clip_prediction(scaler_y.inverse_transform([[lstm_scaled]])[0][0])

    raw_xgb, pred_xgb = predict_xgb(features)
    pred_xgb, xgb_used_fallback, xgb_fallback_reason = cold_start_xgb_fallback(
        raw_xgb,
        pred_lgb,
        pred_cb,
        pred_lstm,
        is_cold_start,
    )

    print(f"DEBUG - Timestamp: {ts}")
    print(f"DEBUG - Features: {features}")
    print(
        f"DEBUG - Predictions: "
        f"XGB_raw={raw_xgb}, XGB_final={pred_xgb}, XGB_fallback={xgb_used_fallback}, "
        f"XGB_fallback_reason={xgb_fallback_reason}, "
        f"LGB={raw_lgb}, CB={raw_cb}, LSTM={pred_lstm}"
    )

    ensemble_avg = (pred_xgb + pred_lgb + pred_cb + pred_lstm) / 4
    if STATEFUL_FEATURE_HISTORY:
        feature_engineer.add_reading(
            data.wind_speed_ms,
            data.theoretical_power_kwh,
            data.wind_direction,
            ts,
            actual_power=ensemble_avg,
        )

    # Save to history (Supabase)
    if supabase:
        try:
            db_data = {
                "wind_speed": float(data.wind_speed_ms),
                "wind_direction": float(data.wind_direction),
                "theoretical_power": float(data.theoretical_power_kwh),
                "prediction_ensemble": float(ensemble_avg),
                "prediction_xgb": float(pred_xgb),
                "prediction_lgb": float(pred_lgb),
                "prediction_cb": float(pred_cb),
                "prediction_lstm": float(pred_lstm),
            }
            supabase.table("prediction_history").insert(db_data).execute()
            print("DEBUG - Saved manual prediction to Supabase")
        except Exception as e:
            print(f"ERROR - Failed to save manual prediction: {e}")

    return {
        "features_generated": features,
        "debug": {
            "stateful_feature_history": STATEFUL_FEATURE_HISTORY,
            "cold_start": is_cold_start,
            "xgboost_raw": raw_xgb,
            "xgboost_used_fallback": xgb_used_fallback,
            "xgboost_fallback_reason": xgb_fallback_reason,
        },
        "predictions": {
            "XGBoost": pred_xgb,
            "LightGBM": pred_lgb,
            "CatBoost": pred_cb,
            "LSTM": pred_lstm,
            "Ensemble": ensemble_avg,
        },
    }


@app.get("/history")
def get_history(limit: int = 10, start: str = None, end: str = None):
    if not supabase:
        return {"error": "Supabase not configured"}
    
    try:
        query = supabase.table("prediction_history") \
            .select("*") \
            .order("created_at", desc=True)
        
        if start:
            query = query.gte("created_at", start)
        if end:
            query = query.lte("created_at", end)
        
        query = query.limit(limit)
        response = query.execute()
        return response.data
    except Exception as e:
        return {"error": f"Failed to fetch history: {str(e)}"}


async def automated_prediction_job():
    print(f"DEBUG - [{datetime.now()}] Starting automated prediction...")
    # 1. Fetch Weather (Open-Meteo)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={AUTO_LAT}&longitude={AUTO_LON}&current=wind_speed_10m,wind_direction_10m&wind_speed_unit=ms"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        w_data = resp.json().get("current", {})
        wind_speed = w_data.get("wind_speed_10m", 0)
        wind_deg = w_data.get("wind_direction_10m", 0)
    except Exception as e:
        print(f"ERROR - Weather fetch failed: {e}")
        return

    # 2. Make Prediction using predict_smart logic
    # Create input object
    input_data = RawPredictionInput(
        wind_speed_ms=float(wind_speed),
        theoretical_power_kwh=AUTO_THEORETICAL_CAPACITY,
        wind_direction=float(wind_deg)
    )
    
    try:
        # Call the existing logic
        result = predict_smart(input_data)
        preds = result["predictions"]
        
        # 3. Store in Supabase
        if supabase:
            db_data = {
                "wind_speed": float(wind_speed),
                "wind_direction": float(wind_deg),
                "theoretical_power": AUTO_THEORETICAL_CAPACITY,
                "prediction_ensemble": float(preds["Ensemble"]),
                "prediction_xgb": float(preds["XGBoost"]),
                "prediction_lgb": float(preds["LightGBM"]),
                "prediction_cb": float(preds["CatBoost"]),
                "prediction_lstm": float(preds["LSTM"]),
            }
            supabase.table("prediction_history").insert(db_data).execute()
            print(f"DEBUG - Saved automated prediction: {preds['Ensemble']} KW")
        else:
            print("DEBUG - Skip Supabase: Client not initialized.")
    except Exception as e:
        print(f"ERROR - Automated prediction failed: {e}")
