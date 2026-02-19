import sys
import time

print("Step 1: Imports starting...")
sys.stdout.flush()

try:
    import joblib
    print("Joblib imported")
    sys.stdout.flush()

    import numpy as np
    print("Numpy imported")
    sys.stdout.flush()

    from catboost import CatBoostRegressor
    print("CatBoost imported")
    sys.stdout.flush()

    import tensorflow as tf
    print(f"TensorFlow imported: {tf.__version__}")
    sys.stdout.flush()
    
    from tensorflow.keras.models import load_model
    print("Keras load_model imported")
    sys.stdout.flush()

except Exception as e:
    print(f"Error during imports: {e}")
    sys.exit(1)

print("Step 2: Load models starting...")
sys.stdout.flush()

try:
    print("Loading XGBoost...")
    sys.stdout.flush()
    xgb_model = joblib.load("models/xgboost_model.pkl")
    print("XGBoost loaded")
    sys.stdout.flush()

    print("Loading LightGBM...")
    sys.stdout.flush()
    lgb_model = joblib.load("models/lightgbm_model.pkl")
    print("LightGBM loaded")
    sys.stdout.flush()

    print("Loading CatBoost...")
    sys.stdout.flush()
    cb_model = CatBoostRegressor()
    cb_model.load_model("models/catboost_model.cbm")
    print("CatBoost loaded")
    sys.stdout.flush()

    print("Loading LSTM...")
    sys.stdout.flush()
    lstm_model = load_model("models/lstm_model.keras")
    print("LSTM loaded")
    sys.stdout.flush()

    print("Loading Scalers...")
    sys.stdout.flush()
    scaler_X = joblib.load("models/scaler_X.pkl")
    scaler_y = joblib.load("models/scaler_y.pkl")
    print("Scalers loaded")
    sys.stdout.flush()

    print("All models loaded successfully!")

except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)
