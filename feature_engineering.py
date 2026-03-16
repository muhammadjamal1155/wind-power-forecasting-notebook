import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime

class FeatureEngineer:
    def __init__(self, history_size=7):
        """
        history_size should be at least max(lag) + 1 or window_size.
        Here max lag is 6 and window size is 6. 7 should be enough for lags, 
        and 6 is enough for rolling mean.
        We'll keep 10 to be safe.
        """
        self.history = deque(maxlen=10)
        
    def add_reading(self, wind_speed, theoretical_power, wind_direction, timestamp, actual_power=None):
        """
        Adds a new reading to history. 
        If actual_power is not provided (e.g. during prediction), 
        we might need to handle it or use the last prediction (not ideal).
        However, for lag features of power, we need previous power values.
        """
        if isinstance(timestamp, str):
            dt = pd.to_datetime(timestamp)
        else:
            dt = timestamp
            
        reading = {
            'timestamp': dt,
            'wind_speed_ms': float(wind_speed),
            'theoretical_power_kwh': float(theoretical_power),
            'wind_direction': float(wind_direction),
            'wind_power_output': float(actual_power) if actual_power is not None else 0.0,
            'hour_of_day': dt.hour,
            'day_of_week': dt.dayofweek
        }
        self.history.append(reading)
        return reading

    def get_features_for_prediction(self, current_wind_speed, current_theoretical_power, current_wind_direction, current_timestamp):
        """
        Returns the 13 features required by the models.
        Order: wind_speed_ms, theoretical_power_kwh, wind_direction,
               hour_of_day, day_of_week, 
               power_lag_1, wind_speed_ms_lag_1,
               power_lag_2, wind_speed_ms_lag_2,
               power_lag_6, wind_speed_ms_lag_6,
               power_roll_mean_6, wind_speed_ms_roll_mean_6
        """
        dt = pd.to_datetime(current_timestamp)
        hour = dt.hour
        day_of_week = dt.dayofweek
        
        # Convert history to DataFrame for easy calculation
        df_hist = pd.DataFrame(list(self.history))
        
        if len(df_hist) < 1:
            # Not enough history, use current values as fallback to avoid zero-drop anomaly
            # For power lags, we use 80% of theoretical power to approximate actual power
            return [
                float(current_wind_speed), float(current_theoretical_power), float(current_wind_direction),
                float(hour), float(day_of_week),
                float(current_theoretical_power) * 0.8, float(current_wind_speed),
                float(current_theoretical_power) * 0.8, float(current_wind_speed),
                float(current_theoretical_power) * 0.8, float(current_wind_speed),
                float(current_theoretical_power) * 0.8, float(current_wind_speed)
            ]

        # Function to get lag or default
        def get_lag(col, lag_val):
            if len(df_hist) >= lag_val:
                return df_hist[col].iloc[-lag_val]
            return df_hist[col].iloc[-1] # Fallback to last available

        # Function for rolling mean
        def get_roll_mean(col, window):
            # Include current value in rolling mean if needed? 
            # Snippet used: df[col].rolling(window=window).mean()
            # Usually rolling mean includes current.
            vals = df_hist[col].tail(window-1).tolist()
            if col == 'wind_speed_ms':
                vals.append(float(current_wind_speed))
            # For power rolling mean, we don't have current power yet (that's what we predict)
            # So rolling mean of power is likely over previous window_size readings.
            
            if len(vals) == 0: return 0.0
            return sum(vals) / len(vals)

        p_lag1 = get_lag('wind_power_output', 1)
        w_lag1 = get_lag('wind_speed_ms', 1)
        p_lag2 = get_lag('wind_power_output', 2)
        w_lag2 = get_lag('wind_speed_ms', 2)
        p_lag6 = get_lag('wind_power_output', 6)
        w_lag6 = get_lag('wind_speed_ms', 6)
        
        p_roll6 = df_hist['wind_power_output'].tail(6).mean() if len(df_hist) >= 1 else float(current_theoretical_power) * 0.8
        # For wind speed rolling, include current
        w_vals = df_hist['wind_speed_ms'].tail(5).tolist()
        w_vals.append(float(current_wind_speed))
        w_roll6 = sum(w_vals) / len(w_vals)

        return [
            float(current_wind_speed), float(current_theoretical_power), float(current_wind_direction),
            float(hour), float(day_of_week),
            float(p_lag1), float(w_lag1),
            float(p_lag2), float(w_lag2),
            float(p_lag6), float(w_lag6),
            float(p_roll6), float(w_roll6)
        ]

    def update_last_power(self, predicted_power):
        """
        If actual power isn't available, we can update the last history entry 
        with the prediction to serve as lag for the NEXT step.
        """
        if self.history:
            self.history[-1]['wind_power_output'] = float(predicted_power)
