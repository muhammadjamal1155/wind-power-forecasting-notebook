import requests
import json

url = "http://127.0.0.1:8002/predict"
# 13 dummy features
features = [10.0, 1000.0, 180.0, 12.0, 3.0, 900.0, 9.0, 800.0, 8.0, 600.0, 7.0, 850.0, 9.5]
payload = {"features": features}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response:", response.json())
        data = response.json()
        if "XGBoost" in data:
            print("Verification Successful: Response contains XGBoost prediction.")
        else:
            print("Verification Failed: Response content unexpected.")
    else:
        print("Verification Failed: Status code is not 200.")
        print("Response Text:", response.text)
except Exception as e:
    print(f"Request failed: {e}")
