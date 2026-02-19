
import requests
import json
import sys

# Test port 8001 as seen in user error
url = "http://127.0.0.1:8001/predict"
# Dummy features matching length 13
features = [10.0, 1000.0, 180.0, 12.0, 3.0, 900.0, 9.0, 800.0, 8.0, 600.0, 7.0, 850.0, 9.5]
payload = {"features": features}

try:
    print(f"Sending request to {url} with payload keys: {list(payload.keys())}")
    response = requests.post(url, json=payload, timeout=5)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response:", response.json())
        print("SUCCESS")
    else:
        print("FAILURE")
        print("Response Text:", response.text)
except Exception as e:
    print(f"Request failed: {e}")
