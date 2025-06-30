import requests

BASE_URL = "http://127.0.0.1:5000"

# 1. Test /health endpoint
print("Testing /health endpoint...")
health = requests.get(f"{BASE_URL}/health")
print("/health response:", health.status_code, health.json())

# 2. Test /models/info endpoint
print("\nTesting /models/info endpoint...")
models_info = requests.get(f"{BASE_URL}/models/info")
print("/models/info response:", models_info.status_code, models_info.json())

# Prompt user for app details
print("Enter app details for prediction:")
name = input("App name: ")
category = input("Category (e.g., TOOLS): ")
type_ = input("Type (Free/Paid): ")
size_mb = float(input("App size (MB): "))
content_rating = input("Content rating (Everyone, Everyone 10+, Teen, Mature 17+): ")
android_version = float(input("Minimum Android version (e.g., 5.0): "))

predict_data = {
    "name": name,
    "category": category,
    "type": type_,
    "size_mb": size_mb,
    "content_rating": content_rating,
    "android_version": android_version
}

print("\nSending POST request to /predict...")
predict = requests.post(f"{BASE_URL}/predict", json=predict_data)
print("/predict response:", predict.status_code, predict.json()) 