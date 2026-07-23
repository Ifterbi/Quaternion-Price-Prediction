import requests
url = "http://127.0.0.1:8000/api/data?models=lstm_model.keras"
response = requests.get(url).json()

print(f"actual_prices length: {len(response['actual_prices'])}")
print(f"predicted_prices length: {len(response['model_predictions'][0]['predicted_prices'])}")

print("First 5 actual:", response['actual_prices'][:5])
print("First 5 predicted:", response['model_predictions'][0]['predicted_prices'][:5])
