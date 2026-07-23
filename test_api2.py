import requests
url = "http://127.0.0.1:8000/api/data?models=extended_mtl_2026-07-23_014523.keras"
response = requests.get(url).json()

print("Last 5 actual:", response['actual_prices'][-5:])
print("Last 5 predicted:", response['model_predictions'][0]['predicted_prices'][-5:])
print("Dates first 5:", response['dates'][:5])
print("Dates last 5:", response['dates'][-5:])
