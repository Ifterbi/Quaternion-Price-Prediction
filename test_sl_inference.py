import requests

try:
    models_res = requests.get("http://localhost:8000/api/models")
    if models_res.status_code == 200:
        models = models_res.json()["models"]
        # Find self learning oscillator
        self_learning = [m["filename"] for m in models if "self_learning" in m["filename"]]
        if not self_learning:
            print("No self_learning model found.")
        else:
            print(f"Testing with oscillator: {self_learning[-1]}")
            data_res = requests.get(f"http://localhost:8000/api/data?oscillator={self_learning[-1]}")
            if data_res.status_code != 200:
                print(f"API Error: {data_res.status_code} - {data_res.text}")
            else:
                data = data_res.json()
                print("API returned data successfully!")
                print("Keys:", data.keys())
                print("Model Predictions type/length:", type(data.get("model_predictions")), len(data.get("model_predictions", [])))
                print("Model Predictions content:", data.get("model_predictions"))
                signals = data.get("signals")
                print("Actual Prices length:", len(data.get("actual_prices", [])))
                if signals:
                    print(f"Signals type: {signals.get('type')}")
                    print(f"Signals values count: {len(signals.get('values', []))}")
                    print(f"Values sample: {signals.get('values', [])[:5]}")
                    print(f"Buy threshold: {signals.get('buy_threshold')}")
                    print(f"Buy threshold: {signals.get('buy_threshold')}")
                else:
                    print("No signals found in response!")
                print("Next Signal:", data.get("next_signal"))
    else:
        print("Could not fetch models")
except Exception as e:
    print(f"Exception: {e}")
