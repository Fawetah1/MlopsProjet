from flask import Flask, request, jsonify, Response
import os, time
import numpy as np
import pandas as pd
from prometheus_client import Counter, generate_latest
import joblib

app = Flask(__name__)

# -----------------------------
# Load ML model
# -----------------------------
model_path = "traffic_model_fixed2.pkl"
if os.path.exists(model_path):
    local_model = joblib.load(model_path)
else:
    local_model = None

# -----------------------------
# Traffic history storage
# -----------------------------
history_file = "traffic_history.csv"
FEATURE_COUNT = 16  # number of features including time

# Initialize CSV if not exists
if not os.path.exists(history_file):
    df = pd.DataFrame(columns=["timestamp","dow","hour","requests","prediction"])
    df.to_csv(history_file, index=False)

# -----------------------------
# Prometheus metrics
# -----------------------------
REQUEST_COUNT = Counter('http_requests_total', 'Total prediction requests')

# -----------------------------
# Flask routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return "Flask traffic predictor running!"

@app.route("/traffic", methods=["GET"])
def traffic():
    """Return number of requests per minute."""
    df = pd.read_csv(history_file)
    cutoff = time.time() - 60
    recent_requests = df[df.timestamp > cutoff]
    return jsonify({"requests_per_minute": len(recent_requests)})

@app.route("/predict", methods=["POST"])
def predict():
    """Compute features and return prediction."""
    try:
        now_ts = time.time()
        REQUEST_COUNT.inc()
        now_struct = time.localtime(now_ts)

        # Day of week & hour as integers
        dow = now_struct.tm_wday
        hour = now_struct.tm_hour

        # Feature vector
        data = request.get_json()
        if data and "data" in data:
            feature_vector = np.array(data["data"])
            observed_requests = len(data["data"])
        else:
            feature_vector = [ [1]*(FEATURE_COUNT-2) + [dow/6, hour/23] ]
            observed_requests = 1

        if local_model is None:
            raise Exception("No local model available")

        predicted_log = local_model.predict(feature_vector)[0]
        predicted_traffic = np.expm1(predicted_log)
        predicted_traffic = max(0, predicted_traffic)

        # Save traffic & prediction to CSV
        df = pd.read_csv(history_file)
        df = pd.concat([df, pd.DataFrame([{
            "timestamp": now_ts,
            "dow": dow,
            "hour": hour,
            "requests": observed_requests,
            "prediction": predicted_traffic
        }])], ignore_index=True)
        df.to_csv(history_file, index=False)

        return jsonify({"predictions": [predicted_traffic]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype="text/plain")

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)