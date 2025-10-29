import os, time
import requests
import numpy as np
import pandas as pd
from kubernetes import client, config
from prometheus_client import Gauge, start_http_server

# -----------------------------
# Prometheus metrics
# -----------------------------
start_http_server(8000)
predicted_g = Gauge('predicted_traffic', 'Predicted traffic')
replicas_g = Gauge('current_replicas', 'Current pods')

# -----------------------------
# Kubernetes setup
# -----------------------------
try:
    config.load_incluster_config()
except:
    config.load_kube_config()  # local testing

apps_v1 = client.AppsV1Api()
DEPLOYMENT = "traffic-server"
NAMESPACE = "default"
MIN_PODS = 1
MAX_PODS = 10
CHECK_INTERVAL = 10
COOLDOWN = 20
last_scale_time = 0
traffic_per_pod = 3

# -----------------------------
# Historical traffic file
# -----------------------------
history_file = "traffic_history.csv"

# -----------------------------
# Helper functions
# -----------------------------
def get_current_replicas():
    scale = apps_v1.read_namespaced_deployment_scale(DEPLOYMENT, NAMESPACE)
    return scale.status.replicas or MIN_PODS

def get_traffic_history():
    try:
        resp = requests.get("http://traffic-server:5000/traffic", timeout=5)
        resp.raise_for_status()
        return resp.json().get('requests_per_minute', 1)
    except:
        return 1

def get_historical_average(dow, hour):
    """Return average traffic for the same day/hour in history."""
    if os.path.exists(history_file):
        df = pd.read_csv(history_file)
        df_filtered = df[(df.dow==dow) & (df.hour==hour)]
        if len(df_filtered) > 0:
            return df_filtered.requests.mean()
    return None

def build_feature_vector(current_requests):
    now = time.localtime()
    dow_norm = now.tm_wday / 6
    hour_norm = now.tm_hour / 23
    return [np.log1p([max(1, current_requests)]*14 + [dow_norm, hour_norm]).tolist()]

def get_prediction(feature_vector):
    try:
        resp = requests.post("http://traffic-server:5000/predict",
                             json={"data": feature_vector}, timeout=5)
        resp.raise_for_status()
        return resp.json()["predictions"][0]
    except:
        return None

def scale_deployment(desired_replicas):
    global last_scale_time
    body = {"spec": {"replicas": desired_replicas}}
    apps_v1.patch_namespaced_deployment_scale(
        name=DEPLOYMENT,
        namespace=NAMESPACE,
        body=body
    )
    last_scale_time = time.time()
    print(f"✅ Scaled deployment to {desired_replicas} replicas")

# -----------------------------
# Main loop
# -----------------------------
alpha = 0.7  # weight for model prediction
while True:
    current_replicas = get_current_replicas()
    traffic_count = get_traffic_history()
    feature_vector = build_feature_vector(traffic_count)
    predicted_traffic = get_prediction(feature_vector)

    now = time.localtime()
    dow = now.tm_wday
    hour = now.tm_hour
    hist_avg = get_historical_average(dow, hour)

    if predicted_traffic is not None:
        if hist_avg is not None:
            blended_traffic = alpha * predicted_traffic + (1 - alpha) * hist_avg
        else:
            blended_traffic = predicted_traffic

        recommended_pods = max(MIN_PODS, min(MAX_PODS, int(np.ceil(blended_traffic / traffic_per_pod))))

        print(f"Predicted={predicted_traffic:.2f}, "
              f"Hist_avg={hist_avg}, "
              f"Blended={blended_traffic:.2f}, "
              f"Current pods={current_replicas}, Recommended pods={recommended_pods}")

        predicted_g.set(blended_traffic)
        replicas_g.set(current_replicas)

        if time.time() - last_scale_time >= COOLDOWN:
            if recommended_pods != current_replicas:
                scale_deployment(recommended_pods)
            else:
                print("ℹ️ No scaling action needed.")
        else:
            print("⏳ Cooldown active, skipping scaling...")
    else:
        print("⚠️ Prediction failed, keeping current replicas.")

    time.sleep(CHECK_INTERVAL)