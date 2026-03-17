

import os
import glob
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

API_BASE = "http://localhost:8000"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
KEYS_FILE = os.path.join(os.path.dirname(__file__), "api_keys.json")

# Base coordinates
ROORKEE_LAT = 29.865
ROORKEE_LON = 77.890

DELHI_LAT = 28.6139
DELHI_LON = 77.2090

NUM_ROORKEE_SENSORS = 50
NUM_DELHI_SENSORS = 300
MAX_ROWS_PER_SENSOR = 60  # Amount of data to ingest per sensor

np.random.seed(42)  # For reproducible sensor locations

# Discover all CSV files in the data directory to use as templates
csv_files = [os.path.basename(f) for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))]
if not csv_files:
    print(f"❌ No CSV files found in {DATA_DIR}. Please add some data files.")
    exit(1)

SENSORS = []

# Generate Roorkee sensors
for i in range(1, NUM_ROORKEE_SENSORS + 1):
    lat_offset = np.random.uniform(-0.15, 0.15)
    lon_offset = np.random.uniform(-0.15, 0.15)
    SENSORS.append({
        "sensor_id": f"VRK-{1000 + i}",
        "location_name": f"Roorkee Zone {i}",
        "lat": ROORKEE_LAT + lat_offset,
        "long": ROORKEE_LON + lon_offset,
        "csv": np.random.choice(csv_files),
    })

# Generate Delhi sensors
for i in range(1, NUM_DELHI_SENSORS + 1):
    lat_offset = np.random.uniform(-0.25, 0.25)
    lon_offset = np.random.uniform(-0.25, 0.25)
    SENSORS.append({
        "sensor_id": f"DEL-{2000 + i}",
        "location_name": f"Delhi Zone {i}",
        "lat": DELHI_LAT + lat_offset,
        "long": DELHI_LON + lon_offset,
        "csv": np.random.choice(csv_files),
    })

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def z_score_anomaly(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Return boolean mask: True where |z-score| > threshold."""
    mu, sigma = series.mean(), series.std()
    if sigma == 0:
        return pd.Series([False] * len(series))
    return ((series - mu) / sigma).abs() > threshold

def register_all_sensors() -> dict:
    api_keys = {}
    if os.path.isfile(KEYS_FILE):
        with open(KEYS_FILE) as f:
            api_keys = json.load(f)
        print(f"📂 Loaded {len(api_keys)} existing API keys")

    registered_count = 0
    print(f"📡 Registering {len(SENSORS)} sensors - this might take a moment...")
    
    # We use a session for keep-alive to make this much faster
    with requests.Session() as session:
        for s in SENSORS:
            sid = s["sensor_id"]
            if sid in api_keys:
                continue
            try:
                payload = {
                    "sensor_id":    sid,
                    "location_name":s["location_name"],
                    "lat":          s["lat"],
                    "long":         s["long"],
                }
                r = session.post(f"{API_BASE}/register_sensor", json=payload, timeout=10)
                if r.status_code == 201:
                    api_keys[sid] = r.json()["api_key"]
                    registered_count += 1
                elif r.status_code == 409:
                    pass
                else:
                    print(f"  ❌ {sid}: {r.status_code} — {r.text[:80]}")
            except Exception as e:
                print(f"❌ Cannot connect to {API_BASE}. Is FastAPI running? {e}")
                raise SystemExit(1)
                
    if registered_count > 0:
        print(f"  ✅ Registered {registered_count} new sensors.")
        with open(KEYS_FILE, "w") as f:
            json.dump(api_keys, f, indent=2)
        print(f"💾 API keys saved to {KEYS_FILE}\n")
    else:
        print(f"  ⏭️ All {len(SENSORS)} sensors already registered.\n")
        
    return api_keys

def preload_csvs(max_rows: int) -> dict:
    base_dfs = {}
    for csv_file in set(s["csv"] for s in SENSORS):
        csv_path = os.path.join(DATA_DIR, csv_file)
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path).tail(max_rows).copy()
            
            # Find the pm2p5 column (might be named differently)
            pm2p5_col = "pm2p5" if "pm2p5" in df.columns else df.columns[0]
            
            df["is_anomaly"] = z_score_anomaly(pd.to_numeric(df[pm2p5_col], errors='coerce').fillna(50), threshold=3.0).astype(int)
            df.loc[pd.to_numeric(df[pm2p5_col], errors='coerce').fillna(0) > 150, "is_anomaly"] = 1
            if "is_failure" not in df.columns:
                df["is_failure"] = 0
                
            # Pre-calculate base timestamps
            timestamps = []
            time_col = "valid_at" if "valid_at" in df.columns else df.columns[0]
            for raw_ts in df[time_col]:
                try:
                    dt_obj = datetime.strptime(str(raw_ts).strip(), "%Y-%m-%d %H:%M:%S")
                    dt_obj = dt_obj.replace(year=2026, month=3)
                    timestamps.append(dt_obj.strftime("%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    timestamps.append("2026-03-06 00:00:00")
            df["parsed_time"] = timestamps
            base_dfs[csv_file] = df
            
    return base_dfs

def ingest_all_sensors(api_keys: dict, max_rows: int = 60):
    print("📥 Preloading all base datasets...")
    base_dfs = preload_csvs(max_rows)

    print(f"🔥 Generating and ingesting ~{max_rows} readings for {len(SENSORS)} sensors = {max_rows * len(SENSORS)} total points...")
    
    success, failed = 0, 0
    
    with requests.Session() as session:
        for idx, sensor in enumerate(SENSORS):
            sensor_id = sensor["sensor_id"]
            if sensor_id not in api_keys:
                continue
                
            headers = {"Content-Type": "application/json", "x-api-key": api_keys[sensor_id]}
            base_df = base_dfs.get(sensor["csv"])
            if base_df is None:
                continue
            
            for i, row in base_df.iterrows():
                try:
                    # Add random spatial noise
                    noise_factor = np.random.normal(1.0, 0.45) # larger variance
                    
                    pm2p5_col = "pm2p5" if "pm2p5" in row else base_df.columns[0]
                    raw_val = row[pm2p5_col]
                    raw_val = float(raw_val) if pd.notna(raw_val) else 50.0
                    
                    pm25_raw = max(0, raw_val * noise_factor)
                    humidity = float(row.get("relative_humidity", 60.0))
                    if pd.isna(humidity): humidity = 60.0
                    temp     = float(row.get("temperature", 25.0))
                    if pd.isna(temp): temp = 25.0
                    pressure = float(row.get("pressure", 1013.0))
                    if pd.isna(pressure): pressure = 1013.0
                    wind     = float(row.get("wind_speed", 2.0))
                    if pd.isna(wind): wind = 2.0
                    cloud    = float(row.get("cloud_coverage", 30.0))
                    if pd.isna(cloud): cloud = 30.0

                    # Mock /predict locally for speed or call real endpoint
                    try:
                        pred_r = session.post(
                            f"{API_BASE}/predict",
                            json={"features": [pm25_raw, humidity, temp, pressure, wind, cloud, pm25_raw]},
                            timeout=2,
                        )
                        pm25_corrected = (
                            pred_r.json().get("predicted_pm2p5_corrected", pm25_raw)
                            if pred_r.status_code == 200 else pm25_raw
                        )
                    except:
                        pm25_corrected = pm25_raw

                    is_anomaly = 1 if pm25_raw > 150 else int(row.get("is_anomaly", 0))

                    payload = {
                        "sensor_id":       sensor_id,
                        "timestamp":       row["parsed_time"],
                        "temperature":     round(temp, 2),
                        "humidity":        round(humidity, 2),
                        "pm2p5_raw":       round(pm25_raw, 4),
                        "pm2p5_corrected": round(float(pm25_corrected), 4),
                        "is_anomaly":      is_anomaly,
                        "is_failure":      int(row.get("is_failure", 0)),
                    }

                    r = session.post(f"{API_BASE}/ingest", json=payload, headers=headers, timeout=2)
                    if r.status_code == 200:
                        success += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    
            if (idx + 1) % 20 == 0:
                print(f"    … Finished sensor {idx+1}/{len(SENSORS)} ({success} total ingestions)")

    print(f"  ✅ Done: {success} ingested, {failed} failed.")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  Vayu-Rakshak — Massive Dataset Seeder")
    print("  Generating 350 sensors across Roorkee and Delhi")
    print("=" * 64)

    api_keys = register_all_sensors()
    ingest_all_sensors(api_keys, max_rows=MAX_ROWS_PER_SENSOR)

    print("\n🎉 Seeding complete!")
    print("\n   Open http://localhost:8501 to explore the Streamlit dashboard.")

if __name__ == "__main__":
    main()
