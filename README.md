# 🌿 Vayu-Rakshak — Air Quality Monitoring System

A full-stack, real-time air quality monitoring platform built with **FastAPI**, **Streamlit**, **PyTorch**, and **LangChain**.

---

## 📁 Project Structure

```
project micro/
├── main.py              # FastAPI backend (all endpoints, security, background tasks)
├── database.py          # SQLAlchemy models (SensorRegistry + SensorReadings)
├── model_utils.py       # PyTorch model loader + inference
├── agent.py             # LangChain SQL agent + Overpass POI tool ("Dr. Vayu")
├── app.py               # Streamlit dashboard (4 tabs + chatbot)
├── seed_data.py         # One-time DB seeder from CSV files
├── requirements.txt     # All Python dependencies
├── .env.example         # Environment variable template
├── air_quality_model/   # Trained PyTorch model directory
├── air_quality_model.pth# Fallback .pth model checkpoint
└── data/                # Sensor CSV files (ARI-1727 ... ARI-2049)
```

---

## ⚙️ Setup & Installation

### 1. Create a virtual environment
```bash
cd "/home/priyansh/Desktop/project micro"
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key (needed only for the chatbot)
nano .env
```

---

## 🚀 Running the Application

### Terminal 1 — FastAPI Backend
```bash
cd "/home/priyansh/Desktop/project micro"
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

FastAPI docs → http://localhost:8000/docs

### Terminal 2 — Streamlit Dashboard
```bash
cd "/home/priyansh/Desktop/project micro"
source .venv/bin/activate
streamlit run app.py
```

Dashboard → http://localhost:8501

### Terminal 3 — Seed the Database (run once)
```bash
# Make sure FastAPI is running first!
python seed_data.py
```

---

## 🔌 API Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/register_sensor` | None | Register sensor, get api_key |
| `POST` | `/ingest` | `x-api-key` header | Ingest a sensor reading |
| `POST` | `/predict` | None | PM2.5 correction via PyTorch model |
| `GET`  | `/sensors` | None | List all sensors |
| `GET`  | `/readings` | None | All readings (with coords) |
| `GET`  | `/readings/{sensor_id}` | None | Sensor-specific readings |
| `GET`  | `/health` | None | Health check |

### Example: Register a sensor
```bash
curl -X POST http://localhost:8000/register_sensor \
  -H "Content-Type: application/json" \
  -d '{"sensor_id":"ARI-9999","location_name":"Test Area","lat":28.61,"long":77.21}'
# Returns: {"sensor_id":"ARI-9999", "api_key":"<uuid>", ...}
```

### Example: Ingest data
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "x-api-key: <your-api-key>" \
  -d '{"sensor_id":"ARI-9999","timestamp":"2026-03-06 00:00:00","temperature":25.4,"humidity":60.2,"pm2p5_raw":160.5,"pm2p5_corrected":155.1,"is_anomaly":1,"is_failure":0}'
# Triggers: 🚨 ALERT logged to console (pm2p5 > 150 and anomaly)
```

### Example: ML Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[150.0,60.0,25.0,1012.0,2.0,30.0,145.0]}'
# Returns: {"predicted_pm2p5_corrected": 142.37, "unit": "µg/m³"}
```

---

## 🤖 AI Chatbot — Dr. Vayu

The chatbot is a **LangChain SQL Agent** with:
- Direct SQL access to the SQLite database
- A custom **Overpass API POI tool** to find factories, schools, roads near sensors
- A system prompt: "You are Dr. Vayu, a senior environmental scientist…"

**Example questions:**
- *"Why is PM2.5 high at ARI-1885 in the morning?"*
- *"Which sensor has the most anomalies this week?"*
- *"What are the pollution sources near Chandni Chowk?"*
- *"Show me sensors with pm2p5 > 100 in the last 3 days."*

Activate by entering your `OPENAI_API_KEY` in the sidebar.

---

## 🔒 Security

- Every sensor registration generates a **UUID v4 api_key**
- The `/ingest` endpoint refuses requests without a valid `x-api-key` header
- Each sensor's key is validated against its own DB record (not globally shared)

---

## 🚨 Background Alerting

Automatic alerts are triggered when:
- `is_anomaly == 1` → Anomaly detected
- `pm2p5_corrected > 150 µg/m³` → WHO Hazardous threshold exceeded

Alerts are logged to the FastAPI server console:
```
🚨 ALERT: HIGH PM2.5 (155.1 µg/m³ > 150 threshold) at Sensor [ARI-1885]!
```

---

## 🌡️ PM2.5 Health Scale (WHO 2021)

| Range (µg/m³) | Category |
|---|---|
| < 12 | Good |
| 12 – 35 | Moderate |
| 35 – 55 | Unhealthy for Sensitive Groups |
| 55 – 150 | Unhealthy |
| > 150 | Very Unhealthy / Hazardous |
# Vayu-Rakshak
