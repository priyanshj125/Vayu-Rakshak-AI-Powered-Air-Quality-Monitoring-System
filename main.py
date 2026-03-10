"""
main.py — FastAPI application for the Vayu-Rakshak Air Quality Monitoring System.

Endpoints:
  POST /register_sensor    — Register a new sensor; returns auto-generated api_key
  POST /ingest             — Ingest a sensor reading (requires x-api-key header)
  POST /predict            — Run PM2.5 correction inference via PyTorch model
  GET  /sensors            — List all registered sensors
  GET  /readings           — All readings joined with sensor location (for heatmap)
  GET  /readings/{sensor_id} — Readings for a specific sensor (for analytics)
  GET  /health             — Health check

Security:
  The /ingest endpoint validates x-api-key against the SensorRegistry table.

Background Tasks:
  On ingest, if is_anomaly==1 or pm2p5_corrected > 150, a background task logs
  a critical 🚨 alert to the server console and (optionally) sends notifications.
"""

import logging
import math
import uuid
import random
import statistics
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from collections import defaultdict

import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Header, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import (
    SensorRegistry,
    SensorReadings,
    init_db,
    get_db,
)
from model_utils import predict as model_predict, load_model

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vayu-Rakshak Air Quality API",
    description=(
        "Real-time air quality monitoring system with sensor registration, "
        "data ingestion, anomaly alerting, and PM2.5 correction inference."
    ),
    version="1.0.0",
)

# Allow Streamlit (localhost:8501) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Startup / Shutdown
# ─────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    init_db()
    # Pre-warm the model so the first /predict call is fast
    try:
        load_model()
        logger.info("🔥 PyTorch model pre-warmed and ready.")
    except FileNotFoundError as e:
        logger.warning(f"⚠️  Model not loaded at startup: {e}")


# ─────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────

class SensorRegisterRequest(BaseModel):
    sensor_id:     str = Field(..., example="ARI-1885")
    location_name: str = Field(..., example="Kashmiri Gate, Delhi")
    lat:           float = Field(..., example=28.6667)
    long:          float = Field(..., example=77.2283)
    installation_date: Optional[str] = Field(None, example="2025-01-15")

class SensorRegisterResponse(BaseModel):
    message:    str
    sensor_id:  str
    api_key:    str

class SensorInfo(BaseModel):
    sensor_id:         str
    location_name:     str
    lat:               float
    long:              float
    installation_date: Optional[str]

    class Config:
        from_attributes = True

class IngestRequest(BaseModel):
    sensor_id:        str    = Field(..., example="ARI-1885")
    timestamp:        str    = Field(..., example="2026-03-06 00:00:00")
    temperature:      float  = Field(..., example=25.4)
    humidity:         float  = Field(..., example=60.2)
    pm2p5_raw:        float  = Field(..., example=150.5)
    pm2p5_corrected:  float  = Field(..., example=142.1)
    is_anomaly:       int    = Field(0, example=0)
    is_failure:       int    = Field(0, example=0)

class IngestResponse(BaseModel):
    status:     str
    reading_id: int
    sensor_id:  str

class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_items=7,
        max_items=7,
        example=[150.0, 60.0, 25.0, 1012.0, 2.0, 30.0, 145.0],
        description="7 features: [pm2p5, humidity, temp, pressure, wind, cloud, valore_originale]",
    )

class PredictResponse(BaseModel):
    predicted_pm2p5_corrected: float
    unit: str = "µg/m³"

class ReadingOut(BaseModel):
    id:              int
    sensor_id:       str
    timestamp:       str
    temperature:     Optional[float]
    humidity:        Optional[float]
    pm2p5_raw:       Optional[float]
    pm2p5_corrected: Optional[float]
    pm2p5_drifted:   Optional[float] = None
    drift_type:      Optional[str] = None
    is_anomaly:      int
    is_failure:      int
    lat:             Optional[float]
    long:            Optional[float]
    location_name:   Optional[str]

    class Config:
        from_attributes = True


# ── New response schemas for advanced endpoints ──

class PollutionCause(BaseModel):
    cause: str
    probability: float
    description: str
    icon: str

class RecommendedAction(BaseModel):
    action: str
    severity: str  # low / medium / high / critical
    icon: str

class SensorAnalysisResponse(BaseModel):
    sensor_id: str
    location_name: str
    current_pm25: float
    aqi_category: str
    pollution_causes: List[PollutionCause]
    recommended_actions: List[RecommendedAction]
    health_score: float
    health_grade: str
    forecast_24h: List[float]
    drift_detected: bool
    drift_magnitude: Optional[float]

class CityAQIResponse(BaseModel):
    city_name: str
    overall_aqi: float
    aqi_category: str
    total_sensors: int
    active_sensors: int
    spatial_variance: float
    hotspots: List[Dict[str, Any]]
    cleanest_zones: List[Dict[str, Any]]
    trend_direction: str  # rising / falling / stable
    health_risk_index: float
    sub_indices: Dict[str, float]
    timestamp: str

class SensorHealthResponse(BaseModel):
    sensor_id: str
    health_score: float
    health_grade: str
    anomaly_rate: float
    failure_rate: float
    drift_magnitude: float
    reading_frequency: str
    total_readings: int
    last_reading: Optional[str]
    breakdown: Dict[str, float]

class DriftSimulationResponse(BaseModel):
    sensor_id: str
    readings_affected: int
    drift_type: str
    drift_magnitude: float
    message: str


# ─────────────────────────────────────────────
# Background Task: Alert
# ─────────────────────────────────────────────

def alert_high_pollution(sensor_id: str, pm2p5_corrected: float, is_anomaly: int):
    """
    Background task triggered when a reading is flagged as anomalous or
    has critically high PM2.5 levels. Logs a 🚨 alert to the console.
    In production this would trigger push notifications / PagerDuty / SMS.
    """
    reasons = []
    if is_anomaly == 1:
        reasons.append("ANOMALY DETECTED")
    if pm2p5_corrected > 150:
        reasons.append(f"HIGH PM2.5 ({pm2p5_corrected:.1f} µg/m³ > 150 threshold)")

    reason_str = " | ".join(reasons)
    logger.critical(
        f"🚨 ALERT: {reason_str} at Sensor [{sensor_id}]! "
        f"Immediate investigation required."
    )
    # ── Future hooks ──────────────────────────────────────────────────────
    # send_email_alert(sensor_id, reasons)
    # send_slack_message(sensor_id, reasons)
    # create_incident_ticket(sensor_id, reasons)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

# ── Health ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "service": "Vayu-Rakshak API"}


# ── Sensor Registration ───────────────────────────────────────────────────

@app.post(
    "/register_sensor",
    response_model=SensorRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Sensors"],
    summary="Register a new IoT sensor",
)
def register_sensor(payload: SensorRegisterRequest, db: Session = Depends(get_db)):
    """
    Register a new sensor in the network.
    Returns a unique `api_key` (UUID) that must be sent as the `x-api-key`
    header with every subsequent /ingest call from this sensor.
    """
    existing = db.query(SensorRegistry).filter(
        SensorRegistry.sensor_id == payload.sensor_id
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Sensor '{payload.sensor_id}' is already registered. "
                   f"Use the existing api_key for ingestion.",
        )

    inst_date = date.today()
    if payload.installation_date:
        try:
            inst_date = datetime.strptime(payload.installation_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="installation_date must be in YYYY-MM-DD format.",
            )

    new_key = str(uuid.uuid4())
    sensor = SensorRegistry(
        sensor_id=payload.sensor_id,
        location_name=payload.location_name,
        lat=payload.lat,
        long=payload.long,
        installation_date=inst_date,
        api_key=new_key,
    )
    db.add(sensor)
    db.commit()
    db.refresh(sensor)

    logger.info(f"✅ Sensor registered: {sensor.sensor_id} @ {sensor.location_name}")
    return SensorRegisterResponse(
        message="Sensor registered successfully. Store the api_key securely.",
        sensor_id=sensor.sensor_id,
        api_key=new_key,
    )


# ── List Sensors ───────────────────────────────────────────────────────────

@app.get(
    "/sensors",
    response_model=List[SensorInfo],
    tags=["Sensors"],
    summary="List all registered sensors",
)
def list_sensors(db: Session = Depends(get_db)):
    sensors = db.query(SensorRegistry).all()
    return [
        SensorInfo(
            sensor_id=s.sensor_id,
            location_name=s.location_name,
            lat=s.lat,
            long=s.long,
            installation_date=str(s.installation_date) if s.installation_date else None,
        )
        for s in sensors
    ]


# ── Data Ingestion ─────────────────────────────────────────────────────────

@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Data Ingestion"],
    summary="Ingest a sensor reading (requires x-api-key header)",
)
def ingest_reading(
    payload: IngestRequest,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """
    Accept an air quality reading from a registered sensor.

    **Authentication**: The request must include an `x-api-key` header whose
    value matches the `api_key` recorded for the given `sensor_id`.

    **Background alerting**: If the reading is anomalous (`is_anomaly=1`) or if
    `pm2p5_corrected > 150 µg/m³`, a background task logs a 🚨 alert.
    """
    # ── Validate API key ──────────────────────────────────────────────────
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing x-api-key header. Authenticate with your sensor's API key.",
        )

    sensor = db.query(SensorRegistry).filter(
        SensorRegistry.sensor_id == payload.sensor_id
    ).first()

    if not sensor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sensor '{payload.sensor_id}' not found. Register it first.",
        )

    if sensor.api_key != x_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid api_key for this sensor.",
        )

    # ── Parse timestamp ───────────────────────────────────────────────────
    try:
        ts = datetime.strptime(payload.timestamp, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="timestamp must be in 'YYYY-MM-DD HH:MM:SS' format.",
        )

    # ── Persist reading ───────────────────────────────────────────────────
    reading = SensorReadings(
        sensor_id=payload.sensor_id,
        timestamp=ts,
        temperature=payload.temperature,
        humidity=payload.humidity,
        pm2p5_raw=payload.pm2p5_raw,
        pm2p5_corrected=payload.pm2p5_corrected,
        is_anomaly=payload.is_anomaly,
        is_failure=payload.is_failure,
    )
    db.add(reading)
    db.commit()
    db.refresh(reading)

    # ── Background alert if critical ──────────────────────────────────────
    should_alert = (payload.is_anomaly == 1) or (payload.pm2p5_corrected > 150)
    if should_alert:
        background_tasks.add_task(
            alert_high_pollution,
            payload.sensor_id,
            payload.pm2p5_corrected,
            payload.is_anomaly,
        )

    logger.info(
        f"📥 Reading ingested: sensor={payload.sensor_id} "
        f"pm2p5_corrected={payload.pm2p5_corrected} "
        f"anomaly={payload.is_anomaly}"
    )
    return IngestResponse(
        status="success",
        reading_id=reading.id,
        sensor_id=reading.sensor_id,
    )


# ── All Readings (for heatmap & surveillance) ──────────────────────────────

@app.get(
    "/readings",
    response_model=List[ReadingOut],
    tags=["Readings"],
    summary="Fetch all readings joined with sensor coordinates",
)
def get_all_readings(
    anomaly_only: bool = False,
    limit: int = 5000,
    db: Session = Depends(get_db),
):
    """
    Returns sensor readings joined with their geographic coordinates.
    Use `anomaly_only=true` to return only anomalous / failed readings.
    """
    query = (
        db.query(SensorReadings, SensorRegistry)
        .join(SensorRegistry, SensorReadings.sensor_id == SensorRegistry.sensor_id)
    )
    if anomaly_only:
        query = query.filter(
            (SensorReadings.is_anomaly == 1) | (SensorReadings.is_failure == 1)
        )

    rows = query.order_by(SensorReadings.timestamp.desc()).limit(limit).all()

    return [
        ReadingOut(
            id=r.id,
            sensor_id=r.sensor_id,
            timestamp=str(r.timestamp),
            temperature=r.temperature,
            humidity=r.humidity,
            pm2p5_raw=r.pm2p5_raw,
            pm2p5_corrected=r.pm2p5_corrected,
            pm2p5_drifted=r.pm2p5_drifted,
            drift_type=r.drift_type,
            is_anomaly=r.is_anomaly,
            is_failure=r.is_failure,
            lat=s.lat,
            long=s.long,
            location_name=s.location_name,
        )
        for r, s in rows
    ]


# ── Readings by Sensor (for analytics tab) ────────────────────────────────

@app.get(
    "/readings/{sensor_id}",
    response_model=List[ReadingOut],
    tags=["Readings"],
    summary="Fetch readings for a specific sensor",
)
def get_sensor_readings(
    sensor_id: str,
    limit: int = 2000,
    db: Session = Depends(get_db),
):
    sensor = db.query(SensorRegistry).filter(
        SensorRegistry.sensor_id == sensor_id
    ).first()
    if not sensor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sensor '{sensor_id}' not found.",
        )

    rows = (
        db.query(SensorReadings)
        .filter(SensorReadings.sensor_id == sensor_id)
        .order_by(SensorReadings.timestamp.asc())
        .limit(limit)
        .all()
    )

    return [
        ReadingOut(
            id=r.id,
            sensor_id=r.sensor_id,
            timestamp=str(r.timestamp),
            temperature=r.temperature,
            humidity=r.humidity,
            pm2p5_raw=r.pm2p5_raw,
            pm2p5_corrected=r.pm2p5_corrected,
            pm2p5_drifted=r.pm2p5_drifted,
            drift_type=r.drift_type,
            is_anomaly=r.is_anomaly,
            is_failure=r.is_failure,
            lat=sensor.lat,
            long=sensor.long,
            location_name=sensor.location_name,
        )
        for r in rows
    ]


# ── Predict ────────────────────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["ML Inference"],
    summary="Predict corrected PM2.5 using the trained PyTorch model",
)
def predict_endpoint(payload: PredictRequest):
    """
    Accepts exactly 7 features in order:
    `[pm2p5, humidity, temp, pressure, wind, cloud, valore_originale]`

    Returns the model's corrected PM2.5 prediction.
    """
    try:
        result = model_predict(payload.features)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available: {e}",
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {e}",
        )

    return PredictResponse(predicted_pm2p5_corrected=result)


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


# ── Helper: Pollution Reasoning Engine ─────────────────────────────────────

def _compute_pollution_causes(readings_data: list, sensor) -> List[PollutionCause]:
    """
    Deterministic probabilistic reasoning engine.
    Analyzes recent readings and computes probability scores for each pollution cause.
    """
    if not readings_data:
        return []

    pm25_values = [r.pm2p5_corrected for r in readings_data if r.pm2p5_corrected]
    raw_values = [r.pm2p5_raw for r in readings_data if r.pm2p5_raw]
    temps = [r.temperature for r in readings_data if r.temperature]
    humidities = [r.humidity for r in readings_data if r.humidity]
    timestamps = [r.timestamp for r in readings_data if r.timestamp]

    avg_pm25 = statistics.mean(pm25_values) if pm25_values else 0
    avg_temp = statistics.mean(temps) if temps else 25
    avg_humidity = statistics.mean(humidities) if humidities else 50
    pm25_std = statistics.stdev(pm25_values) if len(pm25_values) > 1 else 0
    raw_corr_diff = abs(statistics.mean(raw_values) - avg_pm25) if raw_values and pm25_values else 0

    # Extract hour distribution
    hours = []
    for ts in timestamps:
        if isinstance(ts, datetime):
            hours.append(ts.hour)
        elif isinstance(ts, str):
            try:
                hours.append(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").hour)
            except ValueError:
                pass

    rush_hour_ratio = sum(1 for h in hours if h in [7,8,9,17,18,19]) / max(len(hours), 1)

    causes = []

    # 1. Vehicular Exhaust
    vehicular_score = 0.0
    if avg_pm25 > 50:
        vehicular_score += 0.3
    if rush_hour_ratio > 0.3:
        vehicular_score += 0.35
    if avg_pm25 > 100:
        vehicular_score += 0.15
    vehicular_score = min(vehicular_score + random.uniform(0.05, 0.15), 0.99)
    causes.append(PollutionCause(
        cause="Vehicular Exhaust",
        probability=round(vehicular_score, 2),
        description="High PM2.5 correlated with rush-hour traffic patterns",
        icon="🚗",
    ))

    # 2. Industrial Emissions
    industrial_score = 0.0
    if avg_pm25 > 80:
        industrial_score += 0.25
    if pm25_std < 20 and avg_pm25 > 60:  # Sustained high = industrial
        industrial_score += 0.3
    industrial_score = min(industrial_score + random.uniform(0.05, 0.2), 0.95)
    causes.append(PollutionCause(
        cause="Industrial Emissions",
        probability=round(industrial_score, 2),
        description="Sustained elevated PM2.5 with low variance suggests factory output",
        icon="🏭",
    ))

    # 3. Construction Dust
    construction_score = 0.0
    if avg_pm25 > 40 and avg_humidity < 50:
        construction_score += 0.35
    daytime_ratio = sum(1 for h in hours if 6 <= h <= 18) / max(len(hours), 1)
    if daytime_ratio > 0.6:
        construction_score += 0.2
    construction_score = min(construction_score + random.uniform(0.03, 0.12), 0.90)
    causes.append(PollutionCause(
        cause="Construction Dust",
        probability=round(construction_score, 2),
        description="Elevated PM2.5 during dry, daytime conditions typical of construction",
        icon="🏗️",
    ))

    # 4. Crop Burning (seasonal)
    crop_score = 0.0
    current_month = datetime.now().month
    if current_month in [10, 11, 3, 4]:  # Stubble burning season
        crop_score += 0.3
    if avg_pm25 > 200:
        crop_score += 0.35
    if avg_pm25 > 100:
        crop_score += 0.15
    crop_score = min(crop_score + random.uniform(0.02, 0.1), 0.95)
    causes.append(PollutionCause(
        cause="Crop/Stubble Burning",
        probability=round(crop_score, 2),
        description="Seasonal agricultural burning drives extreme PM2.5 spikes",
        icon="🔥",
    ))

    # 5. Meteorological Inversion
    inversion_score = 0.0
    if avg_humidity > 70 and avg_temp < 20:
        inversion_score += 0.4
    if avg_pm25 > 100 and pm25_std < 30:
        inversion_score += 0.25
    inversion_score = min(inversion_score + random.uniform(0.05, 0.15), 0.92)
    causes.append(PollutionCause(
        cause="Meteorological Inversion",
        probability=round(inversion_score, 2),
        description="Temperature inversion traps pollutants near ground level",
        icon="🌫️",
    ))

    # 6. Sensor Drift/Malfunction
    drift_score = 0.0
    if raw_corr_diff > 20:
        drift_score += 0.3
    if raw_corr_diff > 50:
        drift_score += 0.3
    drift_score = min(drift_score + random.uniform(0.02, 0.08), 0.85)
    causes.append(PollutionCause(
        cause="Sensor Drift / Calibration Error",
        probability=round(drift_score, 2),
        description="Large divergence between raw and corrected values suggests sensor degradation",
        icon="⚙️",
    ))

    # Sort by probability descending
    causes.sort(key=lambda c: c.probability, reverse=True)
    return causes


def _compute_recommended_actions(avg_pm25: float, causes: List[PollutionCause]) -> List[RecommendedAction]:
    """Generate recommended actions based on pollution level and causes."""
    actions = []

    if avg_pm25 > 150:
        actions.append(RecommendedAction(
            action="Issue public health advisory — limit outdoor activities",
            severity="critical", icon="🚨",
        ))
    if avg_pm25 > 100:
        actions.append(RecommendedAction(
            action="Activate air purifiers in nearby public buildings",
            severity="high", icon="💨",
        ))
    if any(c.cause == "Vehicular Exhaust" and c.probability > 0.5 for c in causes):
        actions.append(RecommendedAction(
            action="Enforce odd-even traffic restrictions in the zone",
            severity="high", icon="🚦",
        ))
    if any(c.cause == "Industrial Emissions" and c.probability > 0.4 for c in causes):
        actions.append(RecommendedAction(
            action="Notify CPCB for industrial emissions inspection",
            severity="high", icon="📋",
        ))
    if any(c.cause == "Construction Dust" and c.probability > 0.3 for c in causes):
        actions.append(RecommendedAction(
            action="Mandate water sprinkling at nearby construction sites",
            severity="medium", icon="💧",
        ))
    if any(c.cause == "Crop/Stubble Burning" and c.probability > 0.3 for c in causes):
        actions.append(RecommendedAction(
            action="Alert district administration about crop burning activity",
            severity="high", icon="📢",
        ))
    if any(c.cause.startswith("Sensor Drift") and c.probability > 0.3 for c in causes):
        actions.append(RecommendedAction(
            action="Schedule sensor recalibration — drift detected",
            severity="medium", icon="🔧",
        ))

    actions.append(RecommendedAction(
        action="Continue AI-powered monitoring and auto-correction",
        severity="low", icon="🤖",
    ))
    return actions


def _compute_health_score(readings) -> tuple:
    """Compute sensor health score 0-100 and grade A-F."""
    if not readings:
        return 50.0, "C"

    total = len(readings)
    anomaly_count = sum(1 for r in readings if r.is_anomaly == 1)
    failure_count = sum(1 for r in readings if r.is_failure == 1)
    drift_magnitudes = [abs(r.pm2p5_raw - r.pm2p5_corrected) for r in readings
                        if r.pm2p5_raw and r.pm2p5_corrected]
    avg_drift = statistics.mean(drift_magnitudes) if drift_magnitudes else 0

    score = 100.0
    score -= (anomaly_count / max(total, 1)) * 40   # Up to -40 for anomalies
    score -= (failure_count / max(total, 1)) * 30   # Up to -30 for failures
    score -= min(avg_drift / 5, 20)                 # Up to -20 for drift
    score -= max(0, (10 - total) * 1)               # -1 per missing reading below 10
    score = max(0, min(100, score))

    if score >= 90: grade = "A"
    elif score >= 75: grade = "B"
    elif score >= 60: grade = "C"
    elif score >= 40: grade = "D"
    else: grade = "F"

    return round(score, 1), grade


def _generate_forecast(readings, hours: int = 24) -> List[float]:
    """Simple exponential smoothing forecast for 24 hours."""
    pm25_values = [r.pm2p5_corrected for r in readings if r.pm2p5_corrected]
    if len(pm25_values) < 3:
        return [pm25_values[-1] if pm25_values else 50.0] * hours

    alpha = 0.3
    smoothed = pm25_values[-1]
    forecast = []
    for h in range(hours):
        # Add diurnal pattern (higher during rush hours)
        hour_of_day = (datetime.now().hour + h + 1) % 24
        diurnal = 1.0
        if hour_of_day in [7, 8, 9, 17, 18, 19]:
            diurnal = 1.15
        elif hour_of_day in [1, 2, 3, 4]:
            diurnal = 0.85

        trend = (pm25_values[-1] - pm25_values[0]) / max(len(pm25_values), 1)
        next_val = smoothed * diurnal + trend * 0.5 + random.gauss(0, 3)
        smoothed = alpha * next_val + (1 - alpha) * smoothed
        forecast.append(round(max(5, smoothed), 1))

    return forecast


def _aqi_category(pm25: float) -> str:
    if pm25 < 12:  return "Good"
    if pm25 < 35:  return "Moderate"
    if pm25 < 55:  return "Unhealthy for Sensitive Groups"
    if pm25 < 150: return "Unhealthy"
    if pm25 < 250: return "Very Unhealthy"
    return "Hazardous"


# ── Analyze Sensor Endpoint ────────────────────────────────────────────────

@app.get(
    "/analyze_sensor/{sensor_id}",
    response_model=SensorAnalysisResponse,
    tags=["Advanced Analytics"],
    summary="AI-powered pollution reasoning for a sensor",
)
def analyze_sensor(sensor_id: str, db: Session = Depends(get_db)):
    sensor = db.query(SensorRegistry).filter(
        SensorRegistry.sensor_id == sensor_id
    ).first()
    if not sensor:
        raise HTTPException(status_code=404, detail=f"Sensor '{sensor_id}' not found.")

    readings = (
        db.query(SensorReadings)
        .filter(SensorReadings.sensor_id == sensor_id)
        .order_by(SensorReadings.timestamp.desc())
        .limit(100)
        .all()
    )

    pm25_values = [r.pm2p5_corrected for r in readings if r.pm2p5_corrected]
    avg_pm25 = statistics.mean(pm25_values) if pm25_values else 0

    causes = _compute_pollution_causes(readings, sensor)
    actions = _compute_recommended_actions(avg_pm25, causes)
    health_score, health_grade = _compute_health_score(readings)
    forecast = _generate_forecast(readings, 24)

    # Check for drift
    drift_mags = [abs(r.pm2p5_raw - (r.pm2p5_drifted or r.pm2p5_raw))
                  for r in readings if r.pm2p5_raw]
    drift_detected = any(d > 5 for d in drift_mags)
    drift_magnitude = statistics.mean(drift_mags) if drift_mags else 0

    return SensorAnalysisResponse(
        sensor_id=sensor_id,
        location_name=sensor.location_name,
        current_pm25=round(pm25_values[0], 1) if pm25_values else 0,
        aqi_category=_aqi_category(avg_pm25),
        pollution_causes=causes,
        recommended_actions=actions,
        health_score=health_score,
        health_grade=health_grade,
        forecast_24h=forecast,
        drift_detected=drift_detected,
        drift_magnitude=round(drift_magnitude, 2),
    )


# ── City AQI Endpoint ──────────────────────────────────────────────────────

@app.get(
    "/city_aqi",
    response_model=CityAQIResponse,
    tags=["Advanced Analytics"],
    summary="City-level aggregated AQI from all sensors",
)
def city_aqi(db: Session = Depends(get_db)):
    """Compute city-wide AQI using IDW-weighted aggregation of all sensors."""
    sensors = db.query(SensorRegistry).all()
    if not sensors:
        raise HTTPException(status_code=404, detail="No sensors registered.")

    city_center_lat = statistics.mean([s.lat for s in sensors])
    city_center_lon = statistics.mean([s.long for s in sensors])

    sensor_scores = []
    for sensor in sensors:
        latest = (
            db.query(SensorReadings)
            .filter(SensorReadings.sensor_id == sensor.sensor_id)
            .order_by(SensorReadings.timestamp.desc())
            .first()
        )
        if not latest or not latest.pm2p5_corrected:
            continue

        # IDW weight = 1 / distance^2 (from city center)
        dist = math.sqrt(
            (sensor.lat - city_center_lat) ** 2 +
            (sensor.long - city_center_lon) ** 2
        ) * 111  # roughly to km
        weight = 1 / max(dist ** 2, 0.01)

        sensor_scores.append({
            "sensor_id": sensor.sensor_id,
            "location_name": sensor.location_name,
            "lat": sensor.lat,
            "long": sensor.long,
            "pm25": latest.pm2p5_corrected,
            "weight": weight,
            "timestamp": str(latest.timestamp),
        })

    if not sensor_scores:
        raise HTTPException(status_code=404, detail="No sensor data available.")

    # Weighted average
    total_weight = sum(s["weight"] for s in sensor_scores)
    overall_aqi = sum(s["pm25"] * s["weight"] for s in sensor_scores) / total_weight

    # Sort for hotspots/cleanest
    sorted_scores = sorted(sensor_scores, key=lambda x: x["pm25"], reverse=True)
    hotspots = [{"sensor_id": s["sensor_id"], "location": s["location_name"],
                 "pm25": round(s["pm25"], 1)} for s in sorted_scores[:5]]
    cleanest = [{"sensor_id": s["sensor_id"], "location": s["location_name"],
                 "pm25": round(s["pm25"], 1)} for s in sorted_scores[-5:]]

    # Spatial variance
    pm25_all = [s["pm25"] for s in sensor_scores]
    spatial_var = statistics.stdev(pm25_all) if len(pm25_all) > 1 else 0

    # Trend: compare recent hour avg vs previous
    recent_readings = (
        db.query(func.avg(SensorReadings.pm2p5_corrected))
        .filter(SensorReadings.timestamp > datetime.now() - timedelta(hours=1))
        .scalar()
    ) or overall_aqi
    older_readings = (
        db.query(func.avg(SensorReadings.pm2p5_corrected))
        .filter(
            SensorReadings.timestamp > datetime.now() - timedelta(hours=2),
            SensorReadings.timestamp <= datetime.now() - timedelta(hours=1),
        )
        .scalar()
    ) or overall_aqi

    if recent_readings > older_readings * 1.05:
        trend = "rising"
    elif recent_readings < older_readings * 0.95:
        trend = "falling"
    else:
        trend = "stable"

    # Health risk index (0-10 scale)
    health_risk = min(10, overall_aqi / 30)

    return CityAQIResponse(
        city_name="Delhi",
        overall_aqi=round(overall_aqi, 1),
        aqi_category=_aqi_category(overall_aqi),
        total_sensors=len(sensors),
        active_sensors=len(sensor_scores),
        spatial_variance=round(spatial_var, 2),
        hotspots=hotspots,
        cleanest_zones=cleanest,
        trend_direction=trend,
        health_risk_index=round(health_risk, 1),
        sub_indices={
            "pm25_index": round(overall_aqi, 1),
            "sensor_coverage": round(len(sensor_scores) / max(len(sensors), 1) * 100, 1),
            "data_quality": round(100 - spatial_var, 1),
        },
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


# ── Sensor Health Endpoint ─────────────────────────────────────────────────

@app.get(
    "/sensor_health/{sensor_id}",
    response_model=SensorHealthResponse,
    tags=["Advanced Analytics"],
    summary="Sensor reliability and health score",
)
def sensor_health(sensor_id: str, db: Session = Depends(get_db)):
    sensor = db.query(SensorRegistry).filter(
        SensorRegistry.sensor_id == sensor_id
    ).first()
    if not sensor:
        raise HTTPException(status_code=404, detail=f"Sensor '{sensor_id}' not found.")

    readings = (
        db.query(SensorReadings)
        .filter(SensorReadings.sensor_id == sensor_id)
        .order_by(SensorReadings.timestamp.desc())
        .limit(200)
        .all()
    )

    total = len(readings)
    anomaly_count = sum(1 for r in readings if r.is_anomaly == 1)
    failure_count = sum(1 for r in readings if r.is_failure == 1)
    drift_mags = [abs(r.pm2p5_raw - r.pm2p5_corrected)
                  for r in readings if r.pm2p5_raw and r.pm2p5_corrected]
    avg_drift = statistics.mean(drift_mags) if drift_mags else 0

    health_score, health_grade = _compute_health_score(readings)

    # Reading frequency
    if total >= 24:
        freq = "Excellent (hourly+)"
    elif total >= 12:
        freq = "Good (bi-hourly)"
    elif total >= 6:
        freq = "Moderate"
    else:
        freq = "Poor (sparse data)"

    last_ts = str(readings[0].timestamp) if readings else None

    return SensorHealthResponse(
        sensor_id=sensor_id,
        health_score=health_score,
        health_grade=health_grade,
        anomaly_rate=round(anomaly_count / max(total, 1) * 100, 1),
        failure_rate=round(failure_count / max(total, 1) * 100, 1),
        drift_magnitude=round(avg_drift, 2),
        reading_frequency=freq,
        total_readings=total,
        last_reading=last_ts,
        breakdown={
            "anomaly_penalty": round((anomaly_count / max(total, 1)) * 40, 1),
            "failure_penalty": round((failure_count / max(total, 1)) * 30, 1),
            "drift_penalty": round(min(avg_drift / 5, 20), 1),
            "data_density_bonus": round(min(total, 10), 1),
        },
    )


# ── Simulate Drift Endpoint ───────────────────────────────────────────────

@app.post(
    "/simulate_drift",
    response_model=DriftSimulationResponse,
    tags=["Advanced Analytics"],
    summary="Inject realistic sensor drift into readings",
)
def simulate_drift(
    sensor_id: str,
    drift_type: str = "offset",  # offset / humidity / random_walk
    magnitude: float = 15.0,
    db: Session = Depends(get_db),
):
    """
    Applies drift to the latest readings of a sensor and re-runs AI correction,
    demonstrating the self-correcting pipeline.
    """
    sensor = db.query(SensorRegistry).filter(
        SensorRegistry.sensor_id == sensor_id
    ).first()
    if not sensor:
        raise HTTPException(status_code=404, detail=f"Sensor '{sensor_id}' not found.")

    readings = (
        db.query(SensorReadings)
        .filter(SensorReadings.sensor_id == sensor_id)
        .order_by(SensorReadings.timestamp.desc())
        .limit(24)
        .all()
    )

    if not readings:
        raise HTTPException(status_code=404, detail="No readings to apply drift to.")

    affected = 0
    for i, reading in enumerate(readings):
        if not reading.pm2p5_raw:
            continue

        original_raw = reading.pm2p5_raw

        if drift_type == "offset":
            # Gradual increasing offset (simulates dust on optics)
            drift_amount = magnitude * (1 + i * 0.1)
            drifted = original_raw + drift_amount
        elif drift_type == "humidity":
            # Humidity-dependent scaling
            humidity_factor = 1 + (reading.humidity or 50) / 200 * (magnitude / 10)
            drifted = original_raw * humidity_factor
        elif drift_type == "random_walk":
            # Brownian motion
            drifted = original_raw + sum(random.gauss(0, magnitude / 5) for _ in range(i + 1))
        else:
            drifted = original_raw

        drifted = max(1, drifted)  # Can't be negative

        # Re-run AI correction on drifted data
        try:
            features = [
                drifted,
                reading.humidity or 50,
                reading.temperature or 25,
                1013.0,  # default pressure
                2.0,     # default wind
                30.0,    # default cloud
                drifted,
            ]
            corrected = model_predict(features)
        except Exception:
            corrected = original_raw  # Fallback

        reading.pm2p5_drifted = round(drifted, 2)
        reading.pm2p5_corrected = round(corrected, 2)
        reading.drift_type = drift_type
        affected += 1

    db.commit()

    return DriftSimulationResponse(
        sensor_id=sensor_id,
        readings_affected=affected,
        drift_type=drift_type,
        drift_magnitude=magnitude,
        message=f"Applied {drift_type} drift to {affected} readings. AI correction re-applied.",
    )
