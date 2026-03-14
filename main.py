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

import io
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Header, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import (
    SensorRegistry,
    SensorReadings,
    AlertConfig,
    AlertHistory,
    GeoFenceZone,
    init_db,
    get_db,
)
from model_utils import predict as model_predict, load_model
from azure_storage import initialize_containers, upload_reading_hot, archive_to_cold

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
    # Initialize Azure Containers
    initialize_containers()
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
    city:          str = Field("Delhi", example="Delhi")
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
    city:              str
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


# ── Schemas for new advanced features ──

class AlertConfigRequest(BaseModel):
    webhook_url: str = Field(..., example="https://hooks.slack.com/services/XXX")
    alert_type: str = Field("webhook", example="webhook")
    threshold_pm25: float = Field(150.0, example=150.0)
    label: str = Field("My Alert", example="Slack Critical Alert")

class AlertConfigResponse(BaseModel):
    id: int
    webhook_url: str
    alert_type: str
    threshold_pm25: float
    label: str
    is_active: int

class AlertHistoryOut(BaseModel):
    id: int
    sensor_id: str
    message: str
    severity: str
    pm25_value: Optional[float]
    delivered: int
    webhook_url: Optional[str]
    timestamp: str

class DispersionPoint(BaseModel):
    lat: float
    lon: float
    concentration: float  # predicted PM2.5

class DispersionResponse(BaseModel):
    sensor_id: str
    source_pm25: float
    wind_speed: float
    wind_direction: float  # degrees
    stability_class: str
    grid: List[DispersionPoint]
    affected_area_km2: float

class HealthRisk(BaseModel):
    risk_type: str
    risk_level: str  # low / moderate / high / very_high / critical
    score: float  # 0-100
    description: str
    icon: str

class HealthImpactResponse(BaseModel):
    sensor_id: str
    location_name: str
    current_pm25: float
    aqi_category: str
    exposure_duration_safe_hours: float
    health_risks: List[HealthRisk]
    vulnerable_groups: List[str]
    recommendations: List[str]
    estimated_population_exposed: int  # within 1km radius estimate

class CityHealthImpactResponse(BaseModel):
    city_name: str
    overall_aqi: float
    estimated_population_affected: int
    health_risks: List[HealthRisk]
    zone_breakdown: List[Dict[str, Any]]
    advisory_level: str
    recommendations: List[str]

class GeoFenceRequest(BaseModel):
    name: str = Field(..., example="Delhi Public School")
    zone_type: str = Field("school", example="school")
    center_lat: float = Field(..., example=28.6139)
    center_lon: float = Field(..., example=77.2090)
    radius_m: float = Field(500.0, example=500.0)
    pm25_threshold: float = Field(55.0, example=55.0)

class GeoFenceOut(BaseModel):
    id: int
    name: str
    zone_type: str
    center_lat: float
    center_lon: float
    radius_m: float
    pm25_threshold: float
    is_active: int
    current_pm25: Optional[float] = None
    is_breached: Optional[bool] = None

class CityComparisonEntry(BaseModel):
    city: str
    aqi: float
    aqi_category: str
    sensor_count: int
    active_sensors: int
    avg_health_score: float
    health_grade: str
    anomaly_rate: float
    trend: str
    coverage_pct: float

class CityComparisonResponse(BaseModel):
    cities: List[CityComparisonEntry]
    best_city: str
    worst_city: str
    generated_at: str


# ─────────────────────────────────────────────
# Background Task: Alert (Enhanced with Webhooks)
# ─────────────────────────────────────────────

def alert_high_pollution(sensor_id: str, pm2p5_corrected: float, is_anomaly: int):
    """
    Background task triggered when a reading is flagged as anomalous or
    has critically high PM2.5 levels. Fires configured webhooks and logs alerts.
    """
    import requests as req_lib
    reasons = []
    if is_anomaly == 1:
        reasons.append("ANOMALY DETECTED")
    if pm2p5_corrected > 150:
        reasons.append(f"HIGH PM2.5 ({pm2p5_corrected:.1f} µg/m³ > 150 threshold)")

    reason_str = " | ".join(reasons)
    severity = "critical" if pm2p5_corrected > 200 else "high" if pm2p5_corrected > 150 else "medium"
    message = f"🚨 ALERT: {reason_str} at Sensor [{sensor_id}]! Immediate investigation required."
    
    logger.critical(message)
    
    # Fire all active webhooks that match threshold
    db = next(get_db())
    try:
        configs = db.query(AlertConfig).filter(
            AlertConfig.is_active == 1,
            AlertConfig.threshold_pm25 <= pm2p5_corrected,
        ).all()
        
        for config in configs:
            delivered = 0
            try:
                payload = {
                    "text": message,
                    "sensor_id": sensor_id,
                    "pm25": pm2p5_corrected,
                    "severity": severity,
                    "timestamp": datetime.now().isoformat(),
                    "source": "Vayu-Rakshak Alert System",
                }
                resp = req_lib.post(config.webhook_url, json=payload, timeout=5)
                delivered = 1 if resp.status_code < 400 else -1
            except Exception as e:
                logger.error(f"Webhook delivery failed: {e}")
                delivered = -1
            
            # Log alert
            alert_log = AlertHistory(
                sensor_id=sensor_id,
                message=message,
                severity=severity,
                pm25_value=pm2p5_corrected,
                delivered=delivered,
                webhook_url=config.webhook_url,
            )
            db.add(alert_log)
        
        # Also log even without webhooks
        if not configs:
            alert_log = AlertHistory(
                sensor_id=sensor_id,
                message=message,
                severity=severity,
                pm25_value=pm2p5_corrected,
                delivered=0,
            )
            db.add(alert_log)
        
        db.commit()
    except Exception as e:
        logger.error(f"Alert processing error: {e}")
    finally:
        db.close()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

# ── Health ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "service": "Vayu-Rakshak API"}

# ── Archival ──────────────────────────────────────────────────────────────

@app.post("/archive", tags=["System"], summary="Archive older blobs from Hot to Cold Azure Storage")
def archive_blobs(limit: int = 100):
    archived_count = archive_to_cold(limit)
    return {"status": "success", "archived_count": archived_count}


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
        city=payload.city,
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
            city=s.city,
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

    # ── Async upload to Azure Hot Storage ────────────────────────────────
    background_tasks.add_task(upload_reading_hot, payload.dict())

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
def city_aqi(city: Optional[str] = None, db: Session = Depends(get_db)):
    """Compute city-wide AQI using IDW-weighted aggregation of all sensors."""
    query = db.query(SensorRegistry)
    if city:
        query = query.filter(SensorRegistry.city == city)
    
    sensors = query.all()
    if not sensors:
        raise HTTPException(status_code=404, detail=f"No sensors registered for city: {city or 'any'}")

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
        city_name=city if city else "All Sensors",
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


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 1: ALERT WEBHOOK SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/alerts/configure",
    response_model=AlertConfigResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Alert System"],
    summary="Register a webhook for pollution alerts",
)
def configure_alert(payload: AlertConfigRequest, db: Session = Depends(get_db)):
    config = AlertConfig(
        webhook_url=payload.webhook_url,
        alert_type=payload.alert_type,
        threshold_pm25=payload.threshold_pm25,
        label=payload.label,
    )
    db.add(config)
    db.commit()
    db.refresh(config)
    return AlertConfigResponse(
        id=config.id, webhook_url=config.webhook_url, alert_type=config.alert_type,
        threshold_pm25=config.threshold_pm25, label=config.label, is_active=config.is_active,
    )


@app.get("/alerts/configs", response_model=List[AlertConfigResponse], tags=["Alert System"])
def list_alert_configs(db: Session = Depends(get_db)):
    configs = db.query(AlertConfig).all()
    return [
        AlertConfigResponse(
            id=c.id, webhook_url=c.webhook_url, alert_type=c.alert_type,
            threshold_pm25=c.threshold_pm25, label=c.label, is_active=c.is_active,
        ) for c in configs
    ]


@app.get("/alerts/history", response_model=List[AlertHistoryOut], tags=["Alert System"])
def alert_history(limit: int = 50, db: Session = Depends(get_db)):
    alerts = db.query(AlertHistory).order_by(AlertHistory.timestamp.desc()).limit(limit).all()
    return [
        AlertHistoryOut(
            id=a.id, sensor_id=a.sensor_id, message=a.message, severity=a.severity,
            pm25_value=a.pm25_value, delivered=a.delivered, webhook_url=a.webhook_url,
            timestamp=str(a.timestamp),
        ) for a in alerts
    ]


@app.delete("/alerts/configure/{config_id}", tags=["Alert System"])
def delete_alert_config(config_id: int, db: Session = Depends(get_db)):
    config = db.query(AlertConfig).filter(AlertConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Alert config not found.")
    db.delete(config)
    db.commit()
    return {"status": "deleted", "id": config_id}


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 2: GAUSSIAN PLUME DISPERSION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def _gaussian_plume(x: float, y: float, Q: float, u: float, H: float,
                     sigma_y: float, sigma_z: float) -> float:
    """Gaussian plume concentration at point (x, y) from a source."""
    if x <= 0 or u <= 0:
        return 0.0
    C = (Q / (2 * math.pi * u * sigma_y * sigma_z)) * \
        math.exp(-0.5 * (y / sigma_y) ** 2) * \
        (math.exp(-0.5 * ((0 - H) / sigma_z) ** 2) + math.exp(-0.5 * ((0 + H) / sigma_z) ** 2))
    return max(0, C)


def _stability_params(stability_class: str, x_km: float):
    """Pasquill-Gifford dispersion coefficients."""
    x_m = max(x_km * 1000, 100)
    params = {
        "A": (0.22, 0.20, 0.0001, 0.5),
        "B": (0.16, 0.12, 0.0001, 0.5),
        "C": (0.11, 0.08, 0.0001, 0.5),
        "D": (0.08, 0.06, 0.0003, 0.5),
        "E": (0.06, 0.03, 0.0003, 0.5),
        "F": (0.04, 0.016, 0.0003, 0.5),
    }
    a, b, c, d = params.get(stability_class, params["D"])
    sigma_y = a * x_m / (1 + c * x_m) ** d
    sigma_z = b * x_m / (1 + c * x_m) ** d
    return max(sigma_y, 1), max(sigma_z, 1)


@app.get(
    "/dispersion/{sensor_id}",
    response_model=DispersionResponse,
    tags=["Advanced Analytics"],
    summary="Gaussian plume dispersion prediction",
)
def dispersion_model(sensor_id: str, wind_speed: float = 3.0,
                      wind_direction: float = 225.0,
                      db: Session = Depends(get_db)):
    """
    Computes a Gaussian plume model to predict how pollution spreads
    from a sensor's location based on wind speed and direction.
    """
    sensor = db.query(SensorRegistry).filter(
        SensorRegistry.sensor_id == sensor_id
    ).first()
    if not sensor:
        raise HTTPException(status_code=404, detail=f"Sensor '{sensor_id}' not found.")

    latest = (
        db.query(SensorReadings)
        .filter(SensorReadings.sensor_id == sensor_id)
        .order_by(SensorReadings.timestamp.desc())
        .first()
    )
    if not latest:
        raise HTTPException(status_code=404, detail="No readings for this sensor.")

    source_pm25 = latest.pm2p5_corrected or 50.0
    Q = source_pm25 * 10  # Emission rate proxy

    # Determine atmospheric stability from temperature/humidity
    temp = latest.temperature or 25
    humidity = latest.humidity or 50
    if temp > 30 and humidity < 40:
        stability = "A"
    elif temp > 25:
        stability = "B"
    elif temp > 15:
        stability = "D"
    else:
        stability = "E"

    # Wind direction in radians (meteorological: from)
    wind_rad = math.radians(wind_direction)

    # Generate grid points (5km x 5km around sensor)
    grid = []
    total_area = 0
    for dx_idx in range(-5, 6):
        for dy_idx in range(-5, 6):
            dx_km = dx_idx * 0.5  # 500m steps
            dy_km = dy_idx * 0.5

            # Rotate by wind direction
            x_wind = dx_km * math.cos(wind_rad) + dy_km * math.sin(wind_rad)
            y_wind = -dx_km * math.sin(wind_rad) + dy_km * math.cos(wind_rad)

            if x_wind > 0:
                sigma_y, sigma_z = _stability_params(stability, x_wind)
                conc = _gaussian_plume(x_wind * 1000, y_wind * 1000, Q,
                                       max(wind_speed, 0.5), 10, sigma_y, sigma_z)
            else:
                conc = source_pm25 * 0.1 * max(0, 1 - math.sqrt(dx_km**2 + dy_km**2) / 2)

            # Convert km offset to lat/lon
            lat_offset = dy_km / 111.0
            lon_offset = dx_km / (111.0 * math.cos(math.radians(sensor.lat)))

            if conc > 1:
                total_area += 0.25  # 0.5km × 0.5km = 0.25 km²
                grid.append(DispersionPoint(
                    lat=round(sensor.lat + lat_offset, 6),
                    lon=round(sensor.long + lon_offset, 6),
                    concentration=round(min(conc, source_pm25 * 2), 2),
                ))

    return DispersionResponse(
        sensor_id=sensor_id,
        source_pm25=round(source_pm25, 1),
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        stability_class=stability,
        grid=grid,
        affected_area_km2=round(total_area, 2),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 3: HEALTH IMPACT CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_health_risks(pm25: float) -> List[HealthRisk]:
    """WHO exposure-response function for PM2.5 health impacts."""
    risks = []

    # Respiratory risk
    resp_score = min(100, pm25 * 0.6)
    if resp_score > 75: resp_level = "critical"
    elif resp_score > 50: resp_level = "very_high"
    elif resp_score > 30: resp_level = "high"
    elif resp_score > 15: resp_level = "moderate"
    else: resp_level = "low"
    risks.append(HealthRisk(
        risk_type="Respiratory",
        risk_level=resp_level,
        score=round(resp_score, 1),
        description=f"Asthma, bronchitis, and COPD exacerbation risk at {pm25:.0f} µg/m³",
        icon="🫁",
    ))

    # Cardiovascular risk
    cardio_score = min(100, pm25 * 0.45)
    if cardio_score > 60: cardio_level = "critical"
    elif cardio_score > 40: cardio_level = "high"
    elif cardio_score > 20: cardio_level = "moderate"
    else: cardio_level = "low"
    risks.append(HealthRisk(
        risk_type="Cardiovascular",
        risk_level=cardio_level,
        score=round(cardio_score, 1),
        description="Heart attack, stroke, and arrhythmia risk from fine particle inhalation",
        icon="❤️",
    ))

    # Neurological risk
    neuro_score = min(100, pm25 * 0.25)
    if neuro_score > 40: neuro_level = "high"
    elif neuro_score > 20: neuro_level = "moderate"
    else: neuro_level = "low"
    risks.append(HealthRisk(
        risk_type="Neurological",
        risk_level=neuro_level,
        score=round(neuro_score, 1),
        description="Cognitive decline, headaches, and neuroinflammation from prolonged exposure",
        icon="🧠",
    ))

    # Immune system
    immune_score = min(100, pm25 * 0.3)
    if immune_score > 45: immune_level = "high"
    elif immune_score > 22: immune_level = "moderate"
    else: immune_level = "low"
    risks.append(HealthRisk(
        risk_type="Immune System",
        risk_level=immune_level,
        score=round(immune_score, 1),
        description="Weakened immune response increasing susceptibility to infections",
        icon="🛡️",
    ))

    # Cancer risk (long-term)
    cancer_score = min(100, pm25 * 0.15)
    if cancer_score > 30: cancer_level = "high"
    elif cancer_score > 15: cancer_level = "moderate"
    else: cancer_level = "low"
    risks.append(HealthRisk(
        risk_type="Cancer (Long-term)",
        risk_level=cancer_level,
        score=round(cancer_score, 1),
        description="IARC Group 1 carcinogen — long-term PM2.5 exposure linked to lung cancer",
        icon="⚠️",
    ))

    return risks


def _safe_exposure_hours(pm25: float) -> float:
    """Estimate safe outdoor exposure duration based on PM2.5."""
    if pm25 < 12: return 24.0
    if pm25 < 35: return 12.0
    if pm25 < 55: return 4.0
    if pm25 < 150: return 1.5
    if pm25 < 250: return 0.5
    return 0.0


@app.get(
    "/health_impact/{sensor_id}",
    response_model=HealthImpactResponse,
    tags=["Health Impact"],
    summary="Health impact assessment for a sensor location",
)
def health_impact_sensor(sensor_id: str, db: Session = Depends(get_db)):
    sensor = db.query(SensorRegistry).filter(
        SensorRegistry.sensor_id == sensor_id
    ).first()
    if not sensor:
        raise HTTPException(status_code=404, detail=f"Sensor '{sensor_id}' not found.")

    latest = (
        db.query(SensorReadings)
        .filter(SensorReadings.sensor_id == sensor_id)
        .order_by(SensorReadings.timestamp.desc())
        .first()
    )
    pm25 = latest.pm2p5_corrected if latest and latest.pm2p5_corrected else 50.0

    risks = _compute_health_risks(pm25)

    vulnerable = []
    if pm25 > 35:
        vulnerable.extend(["Children (< 14 years)", "Elderly (> 65 years)", "Pregnant women"])
    if pm25 > 55:
        vulnerable.extend(["Asthma/COPD patients", "Heart disease patients"])
    if pm25 > 100:
        vulnerable.extend(["All outdoor workers", "Athletes"])
    if pm25 > 150:
        vulnerable.append("General population at risk")

    recs = []
    if pm25 > 150:
        recs.append("🚨 Stay indoors — use air purifiers with HEPA filters")
        recs.append("🏥 People with respiratory/heart conditions should seek medical advice")
    if pm25 > 100:
        recs.append("😷 Wear N95/KN95 masks outdoors")
        recs.append("🏫 Schools should cancel outdoor activities")
    if pm25 > 55:
        recs.append("🚫 Avoid prolonged outdoor exercise")
        recs.append("🪟 Keep windows and doors closed")
    if pm25 > 35:
        recs.append("💧 Stay hydrated and consume antioxidant-rich foods")
    recs.append("📱 Monitor real-time AQI updates on Vayu-Rakshak")

    # Rough population density estimate per km² (urban India avg ~11,000/km²)
    pop_density = 11000
    coverage_km2 = math.pi * 1 ** 2  # 1km sensor radius
    estimated_pop = int(pop_density * coverage_km2)

    return HealthImpactResponse(
        sensor_id=sensor_id,
        location_name=sensor.location_name,
        current_pm25=round(pm25, 1),
        aqi_category=_aqi_category(pm25),
        exposure_duration_safe_hours=_safe_exposure_hours(pm25),
        health_risks=risks,
        vulnerable_groups=vulnerable,
        recommendations=recs,
        estimated_population_exposed=estimated_pop,
    )


@app.get(
    "/city_health_impact",
    response_model=CityHealthImpactResponse,
    tags=["Health Impact"],
    summary="City-wide health impact assessment",
)
def city_health_impact(city: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(SensorRegistry)
    if city:
        query = query.filter(SensorRegistry.city == city)
    sensors = query.all()
    if not sensors:
        raise HTTPException(status_code=404, detail="No sensors found.")

    pm25_values = []
    zone_breakdown = []
    for s in sensors:
        latest = (
            db.query(SensorReadings)
            .filter(SensorReadings.sensor_id == s.sensor_id)
            .order_by(SensorReadings.timestamp.desc())
            .first()
        )
        if latest and latest.pm2p5_corrected:
            pm25_values.append(latest.pm2p5_corrected)
            zone_breakdown.append({
                "sensor_id": s.sensor_id,
                "location": s.location_name,
                "pm25": round(latest.pm2p5_corrected, 1),
                "health_risk_score": round(min(100, latest.pm2p5_corrected * 0.6), 1),
            })

    overall = statistics.mean(pm25_values) if pm25_values else 0
    risks = _compute_health_risks(overall)

    # Population estimate
    pop_per_sensor = 11000 * math.pi  # ~34,558 per sensor coverage area
    total_pop = int(pop_per_sensor * len(pm25_values))

    if overall > 150: advisory = "RED — Public Health Emergency"
    elif overall > 100: advisory = "ORANGE — Health Alert"
    elif overall > 55: advisory = "YELLOW — Caution Advisory"
    elif overall > 35: advisory = "GREEN — Moderate"
    else: advisory = "BLUE — Good Air Quality"

    recs = []
    if overall > 150:
        recs.append("Issue city-wide health emergency advisory")
        recs.append("Deploy emergency air purification trucks to hotspots")
        recs.append("Enforce complete construction ban")
    if overall > 100:
        recs.append("Activate odd-even traffic restrictions")
        recs.append("Close outdoor markets and restrict industrial operations")
    if overall > 55:
        recs.append("Issue advisory for schools to suspend outdoor activities")
        recs.append("Increase water sprinkling on roads")
    recs.append("Continue 24/7 AI-powered monitoring with Vayu-Rakshak")

    return CityHealthImpactResponse(
        city_name=city or "All Cities",
        overall_aqi=round(overall, 1),
        estimated_population_affected=total_pop,
        health_risks=risks,
        zone_breakdown=sorted(zone_breakdown, key=lambda x: x["pm25"], reverse=True)[:20],
        advisory_level=advisory,
        recommendations=recs,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 4: PDF REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/report/{city}",
    tags=["Reports"],
    summary="Generate a professional PDF air quality report",
)
def generate_report(city: str, db: Session = Depends(get_db)):
    from fastapi.responses import StreamingResponse
    from report_generator import generate_city_report

    try:
        pdf_bytes = generate_city_report(city, db)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="vayu_rakshak_{city}_{datetime.now().strftime("%Y%m%d")}.pdf"'
            },
        )
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 5: GEOFENCING & SMART ALERT ZONES
# ═══════════════════════════════════════════════════════════════════════════════

def _haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


@app.post(
    "/geofence",
    response_model=GeoFenceOut,
    status_code=status.HTTP_201_CREATED,
    tags=["Geofencing"],
    summary="Create a geofence zone around a sensitive location",
)
def create_geofence(payload: GeoFenceRequest, db: Session = Depends(get_db)):
    zone = GeoFenceZone(
        name=payload.name,
        zone_type=payload.zone_type,
        center_lat=payload.center_lat,
        center_lon=payload.center_lon,
        radius_m=payload.radius_m,
        pm25_threshold=payload.pm25_threshold,
    )
    db.add(zone)
    db.commit()
    db.refresh(zone)
    return GeoFenceOut(
        id=zone.id, name=zone.name, zone_type=zone.zone_type,
        center_lat=zone.center_lat, center_lon=zone.center_lon,
        radius_m=zone.radius_m, pm25_threshold=zone.pm25_threshold,
        is_active=zone.is_active,
    )


@app.get("/geofences", response_model=List[GeoFenceOut], tags=["Geofencing"],
         summary="List all geofence zones with breach status")
def list_geofences(db: Session = Depends(get_db)):
    zones = db.query(GeoFenceZone).filter(GeoFenceZone.is_active == 1).all()
    result = []
    for zone in zones:
        # Find nearest sensor and check breach
        sensors = db.query(SensorRegistry).all()
        current_pm25 = None
        is_breached = False

        for sensor in sensors:
            dist = _haversine_km(zone.center_lat, zone.center_lon, sensor.lat, sensor.long)
            if dist <= zone.radius_m / 1000:
                latest = (
                    db.query(SensorReadings)
                    .filter(SensorReadings.sensor_id == sensor.sensor_id)
                    .order_by(SensorReadings.timestamp.desc())
                    .first()
                )
                if latest and latest.pm2p5_corrected:
                    if current_pm25 is None or latest.pm2p5_corrected > current_pm25:
                        current_pm25 = latest.pm2p5_corrected
                    if latest.pm2p5_corrected > zone.pm25_threshold:
                        is_breached = True

        result.append(GeoFenceOut(
            id=zone.id, name=zone.name, zone_type=zone.zone_type,
            center_lat=zone.center_lat, center_lon=zone.center_lon,
            radius_m=zone.radius_m, pm25_threshold=zone.pm25_threshold,
            is_active=zone.is_active,
            current_pm25=round(current_pm25, 1) if current_pm25 else None,
            is_breached=is_breached,
        ))
    return result


@app.get("/geofence_alerts", tags=["Geofencing"],
         summary="Get currently breached geofence zones")
def geofence_alerts(db: Session = Depends(get_db)):
    all_zones = list_geofences(db)
    breached = [z for z in all_zones if z.is_breached]
    return {
        "total_zones": len(all_zones),
        "breached_zones": len(breached),
        "alerts": [z.dict() for z in breached],
    }


@app.delete("/geofence/{zone_id}", tags=["Geofencing"])
def delete_geofence(zone_id: int, db: Session = Depends(get_db)):
    zone = db.query(GeoFenceZone).filter(GeoFenceZone.id == zone_id).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Geofence zone not found.")
    db.delete(zone)
    db.commit()
    return {"status": "deleted", "id": zone_id}


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 6: MULTI-CITY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/city_comparison",
    response_model=CityComparisonResponse,
    tags=["Advanced Analytics"],
    summary="Compare AQI, health, and trends across all cities",
)
def city_comparison(db: Session = Depends(get_db)):
    # Get all distinct cities
    cities_raw = db.query(SensorRegistry.city).distinct().all()
    cities_list = [c[0] for c in cities_raw if c[0]]

    entries = []
    for city_name in cities_list:
        sensors = db.query(SensorRegistry).filter(SensorRegistry.city == city_name).all()
        sensor_count = len(sensors)

        pm25_values = []
        health_scores = []
        anomaly_total = 0
        total_readings = 0

        for sensor in sensors:
            readings = (
                db.query(SensorReadings)
                .filter(SensorReadings.sensor_id == sensor.sensor_id)
                .order_by(SensorReadings.timestamp.desc())
                .limit(50)
                .all()
            )
            if readings:
                latest_pm25 = readings[0].pm2p5_corrected
                if latest_pm25:
                    pm25_values.append(latest_pm25)

                total = len(readings)
                anom = sum(1 for r in readings if r.is_anomaly == 1)
                fail = sum(1 for r in readings if r.is_failure == 1)
                drift_m = [abs(r.pm2p5_raw - r.pm2p5_corrected) for r in readings
                          if r.pm2p5_raw and r.pm2p5_corrected]
                avg_d = statistics.mean(drift_m) if drift_m else 0

                score = 100 - (anom/max(total,1))*40 - (fail/max(total,1))*30 - min(avg_d/5, 20)
                health_scores.append(max(0, min(100, score)))
                anomaly_total += anom
                total_readings += total

        if not pm25_values:
            continue

        city_aqi_val = statistics.mean(pm25_values)
        avg_health = statistics.mean(health_scores) if health_scores else 50

        if avg_health >= 90: grade = "A"
        elif avg_health >= 75: grade = "B"
        elif avg_health >= 60: grade = "C"
        elif avg_health >= 40: grade = "D"
        else: grade = "F"

        # Trend
        recent = db.query(func.avg(SensorReadings.pm2p5_corrected)).join(
            SensorRegistry, SensorReadings.sensor_id == SensorRegistry.sensor_id
        ).filter(
            SensorRegistry.city == city_name,
            SensorReadings.timestamp > datetime.now() - timedelta(hours=2),
        ).scalar() or city_aqi_val

        older = db.query(func.avg(SensorReadings.pm2p5_corrected)).join(
            SensorRegistry, SensorReadings.sensor_id == SensorRegistry.sensor_id
        ).filter(
            SensorRegistry.city == city_name,
            SensorReadings.timestamp > datetime.now() - timedelta(hours=4),
            SensorReadings.timestamp <= datetime.now() - timedelta(hours=2),
        ).scalar() or city_aqi_val

        if recent > older * 1.05: trend = "rising"
        elif recent < older * 0.95: trend = "falling"
        else: trend = "stable"

        entries.append(CityComparisonEntry(
            city=city_name,
            aqi=round(city_aqi_val, 1),
            aqi_category=_aqi_category(city_aqi_val),
            sensor_count=sensor_count,
            active_sensors=len(pm25_values),
            avg_health_score=round(avg_health, 1),
            health_grade=grade,
            anomaly_rate=round(anomaly_total / max(total_readings, 1) * 100, 1),
            trend=trend,
            coverage_pct=round(len(pm25_values) / max(sensor_count, 1) * 100, 1),
        ))

    entries.sort(key=lambda x: x.aqi)
    best = entries[0].city if entries else "N/A"
    worst = entries[-1].city if entries else "N/A"

    return CityComparisonResponse(
        cities=entries,
        best_city=best,
        worst_city=worst,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

