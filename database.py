"""
database.py — SQLAlchemy models and database initialization for the
Vayu-Rakshak Air Quality Monitoring System.
"""

import uuid
from datetime import datetime, date
from typing import Optional

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Date, ForeignKey, create_engine, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import os

# ─────────────────────────────────────────────
# Engine & Session
# ─────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./air_quality.db")

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

class SensorRegistry(Base):
    """
    Stores metadata for each registered IoT sensor in the network.
    An API key is auto-generated on registration and used to authenticate
    data ingestion requests.
    """
    __tablename__ = "sensor_registry"

    sensor_id         = Column(String, primary_key=True, index=True)
    location_name     = Column(String, nullable=False)
    city              = Column(String, nullable=False, default="Delhi")
    lat               = Column(Float, nullable=False)
    long              = Column(Float, nullable=False)
    installation_date = Column(Date, default=date.today)
    api_key           = Column(String, unique=True, nullable=False, default=lambda: str(uuid.uuid4()))

    # Relationship
    readings = relationship("SensorReadings", back_populates="sensor", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<SensorRegistry sensor_id={self.sensor_id} location={self.location_name}>"


class SensorReadings(Base):
    """
    Stores individual time-series readings from each sensor.
    pm2p5_corrected is the AI-corrected value; is_anomaly and is_failure
    flags are set by the upstream data pipeline.
    """
    __tablename__ = "sensor_readings"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    sensor_id        = Column(String, ForeignKey("sensor_registry.sensor_id"), nullable=False, index=True)
    timestamp        = Column(DateTime, nullable=False, default=datetime.utcnow)
    temperature      = Column(Float)
    humidity         = Column(Float)
    pm2p5_raw        = Column(Float)
    pm2p5_corrected  = Column(Float)
    pm2p5_drifted    = Column(Float, nullable=True)   # Drifted raw value before AI correction
    drift_type       = Column(String, nullable=True, default="none")  # offset/humidity/random_walk/none
    is_anomaly       = Column(Integer, default=0)   # 0 = normal, 1 = anomaly
    is_failure       = Column(Integer, default=0)   # 0 = normal, 1 = sensor failure

    # Relationship
    sensor = relationship("SensorRegistry", back_populates="readings")

    def __repr__(self):
        return (
            f"<SensorReadings id={self.id} sensor_id={self.sensor_id} "
            f"timestamp={self.timestamp} pm2p5_corrected={self.pm2p5_corrected}>"
        )


# ─────────────────────────────────────────────
# DB Lifecycle Helpers
# ─────────────────────────────────────────────

def init_db():
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)
    # Migrate existing tables: add new columns if missing
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE sensor_readings ADD COLUMN pm2p5_drifted FLOAT"))
        except Exception:
            pass  # Column already exists
        try:
            conn.execute(text("ALTER TABLE sensor_readings ADD COLUMN drift_type VARCHAR"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE sensor_registry ADD COLUMN city VARCHAR DEFAULT 'Delhi'"))
        except Exception:
            pass
        conn.commit()
    print("✅ Database initialized — tables created (or already exist).")


def get_db():
    """
    Dependency-injection helper for FastAPI route handlers.
    Yields a SQLAlchemy session and ensures it closes after the request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
