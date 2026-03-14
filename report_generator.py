"""
report_generator.py — Automated PDF Report Generator for Vayu-Rakshak.

Generates professional air quality reports with:
  - Executive summary with city AQI overview
  - Sensor network health grid
  - PM2.5 trend chart (matplotlib → embedded PNG)
  - Top hotspots / cleanest zones table
  - AI-powered recommendations

Uses: matplotlib for charts, reportlab for PDF assembly.
"""

import io
import os
import statistics
import math
from datetime import datetime, timedelta
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from sqlalchemy.orm import Session
from database import SensorRegistry, SensorReadings


# ─────────────────────────────────────────────
# Chart generation helpers
# ─────────────────────────────────────────────

def _generate_aqi_gauge_chart(aqi_value: float, category: str) -> bytes:
    """Generate a semi-circular AQI gauge chart."""
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={"projection": "polar"})
    
    # AQI mapped to 0-180 degrees (π radians)
    max_aqi = 500
    angle = min(aqi_value / max_aqi, 1.0) * np.pi
    
    # Background segments
    segments = [
        (0, 12/500*np.pi, "#3fb950", "Good"),
        (12/500*np.pi, 35/500*np.pi, "#e3b341", "Moderate"),
        (35/500*np.pi, 55/500*np.pi, "#f0883e", "USG"),
        (55/500*np.pi, 150/500*np.pi, "#ff7b72", "Unhealthy"),
        (150/500*np.pi, 250/500*np.pi, "#ff453a", "Very Unhealthy"),
        (250/500*np.pi, np.pi, "#da3633", "Hazardous"),
    ]
    
    for start, end, color, _ in segments:
        theta = np.linspace(start, end, 50)
        ax.fill_between(theta, 0.6, 1.0, color=color, alpha=0.3)
    
    # Needle
    ax.plot([angle, angle], [0, 0.95], color="white", linewidth=3)
    ax.plot(angle, 0.95, "o", color="white", markersize=8)
    
    ax.set_ylim(0, 1.2)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    
    ax.text(np.pi/2, 0.3, f"{aqi_value:.0f}", ha="center", va="center",
            fontsize=24, fontweight="bold", color="white")
    ax.text(np.pi/2, 0.05, category, ha="center", va="center",
            fontsize=10, color="#8b949e")
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _generate_trend_chart(readings: list) -> bytes:
    """Generate PM2.5 trend line chart."""
    fig, ax = plt.subplots(figsize=(7, 3))
    
    timestamps = []
    pm25_raw = []
    pm25_corrected = []
    
    for r in sorted(readings, key=lambda x: x.timestamp):
        if r.timestamp and r.pm2p5_corrected:
            timestamps.append(r.timestamp)
            pm25_raw.append(r.pm2p5_raw or 0)
            pm25_corrected.append(r.pm2p5_corrected)
    
    if timestamps:
        ax.plot(timestamps, pm25_raw, color="#8b949e", linewidth=1, alpha=0.5, 
                label="Raw", linestyle="--")
        ax.fill_between(timestamps, pm25_corrected, alpha=0.15, color="#58a6ff")
        ax.plot(timestamps, pm25_corrected, color="#58a6ff", linewidth=2, 
                label="AI Corrected")
        ax.axhline(y=150, color="#ff7b72", linestyle="--", linewidth=1, alpha=0.7,
                   label="Hazardous Threshold")
    
    ax.set_facecolor("#161b22")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.set_xlabel("Time")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3",
              fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#30363d")
    ax.spines["bottom"].set_color("#30363d")
    
    if timestamps:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        fig.autofmt_xdate(rotation=25)
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _generate_sensor_health_bar(sensors_health: list) -> bytes:
    """Generate horizontal bar chart of sensor health scores."""
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(sensors_health) * 0.35)))
    
    names = [s["name"][:20] for s in sensors_health]
    scores = [s["score"] for s in sensors_health]
    bar_colors = []
    for s in scores:
        if s >= 90: bar_colors.append("#3fb950")
        elif s >= 75: bar_colors.append("#58a6ff")
        elif s >= 60: bar_colors.append("#e3b341")
        elif s >= 40: bar_colors.append("#f0883e")
        else: bar_colors.append("#ff453a")
    
    bars = ax.barh(names, scores, color=bar_colors, height=0.6, edgecolor="none")
    
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{score:.0f}", va="center", color="#e6edf3", fontsize=8)
    
    ax.set_xlim(0, 110)
    ax.set_facecolor("#161b22")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_xlabel("Health Score", color="#8b949e", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#30363d")
    ax.spines["bottom"].set_color("#30363d")
    ax.invert_yaxis()
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────
# PDF Report Generator
# ─────────────────────────────────────────────

def _aqi_category(pm25: float) -> str:
    if pm25 < 12:  return "Good"
    if pm25 < 35:  return "Moderate"
    if pm25 < 55:  return "Unhealthy for Sensitive Groups"
    if pm25 < 150: return "Unhealthy"
    if pm25 < 250: return "Very Unhealthy"
    return "Hazardous"


def generate_city_report(city: str, db: Session) -> bytes:
    """
    Generate a comprehensive PDF report for a city.
    Returns the PDF as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=50, leftMargin=50,
        topMargin=50, bottomMargin=50,
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"],
        fontSize=26, textColor=colors.HexColor("#58a6ff"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "CustomSubtitle", parent=styles["Normal"],
        fontSize=12, textColor=colors.HexColor("#8b949e"),
        spaceAfter=20,
    )
    heading_style = ParagraphStyle(
        "CustomHeading", parent=styles["Heading2"],
        fontSize=16, textColor=colors.HexColor("#58a6ff"),
        spaceBefore=20, spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "CustomBody", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#333333"),
        spaceAfter=8, leading=14,
    )
    highlight_style = ParagraphStyle(
        "Highlight", parent=styles["Normal"],
        fontSize=11, textColor=colors.HexColor("#0d1117"),
        backColor=colors.HexColor("#e6edf3"),
        borderPadding=8, spaceAfter=10,
    )
    
    elements = []
    
    # ── Title Page ──
    elements.append(Spacer(1, 80))
    elements.append(Paragraph("🌿 Vayu-Rakshak", title_style))
    elements.append(Paragraph("Air Quality Intelligence Report", ParagraphStyle(
        "BigSub", parent=styles["Normal"], fontSize=18,
        textColor=colors.HexColor("#3fb950"), spaceAfter=10,
    )))
    elements.append(Paragraph(
        f"City: <b>{city}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        subtitle_style
    ))
    elements.append(HRFlowable(
        width="100%", thickness=1, color=colors.HexColor("#30363d"),
        spaceAfter=20,
    ))
    
    # ── Gather data ──
    query = db.query(SensorRegistry)
    if city and city.lower() != "all":
        query = query.filter(SensorRegistry.city == city)
    sensors = query.all()
    
    all_readings = []
    sensor_latest = {}
    sensors_health_data = []
    
    for sensor in sensors:
        readings = (
            db.query(SensorReadings)
            .filter(SensorReadings.sensor_id == sensor.sensor_id)
            .order_by(SensorReadings.timestamp.desc())
            .limit(100)
            .all()
        )
        all_readings.extend(readings)
        if readings:
            sensor_latest[sensor.sensor_id] = {
                "pm25": readings[0].pm2p5_corrected or 0,
                "location": sensor.location_name,
                "timestamp": str(readings[0].timestamp),
            }
            # Health score
            total = len(readings)
            anomaly_count = sum(1 for r in readings if r.is_anomaly == 1)
            failure_count = sum(1 for r in readings if r.is_failure == 1)
            drift_mags = [abs(r.pm2p5_raw - r.pm2p5_corrected) 
                         for r in readings if r.pm2p5_raw and r.pm2p5_corrected]
            avg_drift = statistics.mean(drift_mags) if drift_mags else 0
            score = 100.0
            score -= (anomaly_count / max(total, 1)) * 40
            score -= (failure_count / max(total, 1)) * 30
            score -= min(avg_drift / 5, 20)
            score = max(0, min(100, score))
            sensors_health_data.append({
                "name": f"{sensor.sensor_id} ({sensor.location_name[:15]})",
                "score": round(score, 1),
            })
    
    # Overall AQI
    pm25_values = [v["pm25"] for v in sensor_latest.values() if v["pm25"]]
    overall_aqi = statistics.mean(pm25_values) if pm25_values else 0
    aqi_cat = _aqi_category(overall_aqi)
    
    # ── Executive Summary ──
    elements.append(Paragraph("📊 Executive Summary", heading_style))
    
    summary_data = [
        ["Metric", "Value"],
        ["City", city],
        ["Overall AQI (PM2.5)", f"{overall_aqi:.1f} µg/m³"],
        ["AQI Category", aqi_cat],
        ["Total Sensors", str(len(sensors))],
        ["Active Sensors (with data)", str(len(sensor_latest))],
        ["Total Readings Analyzed", str(len(all_readings))],
        ["Report Period", f"Last 100 readings per sensor"],
    ]
    
    t = Table(summary_data, colWidths=[200, 280])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f6feb")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f6f8fa")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7de")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f6f8fa"), colors.white]),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 15))
    
    # AQI gauge chart
    gauge_bytes = _generate_aqi_gauge_chart(overall_aqi, aqi_cat)
    gauge_img = Image(io.BytesIO(gauge_bytes), width=250, height=160)
    elements.append(gauge_img)
    elements.append(Spacer(1, 10))
    
    # ── Hotspots & Cleanest ──
    elements.append(Paragraph("🔴 Top Pollution Hotspots", heading_style))
    
    sorted_sensors = sorted(sensor_latest.items(), key=lambda x: x[1]["pm25"], reverse=True)
    
    hotspot_data = [["Rank", "Sensor ID", "Location", "PM2.5 (µg/m³)", "Status"]]
    for i, (sid, data) in enumerate(sorted_sensors[:10]):
        cat = _aqi_category(data["pm25"])
        hotspot_data.append([
            str(i + 1), sid, data["location"][:30],
            f"{data['pm25']:.1f}", cat
        ])
    
    if len(hotspot_data) > 1:
        ht = Table(hotspot_data, colWidths=[40, 80, 150, 90, 120])
        ht.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#da3633")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7de")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#fff5f5"), colors.white]),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (3, 0), (3, -1), "CENTER"),
        ]))
        elements.append(ht)
    
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph("🟢 Cleanest Zones", heading_style))
    clean_data = [["Rank", "Sensor ID", "Location", "PM2.5 (µg/m³)", "Status"]]
    for i, (sid, data) in enumerate(reversed(sorted_sensors[-5:])):
        cat = _aqi_category(data["pm25"])
        clean_data.append([
            str(i + 1), sid, data["location"][:30],
            f"{data['pm25']:.1f}", cat
        ])
    
    if len(clean_data) > 1:
        ct = Table(clean_data, colWidths=[40, 80, 150, 90, 120])
        ct.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3fb950")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7de")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0fff4"), colors.white]),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (3, 0), (3, -1), "CENTER"),
        ]))
        elements.append(ct)
    
    elements.append(PageBreak())
    
    # ── PM2.5 Trend Chart ──
    elements.append(Paragraph("📈 PM2.5 Historical Trend", heading_style))
    elements.append(Paragraph(
        "Combined PM2.5 readings across all sensors showing raw vs AI-corrected values.",
        body_style
    ))
    
    trend_bytes = _generate_trend_chart(all_readings[:500])
    trend_img = Image(io.BytesIO(trend_bytes), width=480, height=200)
    elements.append(trend_img)
    elements.append(Spacer(1, 15))
    
    # ── Sensor Health ──
    elements.append(Paragraph("🏥 Sensor Network Health", heading_style))
    elements.append(Paragraph(
        "Health scores based on anomaly rate, failure rate, sensor drift, and data density. "
        "Grades: A (90+), B (75+), C (60+), D (40+), F (<40).",
        body_style
    ))
    
    top_health = sorted(sensors_health_data, key=lambda x: x["score"], reverse=True)[:20]
    if top_health:
        health_bytes = _generate_sensor_health_bar(top_health)
        health_img = Image(io.BytesIO(health_bytes), width=480,
                          height=max(170, len(top_health) * 24))
        elements.append(health_img)
    
    elements.append(Spacer(1, 15))
    
    # ── AI Recommendations ──
    elements.append(Paragraph("🎯 AI-Powered Recommendations", heading_style))
    
    recommendations = []
    if overall_aqi > 150:
        recommendations.append("🚨 <b>CRITICAL:</b> Issue immediate public health advisory — PM2.5 levels exceed hazardous thresholds. Limit outdoor activities citywide.")
    if overall_aqi > 100:
        recommendations.append("💨 Activate air purification systems in public buildings, schools, and hospitals.")
        recommendations.append("🚦 Consider implementing odd-even traffic restrictions to reduce vehicular emissions.")
    if overall_aqi > 55:
        recommendations.append("📋 Notify pollution control authorities for industrial emissions inspection in hotspot zones.")
        recommendations.append("💧 Mandate water sprinkling at construction sites within 500m of sensitive zones.")
    
    anomaly_total = sum(1 for r in all_readings if r.is_anomaly == 1)
    if anomaly_total > len(all_readings) * 0.1:
        recommendations.append(f"⚙️ High anomaly rate detected ({anomaly_total} anomalies in {len(all_readings)} readings). Schedule network-wide sensor recalibration.")
    
    recommendations.append("🤖 Continue AI-powered real-time monitoring and automated drift correction across all sensors.")
    recommendations.append("📡 Consider deploying additional sensors in identified pollution hotspot areas for better spatial coverage.")
    
    for rec in recommendations:
        elements.append(Paragraph(f"• {rec}", body_style))
    
    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#30363d")))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        f"<i>Report generated by Vayu-Rakshak AI Engine | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Powered by PyTorch ML + LangChain AI</i>",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                       textColor=colors.HexColor("#8b949e"), alignment=TA_CENTER)
    ))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
