"""
app.py — Streamlit Dashboard for the Vayu-Rakshak Air Quality Monitoring System.

Four sidebar tabs:
  1. 🗺️  Dashboard        — Date-filtered Folium heatmap (PM2.5 intensity)
  2. 🚨  Surveillance     — KPI cards for anomalous / failed sensors
  3. ➕  Register Sensor  — Form to register new sensors via the FastAPI API
  4. 📈  Historical Analytics — Plotly time-series + CSV export

Persistent chatbot:
  "Dr. Vayu" LangChain SQL agent is always accessible via the chat panel
  in any tab. It queries the DB AND checks nearby POIs to explain readings.
"""

import os
import io
import re
import json
import logging
import requests
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
API_BASE = os.getenv("VAYU_API_BASE", "http://localhost:8000")

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="Vayu-Rakshak | Air Quality Monitor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — premium dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f2027, #203a43, #2c5364);
    color: #e0f7fa;
  }
  [data-testid="stSidebar"] .stRadio label { color: #b2ebf2 !important; font-weight: 500; }

  /* Main background */
  .stApp { background: #0d1117; color: #e6edf3; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e2a38, #162032);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
  }
  [data-testid="metric-container"] label { color: #8b949e !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #58a6ff !important; font-size: 1.8rem !important;
  }

  /* KPI anomaly card */
  .kpi-card {
    background: linear-gradient(135deg, #2d1515, #1a0a0a);
    border: 1px solid #ff454577;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 0 12px rgba(255,69,58,0.2);
  }
  .kpi-card h4 { color: #ff6b6b; margin: 0; }

  /* Chat messages */
  .user-bubble {
    background: #1f6feb33;
    border-radius: 12px 12px 2px 12px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    border: 1px solid #1f6feb55;
  }
  .assistant-bubble {
    background: #21262d;
    border-radius: 12px 12px 12px 2px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    border: 1px solid #30363d;
  }

  /* Download button */
  .stDownloadButton > button {
    background: linear-gradient(90deg, #1a7f5a, #239970) !important;
    color: white !important; border-radius: 8px !important;
  }

  /* Primary buttons */
  .stButton > button {
    background: linear-gradient(90deg, #1f6feb, #388bfd) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
  }

  /* Form */
  [data-testid="stForm"] {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 12px; padding: 1.5rem;
  }

  /* Tabs header */
  .tab-header {
    font-size: 1.5rem; font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #79c0ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
  }
  .divider { border-top: 1px solid #21262d; margin: 1rem 0; }

  /* Scrollable chat box */
  .chat-scroll { max-height: 420px; overflow-y: auto; padding-right: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper: API calls with error handling
# ─────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def fetch_all_readings(anomaly_only: bool = False) -> pd.DataFrame:
    try:
        params = {"anomaly_only": "true" if anomaly_only else "false", "limit": 5000}
        r = requests.get(f"{API_BASE}/readings", params=params, timeout=10)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception as e:
        st.warning(f"⚠️ Could not load readings: {e}")
        return pd.DataFrame()





@st.cache_data(ttl=30, show_spinner=False)
def fetch_sensor_readings(sensor_id: str) -> pd.DataFrame:
    try:
        r = requests.get(f"{API_BASE}/readings/{sensor_id}", timeout=10)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception as e:
        st.warning(f"⚠️ Could not load readings for {sensor_id}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_sensors() -> pd.DataFrame:
    try:
        r = requests.get(f"{API_BASE}/sensors", timeout=10)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception as e:
        st.warning(f"⚠️ Could not load sensor list: {e}")
        return pd.DataFrame()


def fetch_sensor_analysis(sensor_id: str) -> dict:
    try:
        r = requests.get(f"{API_BASE}/analyze_sensor/{sensor_id}", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_city_aqi() -> dict:
    try:
        r = requests.get(f"{API_BASE}/city_aqi", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def fetch_sensor_health(sensor_id: str) -> dict:
    try:
        r = requests.get(f"{API_BASE}/sensor_health/{sensor_id}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def trigger_drift_simulation(sensor_id: str, drift_type: str = "offset", magnitude: float = 15.0) -> dict:
    try:
        r = requests.post(f"{API_BASE}/simulate_drift",
                          params={"sensor_id": sensor_id, "drift_type": drift_type, "magnitude": magnitude},
                          timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def pm25_color(val: float) -> str:
    if val is None:   return "#8b949e"
    if val < 12:      return "#3fb950"
    if val < 35:      return "#e3b341"
    if val < 55:      return "#f0883e"
    if val < 150:     return "#ff7b72"
    return "#ff453a"


def pm25_label(val: float) -> str:
    if val is None:   return "Unknown"
    if val < 12:      return "Good"
    if val < 35:      return "Moderate"
    if val < 55:      return "Unhealthy for Sensitive"
    if val < 150:     return "Unhealthy"
    return "Very Unhealthy / Hazardous"


def aqi_color(category: str) -> str:
    return {"Good": "#3fb950", "Moderate": "#e3b341",
            "Unhealthy for Sensitive Groups": "#f0883e",
            "Unhealthy": "#ff7b72", "Very Unhealthy": "#ff453a",
            "Hazardous": "#da3633"}.get(category, "#8b949e")


def severity_color(s: str) -> str:
    return {"critical": "#ff453a", "high": "#ff7b72",
            "medium": "#e3b341", "low": "#3fb950"}.get(s, "#8b949e")


# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
if "ui_target" not in st.session_state:
    st.session_state.ui_target = {
        "tab": "🗺️ Dashboard",
        "lat": 28.6139,
        "lon": 77.2090,
        "zoom": 11,
        "sensor_id": None
    }
if "selected_analysis_sensor" not in st.session_state:
    st.session_state.selected_analysis_sensor = None

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/wind.png",
        width=72,
    )
    st.markdown("## 🌿 Vayu-Rakshak")
    st.markdown("*Hyperlocal Air Quality Monitor*")
    st.markdown("---")

    # Find index of current tab in list for radio default
    tab_list = ["🗺️ Dashboard", "🧠 AI Analysis", "🚨 Surveillance", "➕ Register Sensor", "📈 Historical Analytics"]
    try:
        tab_index = tab_list.index(st.session_state.ui_target["tab"])
    except ValueError:
        tab_index = 0

    page = st.radio(
        "Navigation",
        tab_list,
        index=tab_index,
        label_visibility="collapsed",
        key="nav_radio"
    )
    # Sync session state if user clicks radio manually
    st.session_state.ui_target["tab"] = page

    st.markdown("---")
    st.markdown("**API Server**")
    api_status_placeholder = st.empty()
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200:
            api_status_placeholder.success("🟢 Online")
        else:
            api_status_placeholder.error("🔴 Error")
    except Exception:
        api_status_placeholder.error("🔴 Offline")

    st.markdown(f"<small style='color:#8b949e'>Endpoint: {API_BASE}</small>", unsafe_allow_html=True)
    st.markdown("---")

    # OpenAI key (optional — only needed for chatbot)
    openai_key = st.text_input(
        "OpenAI API Key (for chatbot)",
        type="password",
        placeholder="sk-...",
        help="Required only for the Dr. Vayu chatbot. Leave blank to disable it.",
    )


# ═══════════════════════════════════════════════════════
# TAB 1: DASHBOARD (ENHANCED)
# ═══════════════════════════════════════════════════════
if page == "🗺️ Dashboard":
    st.markdown('<p class="tab-header">🗺️ City Air Quality Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── City AQI Hero Card ──
    city_data = fetch_city_aqi()
    if city_data:
        cat_col = aqi_color(city_data["aqi_category"])
        trend_icon = {"rising": "📈 Rising", "falling": "📉 Falling", "stable": "➡️ Stable"}.get(city_data["trend_direction"], "➡️")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
                    border: 1px solid {cat_col}55; border-radius: 16px; padding: 1.5rem 2rem;
                    margin-bottom: 1.2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                <div>
                    <h2 style="margin:0; color:{cat_col}; font-size:2.2rem;">
                        🏙️ {city_data["city_name"]} — {city_data["overall_aqi"]:.0f} AQI
                    </h2>
                    <p style="margin:4px 0 0; color:#b2ebf2; font-size:1.1rem;">
                        {city_data["aqi_category"]} &nbsp;|&nbsp; {trend_icon}
                        &nbsp;|&nbsp; Health Risk: <b>{city_data["health_risk_index"]}/10</b>
                    </p>
                </div>
                <div style="text-align:right;">
                    <p style="margin:0; color:#8b949e; font-size:0.85rem;">
                        📡 {city_data["active_sensors"]}/{city_data["total_sensors"]} sensors active
                        &nbsp;|&nbsp; Variance: {city_data["spatial_variance"]:.1f} &nbsp;|&nbsp;
                        Coverage: {city_data["sub_indices"].get("sensor_coverage", 0):.0f}%
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Hotspot & Cleanest side-by-side
        hcol1, hcol2 = st.columns(2)
        with hcol1:
            st.markdown("##### 🔴 Top Pollution Hotspots")
            for i, h in enumerate(city_data.get("hotspots", [])[:5]):
                st.markdown(f"""<div style="background:#2d1515; border:1px solid #ff454533;
                    border-radius:8px; padding:0.5rem 0.8rem; margin-bottom:0.4rem;">
                    <b style='color:#ff6b6b;'>#{i+1}</b> {h['location']}
                    — <b style='color:#ff7b72;'>{h['pm25']} µg/m³</b>
                    <span style='color:#666; font-size:0.8rem;'>({h['sensor_id']})</span>
                </div>""", unsafe_allow_html=True)
        with hcol2:
            st.markdown("##### 🟢 Cleanest Zones")
            for i, c in enumerate(city_data.get("cleanest_zones", [])[:5]):
                st.markdown(f"""<div style="background:#0d2818; border:1px solid #3fb95033;
                    border-radius:8px; padding:0.5rem 0.8rem; margin-bottom:0.4rem;">
                    <b style='color:#3fb950;'>#{i+1}</b> {c['location']}
                    — <b style='color:#3fb950;'>{c['pm25']} µg/m³</b>
                    <span style='color:#666; font-size:0.8rem;'>({c['sensor_id']})</span>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Date/time filter
    col_d1, col_d2, col_d3 = st.columns([1, 1, 2])
    with col_d1:
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=30))
    with col_d2:
        end_date   = st.date_input("End Date",   value=date.today() + timedelta(days=30))
    with col_d3:
        pm25_threshold = st.slider("PM2.5 ≥", min_value=0, max_value=300, value=0, step=5)

    with st.spinner("Loading sensor data…"):
        df = fetch_all_readings()

    if df.empty:
        st.info("No data found. Ingest some readings or check if the API is running.")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df_filtered = df[
            (df["timestamp"].dt.date >= start_date) &
            (df["timestamp"].dt.date <= end_date) &
            (df["pm2p5_corrected"].fillna(0) >= pm25_threshold)
        ]

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📡 Sensors", df_filtered["sensor_id"].nunique())
        k2.metric("📊 Readings",  len(df_filtered))
        avg_pm = df_filtered["pm2p5_corrected"].mean()
        max_pm = df_filtered["pm2p5_corrected"].max()
        k3.metric("📏 Avg PM2.5",  f"{avg_pm:.1f} µg/m³" if not pd.isna(avg_pm) else "N/A")
        k4.metric("⚠️ Max PM2.5", f"{max_pm:.1f} µg/m³" if not pd.isna(max_pm) else "N/A",
                  delta_color="inverse")

        st.markdown("")

        # ── Map with clickable sensors ──
        if df_filtered.dropna(subset=["lat", "long", "pm2p5_corrected"]).empty:
            st.warning("No readings with valid coordinates in the selected date range.")
        else:
            latest = df_filtered.sort_values("timestamp").groupby("sensor_id").last().reset_index()
            latest = latest.dropna(subset=["lat", "long", "pm2p5_corrected"])

            # Sensor selector for analysis
            sensor_list = latest["sensor_id"].tolist()
            col_map, col_select = st.columns([4, 1])
            with col_select:
                selected_for_analysis = st.selectbox("🔍 Analyze sensor", ["— select —"] + sensor_list, key="dash_sensor_select")
                if st.button("🧠 Deep Analyze", use_container_width=True, key="analyze_btn"):
                    if selected_for_analysis != "— select —":
                        st.session_state.selected_analysis_sensor = selected_for_analysis
                        st.session_state.ui_target["tab"] = "🧠 AI Analysis"
                        st.rerun()

            with col_map:
                if not latest.empty:
                    m = folium.Map(
                        location=[st.session_state.ui_target["lat"], st.session_state.ui_target["lon"]],
                        zoom_start=st.session_state.ui_target["zoom"],
                        tiles="CartoDB dark_matter",
                    )
                    for _, row in latest.iterrows():
                        val = row["pm2p5_corrected"]
                        color = pm25_color(val)
                        folium.Circle(
                            location=[row["lat"], row["long"]], radius=1000,
                            color=color, fill=True, fill_color=color, fill_opacity=0.15, weight=1,
                        ).add_to(m)
                        folium.CircleMarker(
                            location=[row["lat"], row["long"]], radius=8,
                            color=color, fill=True, fill_color=color, fill_opacity=0.85,
                            popup=folium.Popup(
                                f"<b>{row.get('location_name', row['sensor_id'])}</b><br>"
                                f"Sensor: {row['sensor_id']}<br>"
                                f"PM2.5: <b>{val:.1f} µg/m³</b><br>"
                                f"Status: {pm25_label(val)}<br>"
                                f"Last: {row['timestamp']}", max_width=220,
                            ),
                            tooltip=f"{row['sensor_id']} — {val:.1f} µg/m³",
                        ).add_to(m)
                    st_folium(m, use_container_width=True, height=500)

            st.caption("🔵 Good  🟢 Moderate  🟡 Unhealthy for Sensitive  🔴 Unhealthy  ⛔ Very Unhealthy")

        # ── Correlation Heatmap ──
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("##### 📊 Environmental Correlation Matrix")
        corr_cols = ["pm2p5_corrected", "pm2p5_raw", "temperature", "humidity"]
        corr_available = [c for c in corr_cols if c in df_filtered.columns]
        if len(corr_available) >= 2:
            corr_data = df_filtered[corr_available].dropna()
            if len(corr_data) > 5:
                corr_matrix = corr_data.corr()
                labels = {"pm2p5_corrected": "PM2.5 (AI)", "pm2p5_raw": "PM2.5 (Raw)",
                          "temperature": "Temp °C", "humidity": "Humidity %"}
                corr_matrix = corr_matrix.rename(index=labels, columns=labels)
                fig_corr = px.imshow(
                    corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto",
                )
                fig_corr.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#e6edf3"), height=350,
                    margin=dict(t=30, b=30, l=30, r=30),
                )
                st.plotly_chart(fig_corr, use_container_width=True)

        # ── Sensor Network Health Grid ──
        st.markdown("##### 🏥 Sensor Network Health")
        sensors_df = fetch_sensors()
        if not sensors_df.empty:
            health_dots = []
            for sid in sensors_df["sensor_id"].tolist()[:30]:  # limit for performance
                health = fetch_sensor_health(sid)
                if health:
                    gc = {"A": "#3fb950", "B": "#58a6ff", "C": "#e3b341", "D": "#f0883e", "F": "#ff453a"}
                    grade = health["health_grade"]
                    bg_color = gc.get(grade, "#666")
                    health_dots.append(
                        f"<span title='{sid}: {grade}' style='display:inline-block;"
                        f"width:18px; height:18px; border-radius:50%; margin:3px;"
                        f"background:{bg_color};'></span>"
                    )
            if health_dots:
                st.markdown(
                    f"<div style='display:flex; flex-wrap:wrap; gap:2px;'>{''.join(health_dots)}</div>"
                    f"<p style='color:#8b949e; font-size:0.75rem; margin-top:4px;'>"
                    f"🟢A  🔵B  🟡C  🟠D  🔴F  — showing {len(health_dots)} sensors</p>",
                    unsafe_allow_html=True,
                )

# ═══════════════════════════════════════════════════════
# TAB: 🧠 AI ANALYSIS
# ═══════════════════════════════════════════════════════
elif page == "🧠 AI Analysis":
    st.markdown('<p class="tab-header">🧠 AI Pollution Analysis Engine</p>', unsafe_allow_html=True)
    st.markdown("Select a sensor to see AI-powered pollution reasoning, health scoring, forecasts & drift detection.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    sensors_df = fetch_sensors()
    if sensors_df.empty:
        st.info("No sensors registered.")
    else:
        sensor_ids = sensors_df["sensor_id"].tolist()
        default_idx = 0
        if st.session_state.selected_analysis_sensor in sensor_ids:
            default_idx = sensor_ids.index(st.session_state.selected_analysis_sensor)

        col_sel, col_drift = st.columns([3, 1])
        with col_sel:
            analysis_sensor = st.selectbox("📡 Select Sensor", sensor_ids, index=default_idx, key="ai_sensor_select")
            st.session_state.selected_analysis_sensor = analysis_sensor
        with col_drift:
            st.markdown("##### ⚡ Drift Simulator")
            drift_type = st.selectbox("Drift Type", ["offset", "humidity", "random_walk"], key="drift_type_sel")
            drift_mag = st.slider("Magnitude", 5.0, 50.0, 15.0, key="drift_mag_sl")
            if st.button("🌀 Inject Drift", use_container_width=True, key="inject_drift_btn"):
                with st.spinner("Applying drift & re-running AI correction..."):
                    result = trigger_drift_simulation(analysis_sensor, drift_type, drift_mag)
                if result:
                    st.success(f"✅ {result['message']}")
                else:
                    st.error("Failed to apply drift.")

        with st.spinner("Running AI analysis engine..."):
            analysis = fetch_sensor_analysis(analysis_sensor)

        if not analysis:
            st.error("Could not fetch analysis. Is the API running?")
        else:
            # ── Header Card ──
            cat_color = aqi_color(analysis["aqi_category"])
            drift_badge = ("🔴 DRIFT DETECTED" if analysis["drift_detected"]
                           else "🟢 No Drift")
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0f2027, #203a43);
                        border: 1px solid {cat_color}55; border-radius: 14px;
                        padding: 1.2rem 1.8rem; margin-bottom: 1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                    <div>
                        <h3 style="margin:0; color:{cat_color};">
                            📡 {analysis["sensor_id"]} — {analysis["location_name"]}
                        </h3>
                        <p style="margin:4px 0; color:#b2ebf2;">
                            PM2.5: <b>{analysis["current_pm25"]} µg/m³</b> &nbsp;|&nbsp;
                            {analysis["aqi_category"]} &nbsp;|&nbsp; {drift_badge}
                        </p>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:2.5rem; font-weight:700; color:{cat_color};">
                            {analysis["health_grade"]}
                        </div>
                        <div style="color:#8b949e; font-size:0.8rem;">
                            Health: {analysis["health_score"]}/100
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Two-column layout: Causes + Actions ──
            col_causes, col_actions = st.columns(2)

            with col_causes:
                st.markdown("##### 🔬 Pollution Source Probabilities")
                for cause in analysis["pollution_causes"]:
                    pct = cause["probability"] * 100
                    bar_color = "#ff453a" if pct > 60 else "#e3b341" if pct > 30 else "#3fb950"
                    st.markdown(f"""
                    <div style="margin-bottom:0.6rem;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                            <span>{cause["icon"]} {cause["cause"]}</span>
                            <span style="color:{bar_color}; font-weight:600;">{pct:.0f}%</span>
                        </div>
                        <div style="background:#21262d; border-radius:6px; height:10px; overflow:hidden;">
                            <div style="background:{bar_color}; height:100%; width:{pct}%;
                                        border-radius:6px; transition: width 0.5s;"></div>
                        </div>
                        <div style="color:#8b949e; font-size:0.75rem; margin-top:2px;">
                            {cause["description"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_actions:
                st.markdown("##### 🎯 Recommended Actions")
                for action in analysis["recommended_actions"]:
                    sev_col = severity_color(action["severity"])
                    st.markdown(f"""
                    <div style="background:#161b22; border-left:3px solid {sev_col};
                                border-radius:0 8px 8px 0; padding:0.6rem 1rem;
                                margin-bottom:0.5rem;">
                        <span>{action["icon"]}</span>
                        <span style="color:#e6edf3;">{action["action"]}</span>
                        <span style="float:right; color:{sev_col}; font-size:0.75rem;
                                     text-transform:uppercase; font-weight:600;">
                            {action["severity"]}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

            # ── 24h Forecast Sparkline ──
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("##### 🔮 24-Hour PM2.5 Forecast")
            forecast_data = analysis.get("forecast_24h", [])
            if forecast_data:
                hours_labels = [(datetime.now() + timedelta(hours=i+1)).strftime("%H:%M") for i in range(len(forecast_data))]
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=hours_labels, y=forecast_data,
                    fill="tozeroy", fillcolor="rgba(88,166,255,0.1)",
                    line=dict(color="#58a6ff", width=2.5),
                    name="Forecast PM2.5",
                ))
                fig_fc.add_hline(y=150, line_dash="dash", line_color="#ff7b72", line_width=1,
                                 annotation_text="Hazardous", annotation_font_color="#ff7b72")
                fig_fc.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#e6edf3"), height=280,
                    xaxis=dict(title="Time", gridcolor="#21262d", color="#8b949e"),
                    yaxis=dict(title="PM2.5 µg/m³", gridcolor="#21262d", color="#8b949e"),
                    margin=dict(t=20, b=40, l=50, r=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_fc, use_container_width=True)

            # ── Drift Visualization ──
            if analysis["drift_detected"]:
                st.markdown("##### 📉 Sensor Drift Detection")
                st.markdown(f"""
                <div style="background:#2d1515; border:1px solid #ff454555; border-radius:10px;
                            padding:1rem; margin-bottom:0.5rem;">
                    <b style="color:#ff6b6b;">⚠️ Drift Magnitude: {analysis["drift_magnitude"]:.2f} µg/m³</b>
                    <p style="color:#ccc; margin:4px 0 0;">
                        The AI model has detected significant divergence between raw sensor readings and
                        expected values. Automatic correction is being applied, but physical recalibration
                        is recommended.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Show raw vs corrected chart for this sensor
                sensor_readings_df = fetch_sensor_readings(analysis_sensor)
                if not sensor_readings_df.empty:
                    sensor_readings_df["timestamp"] = pd.to_datetime(sensor_readings_df["timestamp"], errors="coerce")
                    sensor_readings_df = sensor_readings_df.sort_values("timestamp")
                    fig_drift = go.Figure()
                    if "pm2p5_drifted" in sensor_readings_df.columns:
                        fig_drift.add_trace(go.Scatter(
                            x=sensor_readings_df["timestamp"],
                            y=sensor_readings_df["pm2p5_drifted"],
                            name="Drifted (Sensor Output)", line=dict(color="#ff7b72", dash="dot", width=1.5),
                        ))
                    fig_drift.add_trace(go.Scatter(
                        x=sensor_readings_df["timestamp"],
                        y=sensor_readings_df["pm2p5_raw"],
                        name="Raw (Original)", line=dict(color="#8b949e", width=1.5),
                    ))
                    fig_drift.add_trace(go.Scatter(
                        x=sensor_readings_df["timestamp"],
                        y=sensor_readings_df["pm2p5_corrected"],
                        name="AI Corrected ✅", line=dict(color="#3fb950", width=2.5),
                        fill="tozeroy", fillcolor="rgba(63,185,80,0.08)",
                    ))
                    fig_drift.update_layout(
                        title="Raw → Drifted → AI Corrected Pipeline",
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        font=dict(color="#e6edf3"), height=350,
                        xaxis=dict(gridcolor="#21262d", color="#8b949e"),
                        yaxis=dict(title="PM2.5 µg/m³", gridcolor="#21262d", color="#8b949e"),
                        legend=dict(bgcolor="rgba(0,0,0,0)"),
                        margin=dict(t=40, b=30, l=50, r=20),
                    )
                    st.plotly_chart(fig_drift, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB: SURVEILLANCE
# ═══════════════════════════════════════════════════════
elif page == "🚨 Surveillance":
    st.markdown('<p class="tab-header">🚨 Anomaly & Failure Surveillance</p>', unsafe_allow_html=True)
    st.markdown("Sensors flagged with `is_anomaly = 1` or `is_failure = 1` are shown below.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    with st.spinner("Loading anomaly data…"):
        df_all   = fetch_all_readings(anomaly_only=False)

    if df_all.empty:
        st.info("No data available.")
    else:
        df_alert = df_all[(df_all["is_anomaly"] == 1) | (df_all["is_failure"] == 1)]

        # Summary KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🔴 Anomaly Readings",  int((df_all["is_anomaly"] == 1).sum()))
        m2.metric("⚙️ Failure Readings",  int((df_all["is_failure"] == 1).sum()))
        m3.metric("📡 Affected Sensors",  df_alert["sensor_id"].nunique())
        m4.metric("🔆 Max PM2.5 in Alerts",
                  f"{df_alert['pm2p5_corrected'].max():.1f} µg/m³"
                  if not df_alert.empty else "N/A")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        if df_alert.empty:
            st.success("✅ No anomalies or sensor failures detected in the current dataset.")
        else:
            # Per-sensor KPI cards
            st.subheader("Flagged Sensors")
            sensors_flagged = df_alert.groupby("sensor_id").agg(
                location_name=("location_name", "first"),
                lat=("lat", "first"),
                long=("long", "first"),
                anomaly_count=("is_anomaly", "sum"),
                failure_count=("is_failure", "sum"),
                max_pm25=("pm2p5_corrected", "max"),
                last_seen=("timestamp", "max"),
            ).reset_index()

            for _, row in sensors_flagged.iterrows():
                flags = []
                if row["anomaly_count"] > 0:
                    flags.append(f"🔴 {int(row['anomaly_count'])} Anomalies")
                if row["failure_count"] > 0:
                    flags.append(f"⚙️ {int(row['failure_count'])} Failures")
                flag_str = "  |  ".join(flags)

                st.markdown(
                    f"""<div class="kpi-card">
                        <h4>📡 {row['sensor_id']} — {row.get('location_name', 'Unknown')}</h4>
                        <p style="margin:4px 0; color:#ccc;">
                          {flag_str} &nbsp;|&nbsp; Max PM2.5: <b>{row['max_pm25']:.1f} µg/m³</b>
                          &nbsp;|&nbsp; Last seen: {row['last_seen']}
                        </p>
                        <p style="margin:0; color:#8b949e; font-size:0.85rem;">
                          📍 ({row['lat']:.4f}, {row['long']:.4f})
                        </p>
                      </div>""",
                    unsafe_allow_html=True,
                )

            # Full alert table
            st.markdown("---")
            st.subheader("Detailed Alert Log")
            disp = df_alert[[
                "timestamp", "sensor_id", "location_name", "pm2p5_raw",
                "pm2p5_corrected", "temperature", "humidity", "is_anomaly", "is_failure"
            ]].sort_values("timestamp", ascending=False)

            st.dataframe(
                disp.style.applymap(
                    lambda v: "background-color: #3d1c1c;" if v == 1 else "",
                    subset=["is_anomaly", "is_failure"],
                ),
                use_container_width=True,
                height=400,
            )


# ═══════════════════════════════════════════════════════
# TAB 3: REGISTER SENSOR
# ═══════════════════════════════════════════════════════
elif page == "➕ Register Sensor":
    st.markdown('<p class="tab-header">➕ Register a New Sensor</p>', unsafe_allow_html=True)
    st.markdown(
        "Add a new IoT sensor to the Vayu-Rakshak network. "
        "You will receive an **API key** — keep it safe; it is required to authenticate data ingestion."
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_form, col_info = st.columns([3, 2])

    with col_form:
        with st.form("register_sensor_form", clear_on_submit=True):
            st.markdown("#### Sensor Details")
            sensor_id_input      = st.text_input("Sensor ID *", placeholder="e.g. ARI-2100")
            location_name_input  = st.text_input("Location Name *", placeholder="e.g. Connaught Place, Delhi")
            c1, c2 = st.columns(2)
            with c1:
                lat_input  = st.number_input("Latitude *",  value=28.6139, format="%.6f", step=0.0001)
            with c2:
                long_input = st.number_input("Longitude *", value=77.2090, format="%.6f", step=0.0001)
            install_date_input = st.date_input("Installation Date", value=date.today())

            submitted = st.form_submit_button("🚀 Register Sensor", use_container_width=True)

        if submitted:
            if not sensor_id_input.strip():
                st.error("Sensor ID is required.")
            elif not location_name_input.strip():
                st.error("Location Name is required.")
            else:
                payload = {
                    "sensor_id":          sensor_id_input.strip(),
                    "location_name":      location_name_input.strip(),
                    "lat":                lat_input,
                    "long":               long_input,
                    "installation_date":  str(install_date_input),
                }
                with st.spinner("Registering sensor…"):
                    try:
                        r = requests.post(
                            f"{API_BASE}/register_sensor",
                            json=payload,
                            timeout=10,
                        )
                        if r.status_code == 201:
                            data = r.json()
                            st.success(f"✅ Sensor **{data['sensor_id']}** registered successfully!")
                            st.code(
                                f"API Key: {data['api_key']}\n\n"
                                "⚠️ Store this key securely. It cannot be recovered later.\n"
                                "Use it as the 'x-api-key' header in /ingest requests.",
                                language="text"
                            )
                            # Invalidate sensors cache
                            fetch_sensors.clear()
                        elif r.status_code == 409:
                            st.warning(r.json().get("detail", "Sensor already exists."))
                        else:
                            st.error(f"Error {r.status_code}: {r.json().get('detail', 'Unknown error')}")
                    except requests.exceptions.ConnectionError:
                        st.error(f"Cannot connect to API at {API_BASE}. Is the FastAPI server running?")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

    with col_info:
        st.markdown("#### 🔑 How API Key Works")
        st.info(
            "After registration, you receive a **UUID v4 API key**.\n\n"
            "Every `/ingest` request from this sensor **must include** the header:\n"
            "```\nx-api-key: <your-api-key>\n```\n"
            "Requests without a valid key are rejected with `403 Forbidden`."
        )
        st.markdown("#### 📍 Finding Coordinates")
        st.markdown(
            "[Google Maps](https://maps.google.com) → right-click on a location "
            "→ *What's here?* → copy lat/lng."
        )

        # Live preview map
        st.markdown("#### Live Location Preview")
        preview_map = folium.Map(location=[lat_input, long_input], zoom_start=14,
                                 tiles="CartoDB dark_matter")
        folium.Marker(
            [lat_input, long_input],
            popup=location_name_input or "New Sensor",
            icon=folium.Icon(color="green", icon="cloud"),
        ).add_to(preview_map)
        st_folium(preview_map, height=280, use_container_width=True, key="register_map")


# ═══════════════════════════════════════════════════════
# TAB 4: HISTORICAL ANALYTICS
# ═══════════════════════════════════════════════════════
elif page == "📈 Historical Analytics":
    st.markdown('<p class="tab-header">📈 Historical Analytics</p>', unsafe_allow_html=True)
    st.markdown("Explore PM2.5 trends over time for any sensor and export data as CSV.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    sensors_df = fetch_sensors()

    if sensors_df.empty:
        st.info("No sensors registered. Go to **Register Sensor** to add one.")
    else:
        sensor_ids = sensors_df["sensor_id"].tolist()

        col_sel, col_export = st.columns([3, 1])
        with col_sel:
            # Determine default index based on ui_target
            default_sensor_idx = 0
            if st.session_state.ui_target["sensor_id"] in sensor_ids:
                default_sensor_idx = sensor_ids.index(st.session_state.ui_target["sensor_id"])
            
            selected_sensor = st.selectbox("Select Sensor", sensor_ids, index=default_sensor_idx)
            # Sync back if user changes manually
            st.session_state.ui_target["sensor_id"] = selected_sensor

        with st.spinner(f"Loading data for {selected_sensor}…"):
            sensor_df = fetch_sensor_readings(selected_sensor)

        with col_export:
            if not sensor_df.empty:
                csv_bytes = sensor_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Export CSV",
                    data=csv_bytes,
                    file_name=f"{selected_sensor}_readings.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        if sensor_df.empty:
            st.info(f"No readings found for sensor **{selected_sensor}**.")
        else:
            sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"], errors="coerce")
            sensor_df = sensor_df.sort_values("timestamp")

            # KPI row
            k1, k2, k3, k4 = st.columns(4)
            latest_row = sensor_df.iloc[-1]
            k1.metric("Last PM2.5 (corrected)",
                      f"{latest_row['pm2p5_corrected']:.1f} µg/m³"
                      if pd.notna(latest_row['pm2p5_corrected']) else "N/A")
            k2.metric("Last PM2.5 (raw)",
                      f"{latest_row['pm2p5_raw']:.1f} µg/m³"
                      if pd.notna(latest_row['pm2p5_raw']) else "N/A")
            k3.metric("Total Readings", len(sensor_df))
            k4.metric("Anomalies Flagged", int(sensor_df["is_anomaly"].sum()))

            # Plotly dual-line chart
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=sensor_df["timestamp"],
                y=sensor_df["pm2p5_raw"],
                name="PM2.5 Raw",
                line=dict(color="#8b949e", width=1.5, dash="dot"),
                opacity=0.7,
            ))

            fig.add_trace(go.Scatter(
                x=sensor_df["timestamp"],
                y=sensor_df["pm2p5_corrected"],
                name="PM2.5 Corrected (AI)",
                line=dict(color="#58a6ff", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(88,166,255,0.07)",
            ))

            # WHO threshold line
            fig.add_hline(
                y=150, line_dash="dash", line_color="#ff7b72", line_width=1,
                annotation_text="WHO Hazardous (150 µg/m³)",
                annotation_position="top left",
                annotation_font_color="#ff7b72",
            )

            fig.update_layout(
                title=dict(
                    text=f"PM2.5 Timeline — Sensor {selected_sensor}",
                    font=dict(size=18, color="#e6edf3"),
                ),
                xaxis=dict(
                    title="Timestamp", showgrid=True, gridcolor="#21262d",
                    color="#8b949e",
                ),
                yaxis=dict(
                    title="PM2.5 (µg/m³)", showgrid=True, gridcolor="#21262d",
                    color="#8b949e",
                ),
                legend=dict(
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3", size=12),
                ),
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
                hovermode="x unified",
                height=450,
                margin=dict(t=60, b=40, l=60, r=30),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Expandable raw data table
            with st.expander("📋 View Raw Data Table"):
                st.dataframe(
                    sensor_df[[
                        "timestamp", "pm2p5_raw", "pm2p5_corrected",
                        "temperature", "humidity", "is_anomaly", "is_failure"
                    ]].sort_values("timestamp", ascending=False),
                    use_container_width=True,
                )


# ═══════════════════════════════════════════════════════
# PERSISTENT CHATBOT — Dr. Vayu
# ═══════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🤖 Dr. Vayu — AI Environmental Scientist")
st.markdown(
    "Ask me anything about the air quality data. "
    "I can query the database, find nearby pollution sources, and explain sensor readings."
)

if not openai_key:
    pass
else:
    # Initialise session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = None

    # Lazy-load agent
    if st.session_state.agent_executor is None:
        with st.spinner("Initialising Dr. Vayu agent…"):
            try:
                from agent import get_agent_executor
                st.session_state.agent_executor = get_agent_executor(openai_key)
                st.success("✅ Dr. Vayu is ready!")
            except Exception as e:
                st.error(f"Could not initialise agent: {e}")

    # Render chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant", avatar="🌿"):
                    st.markdown(content)

    # Input
    user_input = st.chat_input(
        "e.g. 'Why is PM2.5 high at ARI-1885 in the morning?' or 'Show me all anomalies today.'"
    )

    if user_input and st.session_state.agent_executor:
        # Add user message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        # Run agent
        with st.chat_message("assistant", avatar="🌿"):
            with st.spinner("Dr. Vayu is analysing data and nearby sources…"):
                try:
                    response = st.session_state.agent_executor.invoke({"input": user_input})
                    raw_answer = response.get("output", "I could not find a relevant answer.")
                    
                    # Intercept UI signals
                    match = re.search(r"UI_SIGNAL: action=(\w+), params=(.+)", raw_answer)
                    if match:
                        action = match.group(1)
                        params = json.loads(match.group(2))
                        
                        if action == "navigate_to":
                            st.session_state.ui_target["tab"] = params.get("tab", st.session_state.ui_target["tab"])
                            if "sensor_id" in params:
                                st.session_state.ui_target["sensor_id"] = params["sensor_id"]
                        elif action == "zoom_to":
                            st.session_state.ui_target["tab"] = "🗺️ Dashboard"
                            st.session_state.ui_target["lat"] = params.get("lat", st.session_state.ui_target["lat"])
                            st.session_state.ui_target["lon"] = params.get("lon", st.session_state.ui_target["lon"])
                            st.session_state.ui_target["zoom"] = params.get("zoom", 14)
                        
                        # Strip the raw signal from the displayed message
                        answer = raw_answer.split("UI_SIGNAL:")[0].strip()
                        if not answer:
                            answer = f"Sure! I've updated the UI for you."
                        st.rerun() # Trigger rerun to reflect UI changes
                    else:
                        answer = raw_answer
                except Exception as e:
                    answer = f"⚠️ Error during analysis: {e}"
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
