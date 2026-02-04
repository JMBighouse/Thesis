#!/usr/bin/env python3
"""
Cliff Watch – Live Feed Updater
================================
Version: 12.31.25 v6 (Ocean-Enhanced Model with 1-Hour Precipitation)
Author: Janna Michele Bighouse ( AI tools used: Claude Opus 4.5 )

MODEL INFO:
- Ocean-Enhanced Random Forest with 78 predictors
- Trained on 132 samples: 66 landslides + 66 paired controls
- Validated using 5-fold Spatio-Temporal Cross-Validation (STCV)
- Mean AUC: 0.946, Accuracy: 91.7%, F1: 0.918
- Baseline comparison: AUC 0.667 (+27.9% improvement with ocean variables)

FEATURES:
- Auto-relaunches with correct ArcGIS Python environment
- Optimal classification threshold from STCV (loaded from config)
- Risk categories: Low < 0.30, Medium 0.30-0.60, High >= 0.60
- RAINFALL SAFETY OVERRIDE (based on USGS research):
  * Elevated: ≥12.5mm (0.5 in) in 1 hour OR ≥75mm (3 in) in 24 hours
  * High: ≥25mm (1 in) in 1 hour OR ≥100mm (4 in) in 24 hours
  * Critical: ≥40mm (1.6 in) in 1 hour OR ≥200mm (8 in) in 7 days
- Environmental triggers for active warnings
- CDIP/NDBC wave data with station fallback
- MRMS precipitation data (1-hour AND 24-hour)
- USGS seismic activity monitoring
- Coastal distance weighting for ocean features
- Detailed risk_reason field for popup explanations

PRECIPITATION NOTE:
The MRMS ImageServer provides real-time precipitation accumulation for multiple
time periods: 1hr, 3hr, 6hr, 12hr, 24hr, 48hr, 72hr. We fetch 1-hour data for
intensity monitoring (critical for safety) and 24-hour for accumulation tracking.
Multi-day totals beyond 72hr are estimated by multiplying.

USGS RESEARCH CITATIONS:
- Campbell, R.H. (1975). USGS Prof Paper 851: 1 in/hr triggers shallow landslides
- USGS OF 2005-1067: La Conchita - 8 in/15 days caused fatal landslide
- Ventura County Emergency Services: 1 in/hr debris flow threshold

CHANGES IN v6:
- Added 1-hour precipitation fetching (PRCP_1h field)
- Updated rainfall safety override to prioritize hourly intensity
- Thresholds based on USGS debris flow research
"""

import sys
import subprocess

# =============================================================================
# AUTO-RELAUNCH WITH CORRECT PYTHON ENVIRONMENT
# =============================================================================
REQUIRED_PYTHON = r"C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe"

import os as _os
if _os.name == 'nt' and _os.path.exists(REQUIRED_PYTHON):
    if sys.executable.lower() != REQUIRED_PYTHON.lower():
        print(f"[INFO] Relaunching with ArcGIS Python environment...")
        print(f"       Current: {sys.executable}")
        print(f"       Required: {REQUIRED_PYTHON}")
        result = subprocess.call([REQUIRED_PYTHON] + sys.argv)
        sys.exit(result)

# =============================================================================
# IMPORTS
# =============================================================================

import os
import io
import gzip
import json
import time
import math
import re
import traceback
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional, Any

import pandas as pd
import requests

# Joblib for model loading
try:
    from joblib import load as joblib_load
except ImportError:
    import pickle
    def joblib_load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

# ArcGIS imports
from arcgis.gis import GIS
from arcgis.features import FeatureLayer
from arcgis.raster import ImageryLayer

# Fix encoding for Windows Task Scheduler (prevents emoji crash)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

# ArcGIS Online Feature Layer
POINTS_ITEMID = "965af718759e4242b856aab2653267f4"

# Model and config paths - UPDATED FOR 12.31.25 OCEAN-ENHANCED MODEL
MODEL_DIR = r"C:\Users\MB\Desktop\SSCI 594a_Thesis\Thesis_ArcGIS\Code\NewModelScripts12.13.25\model_outputs\deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "cliff_watch_model.joblib")
CONFIG_PATH = os.path.join(MODEL_DIR, "cliff_watch_config.json")

# Logging
LOG_DIR = r"C:\Users\MB\Desktop\SSCI 594a_Thesis\Thesis_ArcGIS\CliffWatch12.13.25\logs"
LOG_FILE = None  # Will be set in main()
DRY_RUN = False  # Set True to preview without writing to ArcGIS

# =============================================================================
# NWS (National Weather Service) "CURRENT CONDITIONS" SETTINGS
# =============================================================================
# NWS requires a descriptive User-Agent. Add contact email/URL.
NWS_USER_AGENT = "CliffWatch/1.0 (contact: your_email@example.com)"

# =============================================================================
# RAINFALL SAFETY OVERRIDE THRESHOLDS
# Based on USGS research on Southern California landslide triggers
# Sources:
#   - Campbell, R.H. (1975). USGS Professional Paper 851
#   - USGS OF 2005-1067: La Conchita Landslide Hazards
#   - Ventura County Emergency Services thresholds
#   - USGS: "Rainfall and Landslides in Southern California"
#
# KEY FINDINGS:
# - 1 inch/hour (25mm/hr) is the critical debris flow trigger threshold
# - 10 inches seasonal saturation + burst triggers shallow slides
# - 8 inches in 15 days triggered 2005 La Conchita (10 fatalities)
# =============================================================================

RAINFALL_OVERRIDE = {
    # Tier 1: ELEVATED - Significant rainfall (USGS threshold as caution)
    "elevated_1h_mm": 25.0,       # 25mm (1 in) in 1 hour
    "elevated_24h_mm": 50.0,      # 50mm (2 in) in 24 hours
    
    # Tier 2: HIGH - Intense rainfall
    "high_1h_mm": 38.0,           # 38mm (1.5 in) in 1 hour
    "high_24h_mm": 75.0,          # 75mm (3 in) in 24 hours
    "high_3d_mm": 125.0,          # 125mm (5 in) in 3 days
    
    # Tier 3: CRITICAL - Extreme/La Conchita-level
    "critical_1h_mm": 50.0,       # 50mm (2 in) in 1 hour
    "critical_7d_mm": 200.0,      # 200mm (8 in) in 7 days
}

# =============================================================================
# TEST MODE - Simulate elevated environmental conditions
# =============================================================================
TEST_MODE = False  # Set True to simulate elevated conditions

# Test scenarios (only used when TEST_MODE = True)
TEST_SCENARIO = "rain_high"  # Options: "rain_elevated", "rain_high", "rain_critical", "waves", "earthquake", "combined"

TEST_CONDITIONS = {
    "rain_elevated": {
        "PRCP_1h": 15.0,      # 15mm in 1h (triggers ELEVATED)
        "PRCP": 30.0,         # 30mm in 24h
        "PRCP_3": 45.0,
        "PRCP_7": 60.0,
        "WVHT_mean_3d": 0.5,
        "EQ_ct_7d_50km": 0,
        "EQ_maxM_7d_50km": 0
    },
    "rain_high": {
        "PRCP_1h": 28.0,      # 28mm in 1h (triggers HIGH - exceeds 1 in/hr)
        "PRCP": 55.0,         # 55mm in 24h
        "PRCP_3": 80.0,
        "PRCP_7": 100.0,
        "WVHT_mean_3d": 0.5,
        "EQ_ct_7d_50km": 0,
        "EQ_maxM_7d_50km": 0
    },
    "rain_critical": {
        "PRCP_1h": 45.0,      # 45mm in 1h (triggers CRITICAL)
        "PRCP": 60.0,
        "PRCP_3": 120.0,
        "PRCP_7": 220.0,      # 220mm in 7 days (also triggers CRITICAL)
        "WVHT_mean_3d": 0.5,
        "EQ_ct_7d_50km": 0,
        "EQ_maxM_7d_50km": 0
    },
    "waves": {
        "PRCP_1h": 0.0,
        "PRCP": 0.0,
        "PRCP_3": 0.0,
        "PRCP_7": 0.0,
        "WVHT_mean_3d": 2.5,  # 2.5m waves (triggers warning)
        "EQ_ct_7d_50km": 0,
        "EQ_maxM_7d_50km": 0
    },
    "earthquake": {
        "PRCP_1h": 0.0,
        "PRCP": 0.0,
        "PRCP_3": 0.0,
        "PRCP_7": 0.0,
        "WVHT_mean_3d": 0.5,
        "EQ_ct_7d_50km": 3,
        "EQ_maxM_7d_50km": 4.2  # M4.2 earthquake (triggers warning)
    },
    "combined": {
        "PRCP_1h": 15.0,      # Rain (elevated hourly)
        "PRCP": 35.0,
        "PRCP_3": 60.0,
        "PRCP_7": 90.0,
        "WVHT_mean_3d": 2.0,  # Waves
        "EQ_ct_7d_50km": 2,
        "EQ_maxM_7d_50km": 3.5  # Earthquake
    }
}

# Study Area Bounding Box (La Jolla)
BBOX = {
    "min_lon": -117.2622089,
    "max_lon": -117.2480000,
    "min_lat": 32.8048211,
    "max_lat": 32.9284839
}
CENTER_LAT = (BBOX["min_lat"] + BBOX["max_lat"]) / 2
CENTER_LON = (BBOX["min_lon"] + BBOX["max_lon"]) / 2

# Service URLs
MRMS_IMAGE_URL = "https://mapservices.weather.noaa.gov/raster/rest/services/obs/mrms_qpe/ImageServer"
USGS_SEISMIC_URL = "https://services9.arcgis.com/RHVPKKiFTONKtxq3/arcgis/rest/services/USGS_Seismic_Data_v1/FeatureServer/0"

# CDIP Settings (Scripps Pier - Station 073)
CDIP_ENABLED = True
CDIP_LOOKBACK_DAYS = 45
CDIP_MAX_AGE_HOURS = 12  # Data must be within 12 hours

# NDBC Fallback Stations (ordered by preference)
NDBC_STATIONS = ["46254", "46225", "46273", "46224"]

# Physical constants for wave energy
RHO = 1025.0  # seawater density kg/m³
G = 9.80665   # gravity m/s²

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def now_utc() -> pd.Timestamp:
    """Get current UTC timestamp."""
    return pd.Timestamp.now(tz="UTC")

def now_utc_iso() -> str:
    """Get current UTC time as ISO string."""
    return now_utc().isoformat(timespec="seconds")

def log(msg: str, level: str = "INFO"):
    """Log to both console and file."""
    global LOG_FILE
    timestamp = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    log_line = f"[{timestamp}] [{level}] {msg}"
    print(log_line)
    
    # Write to log file if initialized
    if LOG_FILE:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
        except Exception as e:
            print(f"[WARNING] Could not write to log file: {e}")

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers."""
    R = 6371.0088  # Earth's mean radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def mm_to_inches(mm: float) -> float:
    """Convert millimeters to inches."""
    return mm / 25.4

def classify_risk(prob: float, low_break: float = 0.30, high_break: float = 0.60) -> str:
    """Classify probability into risk category (susceptibility only)."""
    if pd.isna(prob):
        return "Unknown"
    if prob < low_break:
        return "Low"
    if prob < high_break:
        return "Medium"
    return "High"

def check_rainfall_safety_override(live_data: dict) -> Tuple[Optional[str], str]:
    """
    Check if rainfall conditions warrant a safety override regardless of model output.
    Based on USGS Southern California landslide research.
    
    Key thresholds from USGS:
    - 1 inch/hour (25mm/hr) can trigger debris flows (Ventura County Emergency Services)
    - 10 inches seasonal saturation + 0.2-0.25 in/hr burst triggers shallow slides (USGS PP-851)
    - 8 inches in 15 days triggered 2005 La Conchita (USGS OF-2005-1067)
    
    Returns: (override_level, reason)
        override_level: None, "Elevated", "High", or "Critical"
        reason: Human-readable explanation for the override
    """
    prcp_1h = live_data.get("PRCP_1h", 0) or 0
    prcp_24h = live_data.get("PRCP", 0) or 0
    prcp_3d = live_data.get("PRCP_3", 0) or 0
    prcp_7d = live_data.get("PRCP_7", 0) or 0
    
    # Handle NaN values
    if isinstance(prcp_1h, float) and np.isnan(prcp_1h):
        prcp_1h = 0
    if isinstance(prcp_24h, float) and np.isnan(prcp_24h):
        prcp_24h = 0
    if isinstance(prcp_3d, float) and np.isnan(prcp_3d):
        prcp_3d = 0
    if isinstance(prcp_7d, float) and np.isnan(prcp_7d):
        prcp_7d = 0
    
    reasons = []
    override_level = None
    
    # =========================================================================
    # Check CRITICAL thresholds first (highest priority)
    # =========================================================================
    
    # Critical hourly intensity
    if prcp_1h >= RAINFALL_OVERRIDE["critical_1h_mm"]:
        override_level = "Critical"
        reasons.append(
            f"⚠️ CRITICAL INTENSITY: {prcp_1h:.1f}mm ({mm_to_inches(prcp_1h):.2f}in) in past hour. "
            f"Extreme rainfall rate significantly exceeds USGS debris flow trigger threshold."
        )
    
    # Critical 7-day accumulation (La Conchita-level)
    if prcp_7d >= RAINFALL_OVERRIDE["critical_7d_mm"]:
        if override_level != "Critical":
            override_level = "Critical"
        reasons.append(
            f"⚠️ CRITICAL SATURATION: {prcp_7d:.0f}mm ({mm_to_inches(prcp_7d):.1f}in) est. 7-day total. "
            f"Approaching conditions similar to 2005 La Conchita landslide (USGS OF-2005-1067)."
        )
    
    # =========================================================================
    # Check HIGH thresholds
    # =========================================================================
    
    # HIGH hourly intensity - USGS critical threshold
    if prcp_1h >= RAINFALL_OVERRIDE["high_1h_mm"]:
        if override_level not in ["Critical"]:
            override_level = "High"
        reasons.append(
            f"🔴 HIGH INTENSITY: {prcp_1h:.1f}mm ({mm_to_inches(prcp_1h):.2f}in) in past hour. "
            f"Exceeds USGS threshold of 1 inch/hour for debris flow triggering "
            f"(Ventura County Emergency Services, Campbell 1975)."
        )
    
    # HIGH 24-hour accumulation
    if prcp_24h >= RAINFALL_OVERRIDE["high_24h_mm"]:
        if override_level not in ["Critical", "High"]:
            override_level = "High"
        reasons.append(
            f"🔴 HIGH RAINFALL: {prcp_24h:.0f}mm ({mm_to_inches(prcp_24h):.1f}in) in 24 hours. "
            f"Intense storm activity - monitor for continued precipitation."
        )
    
    # HIGH 3-day accumulation
    if prcp_3d >= RAINFALL_OVERRIDE["high_3d_mm"]:
        if override_level not in ["Critical", "High"]:
            override_level = "High"
        reasons.append(
            f"🔴 HIGH ACCUMULATION: {prcp_3d:.0f}mm ({mm_to_inches(prcp_3d):.1f}in) in 3 days. "
            f"Sustained heavy precipitation increases soil saturation."
        )
    
    # =========================================================================
    # Check ELEVATED thresholds
    # =========================================================================
    
    # Elevated hourly intensity
    if prcp_1h >= RAINFALL_OVERRIDE["elevated_1h_mm"] and override_level is None:
        override_level = "Elevated"
        reasons.append(
            f"🟡 ELEVATED INTENSITY: {prcp_1h:.1f}mm ({mm_to_inches(prcp_1h):.2f}in) in past hour. "
            f"Significant rainfall rate detected. Monitor for intensification."
        )
    
    # Elevated 24-hour accumulation
    if prcp_24h >= RAINFALL_OVERRIDE["elevated_24h_mm"] and override_level is None:
        override_level = "Elevated"
        reasons.append(
            f"🟡 ELEVATED RAINFALL: {prcp_24h:.0f}mm ({mm_to_inches(prcp_24h):.1f}in) in 24 hours. "
            f"Monitor conditions closely."
        )
    
    if not reasons:
        return None, ""
    
    return override_level, " | ".join(reasons)


def check_environmental_conditions(live_data: dict) -> Tuple[bool, str, str]:
    """
    Check if environmental conditions are elevated at the regional level.
    Returns (is_elevated, trigger_type, reason)
    
    This is checked ONCE for the entire study area since environmental
    conditions (waves, rain, seismic) are regional.
    
    Now includes rainfall safety override check.
    """
    triggers = []
    reasons = []
    
    # First check rainfall safety override
    rain_override, rain_reason = check_rainfall_safety_override(live_data)
    if rain_override:
        triggers.append("Precipitation")
        reasons.append(rain_reason)
    
    # Get environmental values
    wvht_3d = live_data.get("WVHT_mean_3d", 0) or 0
    if isinstance(wvht_3d, float) and np.isnan(wvht_3d):
        wvht_3d = 0
    
    prcp_7d = live_data.get("PRCP_7", 0) or 0
    if isinstance(prcp_7d, float) and np.isnan(prcp_7d):
        prcp_7d = 0
    
    eq_count = live_data.get("EQ_ct_7d_50km", 0) or 0
    eq_max_mag = live_data.get("EQ_maxM_7d_50km", 0) or 0
    
    # Check wave conditions
    if wvht_3d >= 2.0:
        triggers.append("Waves")
        reasons.append(f"🌊 High wave activity: {wvht_3d:.1f}m significant wave height (3-day avg). Large waves increase cliff-base erosion and undercutting.")
    elif wvht_3d >= 1.0:
        triggers.append("Waves")
        reasons.append(f"🌊 Elevated wave activity: {wvht_3d:.1f}m significant wave height (3-day avg).")
    
    # Check precipitation (only if not already triggered by safety override)
    if "Precipitation" not in triggers:
        if prcp_7d >= 25:
            triggers.append("Precipitation")
            reasons.append(f"🌧️ Significant rainfall: {prcp_7d:.0f}mm ({mm_to_inches(prcp_7d):.1f}in) estimated 7-day total.")
        elif prcp_7d >= 10:
            triggers.append("Precipitation")
            reasons.append(f"🌧️ Recent rainfall: {prcp_7d:.0f}mm ({mm_to_inches(prcp_7d):.1f}in) estimated 7-day total.")
    
    # Check seismic activity
    if eq_max_mag >= 4.0:
        triggers.append("Seismic")
        reasons.append(f"🔴 Significant earthquake: M{eq_max_mag:.1f} within 50km in past 7 days. Seismic shaking can trigger immediate slope failures.")
    elif eq_max_mag >= 3.0:
        triggers.append("Seismic")
        reasons.append(f"🟡 Minor earthquake: M{eq_max_mag:.1f} within 50km in past 7 days.")
    elif eq_count >= 5:
        triggers.append("Seismic")
        reasons.append(f"🟡 Elevated seismic activity: {int(eq_count)} earthquakes within 50km in past 7 days.")
    
    if len(triggers) == 0:
        return False, "None", "No active environmental triggers detected."
    
    trigger_type = triggers[0] if len(triggers) == 1 else "Combined"
    return True, trigger_type, " | ".join(reasons)


def compute_coastal_weight(distance_ft: float) -> float:
    """
    Compute weight for ocean features based on distance from coast.
    - Within 500ft: full weight (1.0)
    - Beyond 500ft: weight decreases linearly to 0 at some max distance
    
    This prevents ocean variables from affecting inland locations.
    """
    if pd.isna(distance_ft):
        return 0.5  # Default for unknown
    
    # Full weight within 500 feet
    if distance_ft <= 500:
        return 1.0
    
    # Linear decay from 500ft to 2000ft
    max_dist = 2000
    if distance_ft >= max_dist:
        return 0.0
    
    return 1.0 - (distance_ft - 500) / (max_dist - 500)


def determine_risk_trigger_and_reason(row, live_data: dict, risk_class: str, 
                                      rainfall_override: Optional[str] = None) -> Tuple[str, str]:
    """
    Generate human-readable risk explanation based on conditions.
    Returns (trigger_type, detailed_reason)
    """
    triggers = []
    reasons = []
    
    prob_slide = row.get("prob_slide", 0) or 0
    prob_change = row.get("prob_change", 0) or 0
    coast_dist = row.get("NEAR_DIST_COAST", 999) or 999
    
    # Get environmental values from live_data
    wvht_3d = live_data.get("WVHT_mean_3d", 0) or 0
    if isinstance(wvht_3d, float) and np.isnan(wvht_3d):
        wvht_3d = 0
    
    prcp_1h = live_data.get("PRCP_1h", 0) or 0
    prcp_24h = live_data.get("PRCP", 0) or 0
    prcp_3d = live_data.get("PRCP_3", 0) or 0
    prcp_7d = live_data.get("PRCP_7", 0) or 0
    
    eq_count = live_data.get("EQ_ct_7d_50km", 0) or 0
    eq_max_mag = live_data.get("EQ_maxM_7d_50km", 0) or 0
    
    # === RAINFALL SAFETY OVERRIDE (highest priority for explanation) ===
    if rainfall_override == "Critical":
        triggers.append("Precipitation")
        if prcp_1h >= RAINFALL_OVERRIDE["critical_1h_mm"]:
            reasons.append(
                f"⚠️ CRITICAL RAINFALL INTENSITY: {prcp_1h:.1f}mm ({mm_to_inches(prcp_1h):.2f}in) "
                f"in past hour. This extreme rate far exceeds the USGS 1 inch/hour debris flow threshold. "
                f"Evacuate cliff areas immediately."
            )
        if prcp_7d >= RAINFALL_OVERRIDE["critical_7d_mm"]:
            reasons.append(
                f"⚠️ CRITICAL SATURATION: {prcp_7d:.0f}mm ({mm_to_inches(prcp_7d):.1f}in) "
                f"estimated 7-day total approaches saturation levels. The 2005 La Conchita landslide occurred "
                f"after ~200mm in 15 days, killing 10 people. Evacuate low-lying cliff areas."
            )
    elif rainfall_override == "High":
        triggers.append("Precipitation")
        if prcp_1h >= RAINFALL_OVERRIDE["high_1h_mm"]:
            reasons.append(
                f"🔴 HIGH RAINFALL INTENSITY: {prcp_1h:.1f}mm ({mm_to_inches(prcp_1h):.2f}in) in past hour. "
                f"Exceeds USGS threshold of 1 inch/hour (25mm/hr) for triggering debris flows. "
                f"Avoid cliff edges and low-lying areas."
            )
        if prcp_24h >= RAINFALL_OVERRIDE["high_24h_mm"]:
            reasons.append(
                f"🔴 HIGH RAINFALL: {prcp_24h:.0f}mm ({mm_to_inches(prcp_24h):.1f}in) in 24 hours. "
                f"Per USGS research, intense rainfall triggers shallow landslides "
                f"in Southern California when ground is saturated. Avoid cliff edges."
            )
        if prcp_3d >= RAINFALL_OVERRIDE["high_3d_mm"]:
            reasons.append(
                f"🔴 SUSTAINED HEAVY RAIN: {prcp_3d:.0f}mm ({mm_to_inches(prcp_3d):.1f}in) estimated 3-day total. "
                f"Prolonged rainfall increases soil saturation and pore water pressure, "
                f"destabilizing slopes."
            )
    elif rainfall_override == "Elevated":
        triggers.append("Precipitation")
        if prcp_1h >= RAINFALL_OVERRIDE["elevated_1h_mm"]:
            reasons.append(
                f"🟡 ELEVATED RAINFALL INTENSITY: {prcp_1h:.1f}mm ({mm_to_inches(prcp_1h):.2f}in) in past hour. "
                f"Monitor for intensification toward the 1 inch/hour critical threshold."
            )
        elif prcp_24h >= RAINFALL_OVERRIDE["elevated_24h_mm"]:
            reasons.append(
                f"🟡 ELEVATED RAINFALL: {prcp_24h:.0f}mm ({mm_to_inches(prcp_24h):.1f}in) in 24 hours. "
                f"Significant storm activity increases landslide potential. Exercise caution near cliff edges."
            )
    
    # === WAVE CONDITIONS (for coastal locations) ===
    if coast_dist <= 500 and wvht_3d >= 1.0:
        triggers.append("Waves")
        if wvht_3d >= 2.0:
            reasons.append(
                f"🌊 HIGH WAVES: {wvht_3d:.1f}m significant wave height (3-day avg). "
                f"Large waves cause cliff-base erosion and undercutting, which can trigger sudden collapses. "
                f"This location is {coast_dist:.0f}ft from the coast."
            )
        else:
            reasons.append(
                f"🌊 Elevated waves: {wvht_3d:.1f}m significant wave height (3-day avg). "
                f"Wave action contributes to cliff erosion at this coastal location ({coast_dist:.0f}ft from coast)."
            )
    
    # === SEISMIC (only if not already covered by other triggers) ===
    if eq_max_mag >= 3.0 or eq_count >= 5:
        triggers.append("Seismic")
        if eq_max_mag >= 4.0:
            reasons.append(
                f"🔴 SIGNIFICANT EARTHQUAKE: M{eq_max_mag:.1f} within 50km in past 7 days. "
                f"Seismic shaking can trigger immediate slope failures and loosen material for future slides."
            )
        elif eq_max_mag >= 3.0:
            reasons.append(
                f"🟡 Minor earthquake: M{eq_max_mag:.1f} within 50km in past 7 days."
            )
        elif eq_count >= 5:
            reasons.append(
                f"🟡 Elevated seismic activity: {int(eq_count)} earthquakes within 50km in past 7 days."
            )
    
    # === LOW RISK EXPLANATION ===
    if len(triggers) == 0:
        if risk_class == "Low":
            return "Baseline", f"✅ Low risk. Normal conditions with no active environmental triggers."
        elif risk_class == "Medium":
            return "Baseline", f"📊 Moderate susceptibility based on terrain and historical patterns. No active triggers."
        else:
            return "Baseline", f"📊 Model indicates elevated susceptibility. Monitor conditions."
    
    # Remove duplicates while preserving order
    unique_triggers = list(dict.fromkeys(triggers))
    
    # Add model context at the beginning for elevated risks
    if risk_class in ["High", "Elevated", "Critical"]:
        model_context = f"📊 Model probability: {prob_slide:.0%}"
        if prob_change >= 0.05:
            model_context += f" (↑{prob_change:.0%} increase from baseline conditions)"
        reasons.insert(0, model_context)
    
    trigger_type = unique_triggers[0] if len(unique_triggers) == 1 else "Combined"
    return trigger_type, " | ".join(reasons)


# =============================================================================
# PRECIPITATION DATA (MRMS with 1-hour and 24-hour)
# =============================================================================

def fetch_precipitation_mrms(gis: GIS) -> Dict[str, float]:
    """
    Fetch precipitation data from NOAA MRMS ImageServer.
    Returns both 1-hour and 24-hour precipitation for La Jolla.
    
    MRMS provides actual accumulated data for 1hr, 3hr, 6hr, 12hr, 24hr, 48hr, 72hr.
    We fetch 1-hour (for intensity monitoring) and 24-hour (for accumulation).
    Multi-day estimates beyond 72hr are calculated by multiplying.
    
    Note: MRMS returns values in INCHES, we convert to mm for consistency.
    """
    log("Fetching MRMS precipitation data...")
    
    result = {
        "PRCP_1h": 0.0,      # 1-hour accumulation (primary for safety override)
        "PRCP": 0.0,         # 24-hour accumulation (for model compatibility)
        "PRCP_3": 0.0,
        "PRCP_7": 0.0,
        "PRCP_14": 0.0,
        "PRCP_30": 0.0,
        "PRCP_missing_flag": 1
    }
    
    try:
        il = ImageryLayer(MRMS_IMAGE_URL, gis=gis)
        
        # Center point for query
        center_pt = {
            "x": CENTER_LON, 
            "y": CENTER_LAT, 
            "spatialReference": {"wkid": 4326}
        }
        
        # --- Fetch 1-HOUR precipitation (for intensity monitoring) ---
        try:
            samples_1h = il.identify(
                geometry=center_pt,
                time_extent=None,
                return_geometry=False,
                return_catalog_items=False,
                rendering_rule={"rasterFunction": "rft_1hr"}
            )
            
            if samples_1h and 'value' in samples_1h:
                val = samples_1h.get('value')
                if val and val != 'NoData':
                    try:
                        prcp_1h_in = float(val)
                        # MRMS returns values in inches, convert to mm
                        prcp_1h_mm = prcp_1h_in * 25.4
                        if prcp_1h_mm >= 0:
                            result["PRCP_1h"] = prcp_1h_mm
                    except (ValueError, TypeError):
                        pass
                        
        except Exception as e:
            log(f"  MRMS 1-hour fetch failed: {e}", "WARN")
        
        # --- Fetch 24-HOUR precipitation (for model and accumulation) ---
        try:
            samples_24h = il.identify(
                geometry=center_pt,
                time_extent=None,
                return_geometry=False,
                return_catalog_items=False,
                rendering_rule={"rasterFunction": "rft_24hr"}
            )
            
            if samples_24h and 'value' in samples_24h:
                val = samples_24h.get('value')
                if val and val != 'NoData':
                    try:
                        prcp_24h_in = float(val)
                        # MRMS returns values in inches, convert to mm
                        prcp_24h_mm = prcp_24h_in * 25.4
                        if prcp_24h_mm >= 0:
                            result["PRCP"] = prcp_24h_mm
                            result["PRCP_missing_flag"] = 0
                    except (ValueError, TypeError):
                        pass
                        
        except Exception as e:
            log(f"  MRMS 24-hour fetch failed: {e}", "WARN")
        
        # --- Fetch 72-HOUR precipitation (actual, not estimated) ---
        try:
            samples_72h = il.identify(
                geometry=center_pt,
                time_extent=None,
                return_geometry=False,
                return_catalog_items=False,
                rendering_rule={"rasterFunction": "rft_72hr"}
            )
            
            if samples_72h and 'value' in samples_72h:
                val = samples_72h.get('value')
                if val and val != 'NoData':
                    try:
                        prcp_72h_in = float(val)
                        prcp_72h_mm = prcp_72h_in * 25.4
                        if prcp_72h_mm >= 0:
                            result["PRCP_3"] = prcp_72h_mm  # Actual 72hr data
                    except (ValueError, TypeError):
                        pass
                        
        except Exception as e:
            # Fall back to estimate if 72hr fails
            if result["PRCP"] > 0:
                result["PRCP_3"] = result["PRCP"] * 3
            log(f"  MRMS 72-hour failed, using estimate", "WARN")
        
        # Estimate longer periods from 24h (no MRMS products beyond 72hr)
        if result["PRCP_3"] == 0.0 and result["PRCP"] > 0:
            result["PRCP_3"] = result["PRCP"] * 3
        result["PRCP_7"] = result["PRCP"] * 7
        result["PRCP_14"] = result["PRCP"] * 14
        result["PRCP_30"] = result["PRCP"] * 30
        
        # Log results
        prcp_1h = result.get("PRCP_1h", 0)
        prcp_24h = result.get("PRCP", 0)
        log(f"  PRCP (1h):  {prcp_1h:.2f} mm ({mm_to_inches(prcp_1h):.2f} in)")
        log(f"  PRCP (24h): {prcp_24h:.2f} mm ({mm_to_inches(prcp_24h):.2f} in)")
        
        if result["PRCP_missing_flag"] == 0:
            log(f"  Note: Multi-day values beyond 72hr are estimates")
        
        # Check and log safety override status
        override_level, _ = check_rainfall_safety_override(result)
        if override_level:
            log(f"  ⚠️ RAINFALL SAFETY OVERRIDE: {override_level}", "WARN")
        
        return result
        
    except Exception as e:
        log(f"MRMS fetch failed: {e}", "ERROR")
        return result


# =============================================================================
# CURRENT PRECIPITATION (NWS SURFACE OBSERVATIONS)
# =============================================================================

def fetch_precip_now_nws(lat: float, lon: float) -> Dict[str, float]:
    """
    Populate `prec_now` as a *current precipitation indicator* (NOT last-hour accumulation).

    Why:
      - NWS station observations do NOT generally provide an instantaneous "mm/hr rain rate" value.
      - They DO provide *current conditions* and "present weather" (e.g., Rain/Drizzle) via
        the latest observation, which is what most "current conditions" apps reflect.

    What we store in prec_now (Double field):
      - 1.0  => precipitation is currently being reported at/near the nearest NWS station
      - 0.0  => no precipitation currently reported
      - None => could not retrieve an observation (keeps field null)

    Detection logic (robust):
      1) Prefer the structured `presentWeather` array (if present)
      2) Fallback: parse METAR-like `rawMessage` for precip tokens (RA, -RA, +RA, DZ, SN, etc.)

    NOTE: If you truly need a numeric intensity (mm/hr), you will need a different feed
          (e.g., radar-derived rain-rate products or a commercial/third-party API).
    """
    log("Fetching NWS current precipitation condition (prec_now)...")

    headers = {
        "User-Agent": NWS_USER_AGENT,
        "Accept": "application/geo+json"
    }

    try:
        # 1) Resolve point -> observationStations URL
        p = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=headers, timeout=20)
        p.raise_for_status()
        points = p.json()
        stations_url = points["properties"]["observationStations"]

        # 2) Get nearby stations, pick the first (typically nearest)
        s = requests.get(stations_url, headers=headers, timeout=20)
        s.raise_for_status()
        stations = s.json().get("features", [])
        if not stations:
            log("  NWS: No stations returned for this point.", "WARN")
            return {"prec_now": None}

        station_id = stations[0]["properties"].get("stationIdentifier")
        if not station_id:
            log("  NWS: Station identifier missing.", "WARN")
            return {"prec_now": None}

        # 3) Latest observation
        o = requests.get(f"https://api.weather.gov/stations/{station_id}/observations/latest",
                         headers=headers, timeout=20)
        o.raise_for_status()
        obs = o.json().get("properties", {})
        ts = obs.get("timestamp", "")

        # ---------------------------------------------------------------------
        # A) Structured presentWeather (best signal for "raining now")
        # ---------------------------------------------------------------------
        present = obs.get("presentWeather", None)
        if isinstance(present, list) and len(present) > 0:
            # Example entries can vary; we do a broad string search across fields
            present_blob = " ".join([json.dumps(x).lower() for x in present])

            precip_keywords = [
                "rain", "drizzle", "freezing rain", "freezing drizzle",
                "snow", "sleet", "ice pellets", "hail", "graupel", "showers"
            ]
            if any(k in present_blob for k in precip_keywords):
                log(f"  NWS: Station {station_id} indicates PRECIPITATION NOW (presentWeather) @ {ts}")
                return {"prec_now": 1.0}
            else:
                log(f"  NWS: Station {station_id} indicates NO precip now (presentWeather) @ {ts}")
                return {"prec_now": 0.0}

        # ---------------------------------------------------------------------
        # B) Fallback: parse raw METAR message for precip tokens
        # ---------------------------------------------------------------------
        raw = (obs.get("rawMessage") or "").upper()

        # METAR precip tokens (broad but safe)
        # -RA / RA / +RA  (rain)
        # -DZ / DZ / +DZ  (drizzle)
        # SN (snow), SG (snow grains), PL (ice pellets), GR/GS (hail/small hail)
        # FZRA (freezing rain), FZDZ (freezing drizzle)
        precip_re = re.compile(r"(?:^|\s)(?:-?\+?)?(FZRA|FZDZ|RA|DZ|SN|SG|PL|GR|GS)(?:\s|$)")
        if precip_re.search(raw):
            log(f"  NWS: Station {station_id} indicates PRECIPITATION NOW (rawMessage) @ {ts}")
            return {"prec_now": 1.0}

        log(f"  NWS: Station {station_id} indicates NO precip now (no presentWeather; rawMessage fallback) @ {ts}")
        return {"prec_now": 0.0}

    except Exception as e:
        log(f"NWS current precip fetch failed: {e}", "WARN")
        return {"prec_now": None}


# =============================================================================
# SEISMIC DATA (USGS via REST API directly)
# =============================================================================

def fetch_seismic_data(gis: GIS) -> Dict[str, float]:
    """
    Fetch seismic activity from USGS Earthquake API directly.
    More reliable than Living Atlas queries.
    """
    log("Fetching seismic data...")
    
    result = {
        "EQ_ct_7d_25km": 0.0,
        "EQ_maxM_7d_25km": 0.0,
        "EQ_ct_7d_50km": 0.0,
        "EQ_maxM_7d_50km": 0.0,
        "EQ_ct_30d_25km": 0.0,
        "EQ_maxM_30d_25km": 0.0,
        "EQ_ct_30d_50km": 0.0,
        "EQ_maxM_30d_50km": 0.0,
        "eq_latest_time_mag": ""
    }
    
    try:
        # Use USGS Earthquake API directly
        end_time = now_utc()
        start_time = end_time - timedelta(days=30)
        
        url = (
            f"https://earthquake.usgs.gov/fdsnws/event/1/query"
            f"?format=geojson"
            f"&starttime={start_time.strftime('%Y-%m-%d')}"
            f"&endtime={end_time.strftime('%Y-%m-%d')}"
            f"&latitude={CENTER_LAT}"
            f"&longitude={CENTER_LON}"
            f"&maxradiuskm=50"
            f"&minmagnitude=0.5"
        )
        
        resp = requests.get(url, timeout=30)
        
        if resp.status_code == 200:
            data = resp.json()
            features = data.get("features", [])
            
            seven_days_ago = end_time - timedelta(days=7)
            latest_eq = None
            
            for f in features:
                props = f.get("properties", {})
                geom = f.get("geometry", {})
                coords = geom.get("coordinates", [])
                
                if len(coords) < 2:
                    continue
                
                eq_lon, eq_lat = coords[0], coords[1]
                mag = props.get("mag", 0) or 0
                eq_time_ms = props.get("time", 0)
                eq_time = pd.Timestamp(eq_time_ms, unit='ms', tz='UTC')
                
                # Calculate distance
                dist_km = haversine_km(CENTER_LAT, CENTER_LON, eq_lat, eq_lon)
                
                # Track latest earthquake
                if latest_eq is None or eq_time > latest_eq[0]:
                    latest_eq = (eq_time, mag)
                
                # 30-day counts
                if dist_km <= 25:
                    result["EQ_ct_30d_25km"] += 1
                    result["EQ_maxM_30d_25km"] = max(result["EQ_maxM_30d_25km"], mag)
                if dist_km <= 50:
                    result["EQ_ct_30d_50km"] += 1
                    result["EQ_maxM_30d_50km"] = max(result["EQ_maxM_30d_50km"], mag)
                
                # 7-day counts
                if eq_time >= seven_days_ago:
                    if dist_km <= 25:
                        result["EQ_ct_7d_25km"] += 1
                        result["EQ_maxM_7d_25km"] = max(result["EQ_maxM_7d_25km"], mag)
                    if dist_km <= 50:
                        result["EQ_ct_7d_50km"] += 1
                        result["EQ_maxM_7d_50km"] = max(result["EQ_maxM_7d_50km"], mag)
            
            if latest_eq:
                result["eq_latest_time_mag"] = f"{latest_eq[0].strftime('%Y-%m-%d %H:%M')} M{latest_eq[1]:.1f}"
            
            log(f"  7-day (50km): {int(result['EQ_ct_7d_50km'])} events, max M{result['EQ_maxM_7d_50km']:.1f}")
            log(f"  30-day (50km): {int(result['EQ_ct_30d_50km'])} events, max M{result['EQ_maxM_30d_50km']:.1f}")
            
    except Exception as e:
        log(f"Seismic fetch failed: {e}", "ERROR")
    
    return result


# =============================================================================
# WAVE DATA (CDIP + NDBC)
# =============================================================================

def fetch_cdip_wave_data() -> Optional[Dict[str, float]]:
    """
    Fetch wave data for CDIP Station 073 using the per-station CENCOOS ERDDAP dataset.

    Why:
      - Your current CDIP call uses wave_agg + &station="073p1", which is returning errors/empty.
      - The per-station dataset edu_ucsd_cdip_073 on CENCOOS ERDDAP works (and is near-real-time).

    Returns a dict compatible with the rest of the updater via compute_wave_aggregates().
    """
    log("Fetching CDIP wave data (Station 073) via CENCOOS ERDDAP...")

    if not CDIP_ENABLED:
        log("  CDIP disabled, skipping")
        return None

    try:
        end_time = pd.Timestamp.now(tz="UTC").to_pydatetime()
        start_time = end_time - timedelta(days=CDIP_LOOKBACK_DAYS)

        base = "https://erddap.cencoos.org/erddap/tabledap/edu_ucsd_cdip_073.csv"
        varlist = ",".join([
            "time",
            "sea_surface_wave_significant_height",
            "sea_surface_wave_period_at_variance_spectral_density_maximum",
            "sea_surface_wave_mean_period",
        ])

        csv_url = (
            f"{base}?{varlist}"
            f"&time>={start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            f"&time<={end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            '&orderBy("time")'
        )

        # Encode constraint operators for requests reliability on Windows/proxies
        safe_url = csv_url.replace(">=", "%3E%3D").replace("<=", "%3C%3D")

        resp = requests.get(safe_url, timeout=60)

        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text), skiprows=[1])

            if len(df) > 0:
                df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
                df = df.dropna(subset=["time"]).sort_values("time")
                # Log CDIP observation timestamp freshness
                latest_obs = df["time"].max()
                now_ts = pd.Timestamp.now(tz="UTC")
                age_hours = (now_ts - latest_obs).total_seconds() / 3600.0 if pd.notna(latest_obs) else float("inf")
                log(f"  CDIP latest observation time: {latest_obs.isoformat()} (age: {age_hours:.2f} hours)")

                # Standardize to the columns compute_wave_aggregates expects
                df["Hs"] = pd.to_numeric(df["sea_surface_wave_significant_height"], errors="coerce")

                # Prefer "period at PSD maximum" (closest analog to DPD/Tp)
                df["Tp"] = pd.to_numeric(
                    df["sea_surface_wave_period_at_variance_spectral_density_maximum"],
                    errors="coerce"
                )

                # If Tp is missing everywhere, fall back to mean period
                if df["Tp"].isna().all():
                    df["Tp"] = pd.to_numeric(df["sea_surface_wave_mean_period"], errors="coerce")

                df_norm = df[["time", "Hs", "Tp"]]

                # Calculate aggregates (uses your existing function)
                result = compute_wave_aggregates(df_norm, "CDIP_CENCOOS_073")

                if result:
                    log(f"  CDIP success: {len(df_norm)} records, latest Hs={result.get('WVHT_mean', 0):.2f}m")
                    return result

        log("  CDIP data unavailable or empty")
        return None

    except Exception as e:
        log(f"  CDIP fetch failed: {e}", "WARN")
        return None


def fetch_ndbc_wave_data() -> Optional[Dict[str, float]]:
    """Fetch wave data from NDBC stations as fallback."""
    log("Fetching NDBC wave data (fallback)...")
    
    for station_id in NDBC_STATIONS:
        try:
            # Try realtime2 first
            url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
            resp = requests.get(url, timeout=30)
            
            if resp.status_code == 200:
                lines = resp.text.strip().split('\n')
                
                if len(lines) > 2:
                    records = []
                    for line in lines[2:]:  # Skip header and units rows
                        parts = line.split()
                        if len(parts) >= 9:
                            try:
                                yr, mo, dy, hr, mn = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                                wvht = float(parts[8]) if parts[8] != 'MM' else np.nan
                                dpd = float(parts[9]) if len(parts) > 9 and parts[9] != 'MM' else np.nan
                                apd = float(parts[10]) if len(parts) > 10 and parts[10] != 'MM' else np.nan
                                
                                dt = pd.Timestamp(year=yr, month=mo, day=dy, hour=hr, minute=mn, tz='UTC').to_pydatetime()
                                records.append({
                                    "time": dt,
                                    "WVHT": wvht,
                                    "DPD": dpd,
                                    "APD": apd
                                })
                            except (ValueError, IndexError):
                                continue
                    
                    if records:
                        df = pd.DataFrame(records)
                        df = df.dropna(subset=["WVHT"])
                        
                        # Log NDBC observation timestamp freshness
                        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
                        latest_obs = df["time"].max()
                        now_ts = pd.Timestamp.now(tz="UTC")
                        age_hours = (now_ts - latest_obs).total_seconds() / 3600.0 if pd.notna(latest_obs) else float("inf")
                        log(f"  NDBC {station_id} latest observation time: {latest_obs.isoformat()} (age: {age_hours:.2f} hours)")



                        if len(df) > 0:
                            result = compute_ndbc_aggregates(df, station_id)
                            if result:
                                log(f"  NDBC {station_id} success: {len(df)} records")
                                return result
            
        except Exception as e:
            log(f"  NDBC {station_id} failed: {e}", "WARN")
            continue
    
    log("  All NDBC stations failed")
    return None


def compute_wave_aggregates(df: pd.DataFrame, source: str) -> Optional[Dict[str, float]]:
    """Compute wave statistics from CDIP data."""
    if df.empty:
        return None
    
    result = {"wave_source": source}
    
    now = pd.Timestamp.now(tz="UTC")
    
    # Current/latest values
    latest = df.iloc[-1]
    result["WVHT_mean"] = float(latest.get("Hs", 0)) if pd.notna(latest.get("Hs")) else 0.0
    result["DPD_mean"] = float(latest.get("Tp", 0)) if pd.notna(latest.get("Tp")) else 0.0
    result["APD_mean"] = result["DPD_mean"]  # CDIP doesn't have APD
    
    # Same-day values
    result["WVHT_mean_1d"] = result["WVHT_mean"]
    result["DPD_mean_1d"] = result["DPD_mean"]
    result["APD_mean_1d"] = result["APD_mean"]
    
    # Time-windowed averages
    for window_days, suffix in [(3, "3d"), (7, "7d"), (14, "14d"), (30, "30d")]:
        cutoff = now - timedelta(days=window_days)
        window_df = df[df["time"] >= cutoff]
        
        if len(window_df) > 0:
            result[f"WVHT_mean_{suffix}"] = float(window_df["Hs"].mean()) if "Hs" in window_df else 0.0
            result[f"DPD_mean_{suffix}"] = float(window_df["Tp"].mean()) if "Tp" in window_df else 0.0
            result[f"APD_mean_{suffix}"] = result[f"DPD_mean_{suffix}"]
            
            # For 14d and 30d windows, compute additional stats
            if suffix in ["14d", "30d"]:
                result[f"Hs_max_sum_{suffix}"] = float(window_df["Hs"].max()) if "Hs" in window_df else 0.0
                result[f"Hs_max_max_{suffix}"] = result[f"Hs_max_sum_{suffix}"]
                result[f"Tp_mean_mean_{suffix}"] = result[f"DPD_mean_{suffix}"]
                
                # Wave energy approximation
                hs_vals = window_df["Hs"].dropna()
                if len(hs_vals) > 0:
                    energy = (RHO * G**2 / 64 / np.pi) * (hs_vals**2).sum()
                    result[f"WaveEnergy_sum_{suffix}"] = float(energy)
                    result[f"log1p_WaveEnergy_sum_{suffix}"] = float(np.log1p(energy))
                else:
                    result[f"WaveEnergy_sum_{suffix}"] = 0.0
                    result[f"log1p_WaveEnergy_sum_{suffix}"] = 0.0
                
                # Peak offset (days since max Hs)
                if len(hs_vals) > 0:
                    peak_idx = window_df["Hs"].idxmax()
                    peak_time = window_df.loc[peak_idx, "time"]
                    result[f"Hs_peak_offset_{suffix}"] = (now - peak_time).total_seconds() / 86400
                else:
                    result[f"Hs_peak_offset_{suffix}"] = window_days
                
                result[f"wave_has_{suffix}"] = 1
        else:
            result[f"WVHT_mean_{suffix}"] = 0.0
            result[f"DPD_mean_{suffix}"] = 0.0
            result[f"APD_mean_{suffix}"] = 0.0
            if suffix in ["14d", "30d"]:
                result[f"Hs_max_sum_{suffix}"] = 0.0
                result[f"Hs_max_max_{suffix}"] = 0.0
                result[f"Tp_mean_mean_{suffix}"] = 0.0
                result[f"WaveEnergy_sum_{suffix}"] = 0.0
                result[f"log1p_WaveEnergy_sum_{suffix}"] = 0.0
                result[f"Hs_peak_offset_{suffix}"] = window_days
                result[f"wave_has_{suffix}"] = 0
    
    return result


def compute_ndbc_aggregates(df: pd.DataFrame, station_id: str) -> Optional[Dict[str, float]]:
    """Compute wave statistics from NDBC data."""
    if df.empty:
        return None
    
    result = {"wave_source": "NDBC", "ndbc_station_used": station_id}
    
    now = pd.Timestamp.now(tz="UTC")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    
    # Current/latest values
    latest = df.iloc[-1]
    result["WVHT_mean"] = float(latest.get("WVHT", 0)) if pd.notna(latest.get("WVHT")) else 0.0
    result["DPD_mean"] = float(latest.get("DPD", 0)) if pd.notna(latest.get("DPD")) else 0.0
    result["APD_mean"] = float(latest.get("APD", 0)) if pd.notna(latest.get("APD")) else result["DPD_mean"]
    
    # Same-day values
    result["WVHT_mean_1d"] = result["WVHT_mean"]
    result["DPD_mean_1d"] = result["DPD_mean"]
    result["APD_mean_1d"] = result["APD_mean"]
    
    # Time-windowed averages
    for window_days, suffix in [(3, "3d"), (7, "7d"), (14, "14d"), (30, "30d")]:
        cutoff = now - timedelta(days=window_days)
        window_df = df[df["time"] >= cutoff]
        
        if len(window_df) > 0:
            result[f"WVHT_mean_{suffix}"] = float(window_df["WVHT"].mean()) if "WVHT" in window_df else 0.0
            result[f"DPD_mean_{suffix}"] = float(window_df["DPD"].mean()) if "DPD" in window_df else 0.0
            result[f"APD_mean_{suffix}"] = float(window_df["APD"].mean()) if "APD" in window_df else result[f"DPD_mean_{suffix}"]
            
            # For 14d and 30d windows, compute additional stats
            if suffix in ["14d", "30d"]:
                result[f"Hs_max_sum_{suffix}"] = float(window_df["WVHT"].max()) if "WVHT" in window_df else 0.0
                result[f"Hs_max_max_{suffix}"] = result[f"Hs_max_sum_{suffix}"]
                result[f"Tp_mean_mean_{suffix}"] = result[f"DPD_mean_{suffix}"]
                
                # Wave energy approximation
                hs_vals = window_df["WVHT"].dropna()
                if len(hs_vals) > 0:
                    energy = (RHO * G**2 / 64 / np.pi) * (hs_vals**2).sum()
                    result[f"WaveEnergy_sum_{suffix}"] = float(energy)
                    result[f"log1p_WaveEnergy_sum_{suffix}"] = float(np.log1p(energy))
                else:
                    result[f"WaveEnergy_sum_{suffix}"] = 0.0
                    result[f"log1p_WaveEnergy_sum_{suffix}"] = 0.0
                
                # Peak offset (days since max Hs)
                if len(hs_vals) > 0:
                    peak_idx = window_df["WVHT"].idxmax()
                    peak_time = window_df.loc[peak_idx, "time"]
                    result[f"Hs_peak_offset_{suffix}"] = (now - peak_time).total_seconds() / 86400
                else:
                    result[f"Hs_peak_offset_{suffix}"] = window_days
                
                result[f"wave_has_{suffix}"] = 1
        else:
            result[f"WVHT_mean_{suffix}"] = 0.0
            result[f"DPD_mean_{suffix}"] = 0.0
            result[f"APD_mean_{suffix}"] = 0.0
            if suffix in ["14d", "30d"]:
                result[f"Hs_max_sum_{suffix}"] = 0.0
                result[f"Hs_max_max_{suffix}"] = 0.0
                result[f"Tp_mean_mean_{suffix}"] = 0.0
                result[f"WaveEnergy_sum_{suffix}"] = 0.0
                result[f"log1p_WaveEnergy_sum_{suffix}"] = 0.0
                result[f"Hs_peak_offset_{suffix}"] = window_days
                result[f"wave_has_{suffix}"] = 0
    
    return result


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main updater function."""
    global LOG_FILE
    
    t0 = time.time()
    
    # Initialize logging
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, f"updater_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}.log")
    
    log("=" * 60)
    log("CLIFF WATCH UPDATER v12.31.25 v4 (Ocean-Enhanced Model)")
    log("=" * 60)
    
    # ==========================================================================
    # CONNECT TO ARCGIS ONLINE
    # ==========================================================================
    log("Connecting to ArcGIS Online...")
    try:
        gis = GIS("home")
        log(f"  Connected as: {gis.users.me.username}")
    except Exception as e:
        log(f"Failed to connect to ArcGIS Online: {e}", "ERROR")
        return
    
    # ==========================================================================
    # LOAD FEATURE LAYER
    # ==========================================================================
    log("Loading feature layer...")
    try:
        item = gis.content.get(POINTS_ITEMID)
        fl = item.layers[0]
        log(f"  Layer: {fl.properties.name}")
        
        # Query all features
        features = fl.query(where="1=1", out_fields="*", return_geometry=False)
        df = features.sdf
        log(f"  Features loaded: {len(df)}")
        
    except Exception as e:
        log(f"Failed to load feature layer: {e}", "ERROR")
        return
    
    # ==========================================================================
    # LOAD MODEL AND CONFIG
    # ==========================================================================
    log("Loading model...")
    try:
        model = joblib_load(MODEL_PATH)
        
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        predictors = config.get("predictors", [])
        optimal_threshold = config.get("optimal_threshold", 0.406)
        model_type = config.get("model_type", "ocean")
        
        log(f"  Model: Ocean_Enhanced ({model_type})")
        log(f"  Predictors: {len(predictors)}")
        log(f"  Optimal threshold (STCV): {optimal_threshold:.3f}")
        
        # Set risk category breaks
        low_break = 0.30
        high_break = 0.60
        log(f"  Risk categories: Low < {low_break}, Medium < {high_break}, High >= {high_break}")
        
    except Exception as e:
        log(f"Failed to load model: {e}", "ERROR")
        return
    
    # ==========================================================================
    # FETCH ENVIRONMENTAL DATA
    # ==========================================================================
    log("-" * 40)
    log("Fetching environmental data...")
    
    # Precipitation (now includes 1-hour data)
    prcp_data = fetch_precipitation_mrms(gis)
    
    # NEW: Current precipitation indicator from NWS (1.0 = precip now, 0.0 = no precip)
    nws_prec = fetch_precip_now_nws(CENTER_LAT, CENTER_LON)
    
    # Seismic
    seismic_data = fetch_seismic_data(gis)
    
    # Wave data (CDIP with NDBC fallback)
    wave_data = fetch_cdip_wave_data()
    if wave_data is None:
        wave_data = fetch_ndbc_wave_data()
    if wave_data is None:
        wave_data = {
            "wave_source": "none",
            "WVHT_mean": 0.0, "DPD_mean": 0.0, "APD_mean": 0.0,
            "Hs_max": 0.0, "Tp_mean": 0.0, "WaveEnergy": 0.0,
            "wave_has_same_day": 0
        }
    
    # Combine all environmental data
    raw_env_data = {**prcp_data, **nws_prec, **seismic_data, **wave_data}
    
    # Apply test conditions if in test mode
    if TEST_MODE and TEST_SCENARIO in TEST_CONDITIONS:
        log(f"  Applying TEST conditions: {TEST_SCENARIO}")
        for key, val in TEST_CONDITIONS[TEST_SCENARIO].items():
            raw_env_data[key] = val
            log(f"    {key} = {val}")
    
    # ==========================================================================
    # CHECK RAINFALL SAFETY OVERRIDE
    # ==========================================================================
    rainfall_override, rainfall_reason = check_rainfall_safety_override(raw_env_data)
    if rainfall_override:
        log(f"  ⚠️ RAINFALL SAFETY OVERRIDE: {rainfall_override}", "WARN")
        log(f"    Reason: {rainfall_reason[:100]}...")
    
    # Check for any active environmental triggers
    has_active_trigger, trigger_type, trigger_reason = check_environmental_conditions(raw_env_data)
    log(f"  Active trigger: {trigger_type}")
    
    # ==========================================================================
    # PREPARE FEATURES FOR MODEL
    # ==========================================================================
    log("-" * 40)
    log("Preparing features...")
    
    try:
        # For each point, merge static terrain with live environmental data
        for col in predictors:
            if col not in df.columns:
                if col in raw_env_data:
                    df[col] = raw_env_data[col]
                else:
                    df[col] = 0.0
        
        # Apply coastal distance weighting for ocean features
        if "NEAR_DIST_COAST" in df.columns:
            df["coastal_weight"] = df["NEAR_DIST_COAST"].apply(compute_coastal_weight)
            
            # Weight ocean features by coastal proximity
            #ocean_features = [c for c in df.columns if any(x in c.lower() for x in 
                           # ["wvht", "dpd", "apd", "hs_", "tp_", "wave", "energy", "ocean"])]
            
            # Weight ocean features by coastal proximity (exclude string fields)
            ocean_features = [c for c in df.columns if any(x in c.lower() for x in 
                            ["wvht", "dpd", "apd", "hs_", "tp_", "wave", "energy", "ocean"])
                            and c not in ["wave_source", "ndbc_station_used"]]
            
            for col in ocean_features:
                if col in df.columns and col != "coastal_weight":
                    df[col] = df[col] * df["coastal_weight"]
        
        # Calculate interaction terms
        if "PRCP_7" in df.columns:
            for suffix in ["3d", "7d", "14d", "30d"]:
                wave_col = f"WaveEnergy_sum_{suffix}"
                rain_col = f"PRCP_{suffix.replace('d', '')}" if suffix != "3d" else "PRCP_3"
                interaction_col = f"OceanxRain_{suffix}"
                
                if wave_col in df.columns and rain_col in df.columns:
                    wave_val = df[wave_col].fillna(0)
                    rain_val = df[rain_col].fillna(0)
                    df[interaction_col] = np.log1p(wave_val) * np.log1p(rain_val)
        
        # Ensure all predictors exist
        X = df[predictors].copy()
        
        # Fill any remaining NaN
        X = X.fillna(0)
        
        log(f"  Feature matrix: {X.shape}")
        
    except Exception as e:
        log(f"Feature preparation failed: {e}", "ERROR")
        traceback.print_exc()
        return
    
    # ==========================================================================
    # RUN MODEL PREDICTIONS
    # ==========================================================================
    log("-" * 40)
    log("Running predictions...")
    
    try:
        # Get probabilities
        proba_current = model.predict_proba(X)[:, 1]
        df["prob_slide"] = proba_current
        
        # Compute baseline (no environmental triggers)
        X_baseline = X.copy()
        baseline_cols = [
            # Precipitation (dynamic)
            "PRCP_1h", "PRCP", "PRCP_3", "PRCP_7", "PRCP_14", "PRCP_30", "PRCP_missing_flag",
            # Waves / ocean (dynamic)
            "WVHT_mean", "WVHT_mean_1d", "WVHT_mean_3d", "WVHT_mean_7d", "WVHT_mean_14d", "WVHT_mean_30d",
            "DPD_mean", "DPD_mean_1d", "DPD_mean_3d", "DPD_mean_7d", "DPD_mean_14d", "DPD_mean_30d",
            "APD_mean", "APD_mean_1d", "APD_mean_3d", "APD_mean_7d", "APD_mean_14d", "APD_mean_30d",
            "Hs_max_sum_14d", "Hs_max_sum_30d", "Hs_max_max_14d", "Hs_max_max_30d",
            "WaveEnergy_sum_14d", "WaveEnergy_sum_30d", "log1p_WaveEnergy_sum_14d", "log1p_WaveEnergy_sum_30d",
            "Tp_mean_mean_14d", "Tp_mean_mean_30d", "Hs_peak_offset_14d", "Hs_peak_offset_30d",
            "wave_has_14d", "wave_has_30d",
            # Ocean-rain interactions (dynamic)
            "OceanxRain_14d", "OceanxRain_30d",
            # Seismic (dynamic)
            "EQ_ct_7d_25km", "EQ_maxM_7d_25km", "EQ_ct_7d_50km", "EQ_maxM_7d_50km",
            "EQ_ct_30d_25km", "EQ_maxM_30d_25km", "EQ_ct_30d_50km", "EQ_maxM_30d_50km"
        ]
        for col in baseline_cols:
            if col in X_baseline.columns:
                X_baseline[col] = 0
        
        proba_baseline = model.predict_proba(X_baseline)[:, 1]
        prob_change = proba_current - proba_baseline
        df["prob_change"] = prob_change
        df["prob_baseline"] = proba_baseline
        
        # ==========================================================================
        # CLASSIFICATION WITH RAINFALL SAFETY OVERRIDE
        # ==========================================================================
        PROB_INCREASE_THRESHOLD = 0.05  # 5% increase required normally
        
        def classify_with_safety_override(prob_current, prob_change, low_break, high_break, 
                                          has_trigger, rainfall_override_level):
            """
            Classify risk with rainfall safety override.
            
            Safety overrides (based on USGS research):
            - Critical rainfall (≥40mm/1h or ≥200mm/7d): Force CRITICAL
            - High rainfall (≥25mm/1h or ≥100mm/24h): Force HIGH
            - Elevated rainfall (≥12.5mm/1h or ≥75mm/24h): Force at least ELEVATED
            """
            if pd.isna(prob_current):
                return "Unknown"
            
            # Apply rainfall safety overrides FIRST
            if rainfall_override_level == "Critical":
                return "Critical"  # Highest level - extreme intensity or La Conchita conditions
            elif rainfall_override_level == "High":
                return "High"  # USGS-documented trigger threshold (1 in/hr)
            elif rainfall_override_level == "Elevated":
                # Elevated rainfall forces at least "Elevated" status
                if prob_current < low_break:
                    return "Elevated"  # Override Low → Elevated
                elif prob_current < high_break:
                    return "Elevated"  # Keep as Elevated (was Medium)
                else:
                    return "High" if has_trigger else "Elevated"
            
            # Normal classification (no rainfall override)
            if prob_current < low_break:
                return "Low"
            if prob_current < high_break:
                return "Medium"
            
            # High probability - check for active trigger
            if has_trigger and prob_change >= PROB_INCREASE_THRESHOLD:
                return "High"
            else:
                return "Elevated"
        
        df["risk_class"] = [
            classify_with_safety_override(pc, ch, low_break, high_break, 
                                          has_active_trigger, rainfall_override)
            for pc, ch in zip(proba_current, prob_change)
        ]
        df["risk_source"] = "model"
        df["last_update_utc"] = now_utc()
        
        # Add rainfall override tracking
        df["rainfall_override"] = rainfall_override if rainfall_override else "none"
        
        # Summary
        risk_counts = df["risk_class"].value_counts()
        log(f"  Predictions complete (with rainfall safety override):")
        log(f"    Critical: {risk_counts.get('Critical', 0)}")
        log(f"    High: {risk_counts.get('High', 0)}")
        log(f"    Elevated: {risk_counts.get('Elevated', 0)}")
        log(f"    Medium: {risk_counts.get('Medium', 0)}")
        log(f"    Low: {risk_counts.get('Low', 0)}")
        
        if rainfall_override:
            log(f"  ⚠️ Rainfall safety override applied: {rainfall_override}")
        
    except Exception as e:
        log(f"Model prediction failed: {e}", "ERROR")
        traceback.print_exc()
        return
    
    # ==========================================================================
    # UPDATE FEATURE LAYER
    # ==========================================================================
    log("-" * 40)
    
    if DRY_RUN:
        log("DRY RUN - Not updating ArcGIS Online")
        os.makedirs(LOG_DIR, exist_ok=True)
        preview_path = os.path.join(LOG_DIR, f"preview_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}.csv")
        df[["OBJECTID", "prob_slide", "risk_class", "risk_source", "rainfall_override"]].to_csv(preview_path, index=False)
        log(f"  Preview saved to: {preview_path}")
    else:
        log("Updating feature layer...")
        
        updates = []
        for idx, row in df.iterrows():
            attrs = {
                "OBJECTID": int(row["OBJECTID"])
            }
            
            # Add prediction outputs
            if pd.notna(row.get("prob_slide")):
                attrs["prob_slide"] = float(row["prob_slide"])
            if row.get("risk_class"):
                attrs["risk_class"] = str(row["risk_class"])
            if row.get("risk_source"):
                attrs["risk_source"] = str(row["risk_source"])
            
            # Calculate and add risk trigger/reason with detailed explanations
            risk_class = str(row.get("risk_class", "Unknown"))
            trigger, reason = determine_risk_trigger_and_reason(row, raw_env_data, risk_class, rainfall_override)
            attrs["risk_trigger"] = trigger
            attrs["risk_reason"] = reason[:1500]  # Truncate if needed for field limit
            
            # Add timestamp
            try:
                attrs["last_update_utc"] = row["last_update_utc"].to_pydatetime()
            except:
                attrs["last_update_utc"] = pd.Timestamp.now(tz="UTC").to_pydatetime()
            
            # Add environmental data fields (now includes PRCP_1h)
            env_fields = [
                "WVHT_mean", "WVHT_mean_1d", "WVHT_mean_3d", "WVHT_mean_7d", 
                "WVHT_mean_14d", "WVHT_mean_30d",
                "DPD_mean", "DPD_mean_1d", "DPD_mean_3d", "DPD_mean_7d",
                "DPD_mean_14d", "DPD_mean_30d",
                "APD_mean", "APD_mean_1d", "APD_mean_3d", "APD_mean_7d",
                "APD_mean_14d", "APD_mean_30d",
                "Hs_max_sum_14d", "Hs_max_sum_30d",
                "Hs_max_max_14d", "Hs_max_max_30d",
                "WaveEnergy_sum_14d", "WaveEnergy_sum_30d",
                "log1p_WaveEnergy_sum_14d", "log1p_WaveEnergy_sum_30d",
                "Tp_mean_mean_14d", "Tp_mean_mean_30d",
                "Hs_peak_offset_14d", "Hs_peak_offset_30d",
                "wave_has_14d", "wave_has_30d",
                "PRCP_1h", "PRCP", "PRCP_3", "PRCP_7", "PRCP_14", "PRCP_30", "PRCP_missing_flag",
                "prec_now",  # NEW: NWS current precip condition (1=yes,0=no,None=unknown)
                "EQ_ct_7d_25km", "EQ_maxM_7d_25km", "EQ_ct_7d_50km", "EQ_maxM_7d_50km",
                "EQ_ct_30d_25km", "EQ_maxM_30d_25km", "EQ_ct_30d_50km", "EQ_maxM_30d_50km",
                "eq_latest_time_mag",
                "OceanxRain_14d", "OceanxRain_30d",
                "wave_source", "ndbc_station_used"
            ]
            
            for field in env_fields:
                if field in raw_env_data:
                    val = raw_env_data[field]
                    if val is not None:
                        if isinstance(val, (int, float, np.integer, np.floating)):
                            if not (isinstance(val, float) and np.isnan(val)):
                                attrs[field] = float(val)
                        else:
                            attrs[field] = str(val)
            
            updates.append({"attributes": attrs})
        
        # Batch update in chunks
        try:
            total = 0
            failures = 0
            chunk_size = 500
            
            for i in range(0, len(updates), chunk_size):
                batch = updates[i:i+chunk_size]
                result = fl.edit_features(updates=batch)
                
                if "updateResults" in result:
                    for r in result["updateResults"]:
                        if r.get("success"):
                            total += 1
                        else:
                            failures += 1
                            if failures <= 3:
                                log(f"  Update error: {r.get('error')}", "WARN")
            
            log(f"  Updated: {total} features, Failed: {failures}")
            
        except Exception as e:
            log(f"Feature update failed: {e}", "ERROR")
    
    # ==========================================================================
    # DONE
    # ==========================================================================
    elapsed = time.time() - t0
    log("=" * 60)
    log(f"Cliff Watch Updater Complete ({elapsed:.1f} seconds)")
    log("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Updater failed with error: {e}", "ERROR")
        traceback.print_exc()
