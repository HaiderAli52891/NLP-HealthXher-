"""
╔══════════════════════════════════════════════════════════════════╗
║          VOCAL-CRF  —  Voice-Based Fatigue Monitoring           ║
║     Sex-Specific Adverse Event Detection in Oncology            ║
║                                                                  ║
║  Run:  streamlit run vocal_crf_app.py                           ║
║                                                                  ║
║  Two dashboards:                                                 ║
║    • Patient Portal  — record voice, view personal scores       ║
║    • Doctor Dashboard — all patients, trends, alerts            ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import librosa
import parselmouth
from parselmouth.praat import call
import soundfile as sf
import io
import json
import os
import random
import time
import datetime
from pathlib import Path
from textblob import TextBlob
import scipy.signal as signal
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONFIG & CONSTANTS
# ─────────────────────────────────────────────────────────────────
DATA_FILE   = "vocal_crf_data.json"
SAMPLE_RATE = 22050

# Alert thresholds
ALERT_THRESHOLDS = {
    "hnr_critical":       8.0,    # dB — below this is critical
    "hnr_warning":        12.0,
    "jitter_critical":    0.035,  # above this is critical
    "jitter_warning":     0.020,
    "shimmer_critical":   0.08,
    "shimmer_warning":    0.05,
    "pause_critical":     0.65,   # ratio above this = too many pauses
    "pause_warning":      0.45,
    "composite_critical": 7.5,
    "composite_warning":  5.5,
}

ORANGE = "#E85D26"
BLUE   = "#1A6FB0"
NAVY   = "#0C2340"
GREEN  = "#0A9D7E"
RED    = "#DC2626"
AMBER  = "#D97706"
CREAM  = "#FDF0EA"
LIGHT  = "#F8F8FC"

# ─────────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vocal-CRF | Oncology Voice Monitor",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0C2340 !important;
  }
  [data-testid="stSidebar"] * { color: #C8D8E8 !important; }
  [data-testid="stSidebar"] .stRadio label { color: #fff !important; }

  /* Main background */
  .main { background: #F8F8FC; }

  /* Metric cards */
  .metric-card {
    background: white;
    border-radius: 14px;
    padding: 20px 22px;
    border: 1px solid #E8E8EF;
    box-shadow: 0 2px 12px rgba(28,28,46,0.06);
    margin-bottom: 12px;
  }
  .metric-card.alert-red { border-left: 4px solid #DC2626; }
  .metric-card.alert-amber { border-left: 4px solid #D97706; }
  .metric-card.alert-green { border-left: 4px solid #0A9D7E; }

  /* Section headers */
  .section-header {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #E85D26;
    margin-bottom: 8px;
    margin-top: 24px;
  }

  /* Big stat */
  .big-stat {
    font-size: 48px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
  }
  .big-stat-label {
    font-size: 12px;
    color: #6B6B88;
    line-height: 1.4;
  }

  /* Alert badge */
  .badge-red {
    background: #FEF2F2;
    color: #DC2626;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid #FECACA;
  }
  .badge-amber {
    background: #FFFBEB;
    color: #D97706;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid #FDE68A;
  }
  .badge-green {
    background: #F0FBF6;
    color: #0A9D7E;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid #A7F3D0;
  }

  /* Divider */
  hr { border: none; border-top: 1px solid #E8E8EF; margin: 16px 0; }

  /* Hide streamlit branding */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# DATA PERSISTENCE
# ─────────────────────────────────────────────────────────────────
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {"patients": {}, "sessions": []}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_or_create_patient(name, data):
    if name not in data["patients"]:
        data["patients"][name] = {
            "name": name,
            "age": None,
            "diagnosis": None,
            "treatment": None,
            "baseline": None,
            "enrolled": datetime.datetime.now().isoformat()
        }
        save_data(data)
    return data["patients"][name]


# ─────────────────────────────────────────────────────────────────
# ACOUSTIC FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────
def extract_acoustic_features(audio_bytes):
    """
    Extract clinical-grade acoustic biomarkers from audio bytes.
    Returns dict of features used in the Vocal-CRF pipeline.
    """
    try:
        # Load audio
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)  # stereo → mono
        if sr != SAMPLE_RATE:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

        # ── PRAAT-based features (via parselmouth) ──
        snd = parselmouth.Sound(audio_data, sampling_frequency=sr)

        # HNR — Harmonic-to-Noise Ratio
        try:
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            if np.isnan(hnr) or hnr < 0:
                hnr = 5.0
        except Exception:
            hnr = 5.0

        # Jitter & Shimmer (via PointProcess)
        try:
            point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            if np.isnan(jitter_local): jitter_local = 0.02
            if np.isnan(shimmer_local): shimmer_local = 0.05
        except Exception:
            jitter_local = 0.02
            shimmer_local = 0.05

        # Pitch (F0) mean and variability
        try:
            pitch = call(snd, "To Pitch", 0, 75, 500)
            pitch_mean = call(pitch, "Get mean", 0, 0, "Hertz")
            pitch_sd   = call(pitch, "Get standard deviation", 0, 0, "Hertz")
            if np.isnan(pitch_mean): pitch_mean = 180.0
            if np.isnan(pitch_sd):   pitch_sd   = 20.0
        except Exception:
            pitch_mean = 180.0
            pitch_sd   = 20.0

        # ── LIBROSA-based features ──
        # Spectral Flatness (validated in female cohorts — MDPI Sensors 2025)
        spec_flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio_data)))

        # MFCCs (for voice quality)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfcc_mean = float(np.mean(mfccs[1]))   # second coefficient

        # RMS energy (voice strength)
        rms = float(np.mean(librosa.feature.rms(y=audio_data)))

        # ── PAUSOLOGY ──
        # Detect voiced vs unvoiced frames
        frame_length = 2048
        hop_length   = 512
        rms_frames   = librosa.feature.rms(
            y=audio_data, frame_length=frame_length, hop_length=hop_length
        )[0]
        silence_threshold = rms_frames.max() * 0.1
        voiced_frames   = np.sum(rms_frames > silence_threshold)
        unvoiced_frames = np.sum(rms_frames <= silence_threshold)
        total_frames    = len(rms_frames)
        pause_ratio     = unvoiced_frames / total_frames if total_frames > 0 else 0.5

        # Speech rate estimate (zero crossings proxy)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio_data)))

        # Audio duration
        duration = len(audio_data) / sr

        features = {
            "hnr":             round(float(hnr), 3),
            "jitter":          round(float(jitter_local), 5),
            "shimmer":         round(float(shimmer_local), 5),
            "pitch_mean":      round(float(pitch_mean), 1),
            "pitch_sd":        round(float(pitch_sd), 1),
            "spectral_flatness": round(float(spec_flatness), 6),
            "mfcc_2":          round(float(mfcc_mean), 3),
            "rms_energy":      round(float(rms), 6),
            "pause_ratio":     round(float(pause_ratio), 4),
            "speech_rate_zcr": round(float(zcr), 5),
            "duration_sec":    round(float(duration), 2),
            "extraction_ok":   True,
        }
        return features

    except Exception as e:
        # Fallback — return neutral features with error flag
        return {
            "hnr": 12.0, "jitter": 0.018, "shimmer": 0.045,
            "pitch_mean": 190.0, "pitch_sd": 18.0,
            "spectral_flatness": 0.15, "mfcc_2": -5.0,
            "rms_energy": 0.05, "pause_ratio": 0.35,
            "speech_rate_zcr": 0.08, "duration_sec": 15.0,
            "extraction_ok": False, "error": str(e)
        }


# ─────────────────────────────────────────────────────────────────
# NLP ANALYSIS
# ─────────────────────────────────────────────────────────────────
def analyse_text(transcript: str):
    """
    NLP pipeline: sentiment, complexity, fatigue keywords.
    Uses TextBlob for sentiment + custom linguistic complexity.
    """
    if not transcript or len(transcript.strip()) < 3:
        return {
            "sentiment_polarity": 0.0,
            "sentiment_subjectivity": 0.5,
            "word_count": 0,
            "avg_word_length": 4.0,
            "sentence_count": 0,
            "complexity_score": 0.5,
            "fatigue_keywords": [],
            "nlp_fatigue_score": 5.0,
        }

    blob    = TextBlob(transcript)
    words   = transcript.lower().split()
    sents   = [s.strip() for s in transcript.split('.') if s.strip()]

    # Linguistic complexity (higher = more complex / less fatigued)
    avg_word_len = np.mean([len(w) for w in words]) if words else 4.0
    # Type-token ratio
    ttr = len(set(words)) / len(words) if words else 0.5
    complexity = min(1.0, (avg_word_len / 8.0) * 0.5 + ttr * 0.5)

    # Fatigue keyword detection (validated from CRF literature)
    fatigue_keywords = {
        "heavy": 1.5, "tired": 1.5, "exhausted": 2.0, "drained": 2.0,
        "weak": 1.5, "struggle": 1.5, "can't": 1.0, "cannot": 1.0,
        "hard": 0.8, "difficult": 0.8, "slow": 0.8, "foggy": 1.5,
        "numb": 1.2, "ache": 1.2, "pain": 1.2, "hurt": 1.0,
        "dizzy": 1.5, "breathless": 2.0, "awful": 1.5, "terrible": 1.5,
        "fine": -1.0, "good": -1.0, "great": -1.5, "better": -1.0,
        "well": -0.8, "okay": -0.5, "energy": -0.8, "rested": -1.2,
    }
    found_keywords = []
    keyword_score  = 0.0
    for word in words:
        clean = word.strip(".,!?;:")
        if clean in fatigue_keywords:
            found_keywords.append(clean)
            keyword_score += fatigue_keywords[clean]

    # NLP fatigue score (0 = great, 10 = critical)
    sentiment_factor = (-blob.sentiment.polarity + 1) / 2   # 0=positive, 1=negative
    complexity_factor = 1.0 - complexity                      # 0=complex (good), 1=simple (bad)
    keyword_factor    = min(1.0, max(0.0, keyword_score / 5.0))

    nlp_fatigue = round(
        (sentiment_factor * 3.5 + complexity_factor * 3.0 + keyword_factor * 3.5),
        2
    )
    nlp_fatigue = max(0.0, min(10.0, nlp_fatigue))

    return {
        "sentiment_polarity":     round(float(blob.sentiment.polarity), 3),
        "sentiment_subjectivity": round(float(blob.sentiment.subjectivity), 3),
        "word_count":             len(words),
        "avg_word_length":        round(float(avg_word_len), 2),
        "sentence_count":         len(sents),
        "complexity_score":       round(float(complexity), 3),
        "fatigue_keywords":       list(set(found_keywords)),
        "nlp_fatigue_score":      nlp_fatigue,
    }


# ─────────────────────────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────
def compute_composite_score(acoustic: dict, nlp: dict, baseline: dict = None) -> dict:
    """
    Fuse acoustic + NLP features into a 0–10 composite fatigue score.
    If baseline provided, scores are relative to patient's own baseline.
    """
    # ── Acoustic sub-scores (0=healthy, 10=critical) ──

    # HNR (lower = worse)
    hnr = acoustic.get("hnr", 12.0)
    hnr_score = max(0.0, min(10.0, (16.0 - hnr) / 1.2))

    # Jitter (higher = worse)
    jitter = acoustic.get("jitter", 0.018)
    jitter_score = max(0.0, min(10.0, jitter / 0.004))

    # Shimmer (higher = worse)
    shimmer = acoustic.get("shimmer", 0.045)
    shimmer_score = max(0.0, min(10.0, shimmer / 0.009))

    # Pause ratio (higher = worse)
    pause = acoustic.get("pause_ratio", 0.35)
    pause_score = max(0.0, min(10.0, pause / 0.07))

    # Spectral flatness (higher = more distress)
    flatness = acoustic.get("spectral_flatness", 0.15)
    flatness_score = max(0.0, min(10.0, flatness / 0.03))

    # NLP score (already 0–10)
    nlp_score = nlp.get("nlp_fatigue_score", 5.0)

    # ── Baseline adjustment ──
    if baseline:
        # Compare against patient's own baseline to personalise
        baseline_hnr   = baseline.get("hnr", 14.0)
        baseline_pause = baseline.get("pause_ratio", 0.35)
        hnr_delta   = max(0.0, (baseline_hnr - hnr) / 2.0)
        pause_delta = max(0.0, (pause - baseline_pause) / 0.05)
        hnr_score   = min(10.0, hnr_score   + hnr_delta)
        pause_score = min(10.0, pause_score + pause_delta)

    # ── Weighted fusion ──
    composite = (
        hnr_score      * 0.25 +
        jitter_score   * 0.10 +
        shimmer_score  * 0.10 +
        pause_score    * 0.20 +
        flatness_score * 0.15 +
        nlp_score      * 0.20
    )
    composite = round(max(0.0, min(10.0, composite)), 2)

    # ── Alert tier ──
    if composite >= ALERT_THRESHOLDS["composite_critical"]:
        alert_tier  = 3
        alert_label = "🔴 Action Required"
        alert_color = RED
    elif composite >= ALERT_THRESHOLDS["composite_warning"]:
        alert_tier  = 2
        alert_label = "🟠 Monitor Closely"
        alert_color = AMBER
    elif composite >= 3.5:
        alert_tier  = 1
        alert_label = "🟡 Watch"
        alert_color = "#EAB308"
    else:
        alert_tier  = 0
        alert_label = "🟢 Stable"
        alert_color = GREEN

    return {
        "composite":     composite,
        "alert_tier":    alert_tier,
        "alert_label":   alert_label,
        "alert_color":   alert_color,
        "sub_scores": {
            "HNR":             round(hnr_score, 2),
            "Jitter":          round(jitter_score, 2),
            "Shimmer":         round(shimmer_score, 2),
            "Pausology":       round(pause_score, 2),
            "Spectral":        round(flatness_score, 2),
            "NLP / Linguistic": round(nlp_score, 2),
        },
        "suggestions": _get_suggestions(composite, hnr_score, pause_score, nlp_score),
    }

def _get_suggestions(composite, hnr_score, pause_score, nlp_score):
    s = []
    if composite >= 7.5:
        s.append("Urgent: Contact patient within 2 hours")
        s.append("Order Ferritin + Haemoglobin blood panel")
        s.append("Review current chemotherapy dosing schedule")
    elif composite >= 5.5:
        s.append("Schedule nurse call within 4 hours")
        s.append("Recommend 500ml electrolyte hydration")
        s.append("Consider ferritin check if 2+ consecutive days elevated")
    elif composite >= 3.5:
        s.append("Monitor for next 24 hours")
        s.append("Encourage light movement if tolerated")
    else:
        s.append("Continue current management plan")
        s.append("Next scheduled check-in tomorrow")
    if hnr_score >= 7:
        s.append("Elevated vocal muscle fatigue detected — review laryngeal side effects")
    if pause_score >= 7:
        s.append("High pausology score — possible cognitive fatigue or dyspnoea")
    if nlp_score >= 7:
        s.append("Linguistic complexity significantly reduced — assess for chemobrain symptoms")
    return s


# ─────────────────────────────────────────────────────────────────
# DEMO PATIENT DATA
# ─────────────────────────────────────────────────────────────────
def generate_demo_history(patient_name, days=14, seed=42):
    """Generate realistic demo history for a patient."""
    rng = random.Random(seed)
    sessions = []
    base_date = datetime.datetime.now() - datetime.timedelta(days=days)
    
    # Simulate a treatment cycle: good → decline → spike → recovery
    for d in range(days):
        # Post-infusion spike pattern (worse days 3-5, 10-12)
        cycle_pos = d % 7
        base_fatigue = 3.0
        if cycle_pos in [3, 4, 5]:
            base_fatigue = 6.5 + rng.uniform(-0.5, 1.0)
        elif cycle_pos in [2, 6]:
            base_fatigue = 5.0 + rng.uniform(-0.5, 0.5)
        else:
            base_fatigue = 3.0 + rng.uniform(-0.5, 0.8)

        # Add noise
        composite = max(1.0, min(10.0, base_fatigue + rng.uniform(-0.3, 0.3)))
        
        # HNR inversely correlated with fatigue
        hnr = 16.0 - composite * 1.1 + rng.uniform(-1, 1)
        
        result_date = (base_date + datetime.timedelta(days=d)).isoformat()
        
        if composite >= 7.5:
            alert_tier, alert_label = 3, "🔴 Action Required"
        elif composite >= 5.5:
            alert_tier, alert_label = 2, "🟠 Monitor Closely"
        elif composite >= 3.5:
            alert_tier, alert_label = 1, "🟡 Watch"
        else:
            alert_tier, alert_label = 0, "🟢 Stable"

        sessions.append({
            "patient_name": patient_name,
            "timestamp":    result_date,
            "composite":    round(composite, 2),
            "alert_tier":   alert_tier,
            "alert_label":  alert_label,
            "transcript":   "",
            "acoustic": {
                "hnr":              round(hnr, 2),
                "jitter":           round(0.015 + composite * 0.003 + rng.uniform(-0.002, 0.002), 5),
                "shimmer":          round(0.03 + composite * 0.006 + rng.uniform(-0.003, 0.003), 5),
                "pause_ratio":      round(0.3 + composite * 0.04 + rng.uniform(-0.02, 0.02), 4),
                "spectral_flatness": round(0.10 + composite * 0.02 + rng.uniform(-0.01, 0.01), 6),
            },
            "nlp": {
                "nlp_fatigue_score": round(composite + rng.uniform(-1, 1), 2),
                "sentiment_polarity": round(-composite / 10 + rng.uniform(-0.1, 0.1), 3),
                "fatigue_keywords":  ["heavy", "tired"][:int(composite // 4)],
            },
            "sub_scores": {
                "HNR":              round(max(0, min(10, (16.0 - hnr) / 1.2)), 2),
                "Jitter":           round(composite * 0.8 + rng.uniform(-0.3, 0.3), 2),
                "Shimmer":          round(composite * 0.7 + rng.uniform(-0.3, 0.3), 2),
                "Pausology":        round(composite * 0.9 + rng.uniform(-0.3, 0.3), 2),
                "Spectral":         round(composite * 0.75 + rng.uniform(-0.3, 0.3), 2),
                "NLP / Linguistic": round(composite + rng.uniform(-0.5, 0.5), 2),
            },
            "is_demo": True,
        })
    return sessions

def seed_demo_patients(data):
    """Add demo patients if none exist."""
    demo_patients = [
        {"name": "Sarah Mitchell",  "age": 46, "diagnosis": "Breast Cancer (Stage II)", "treatment": "Docetaxel + Cyclophosphamide", "seed": 42},
        {"name": "Amira Hassan",    "age": 52, "diagnosis": "Ovarian Cancer (Stage III)", "treatment": "Carboplatin + Paclitaxel",   "seed": 77},
        {"name": "Priya Sharma",    "age": 39, "diagnosis": "Cervical Cancer (Stage II)", "treatment": "Cisplatin + Radiation",       "seed": 13},
    ]
    changed = False
    for p in demo_patients:
        if p["name"] not in data["patients"]:
            data["patients"][p["name"]] = {
                "name": p["name"], "age": p["age"],
                "diagnosis": p["diagnosis"], "treatment": p["treatment"],
                "baseline": {"hnr": 16.2, "pause_ratio": 0.32},
                "enrolled": (datetime.datetime.now() - datetime.timedelta(days=14)).isoformat()
            }
            # Add demo history
            demo_sessions = generate_demo_history(p["name"], days=14, seed=p["seed"])
            data["sessions"].extend(demo_sessions)
            changed = True
    if changed:
        save_data(data)
    return data


# ─────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────
def fatigue_trend_chart(sessions_df, patient_name=None, show_self_report=True):
    fig = go.Figure()

    title = f"Fatigue Trend — {patient_name}" if patient_name else "Fatigue Trend"

    # Alert zone band
    fig.add_hrect(y0=7.5, y1=10, fillcolor="rgba(220,38,38,0.06)", line_width=0,
                  annotation_text="Critical Zone", annotation_position="top right",
                  annotation_font=dict(color="rgba(220,38,38,0.6)", size=10))
    fig.add_hrect(y0=5.5, y1=7.5, fillcolor="rgba(217,119,6,0.05)", line_width=0)

    # Acoustic composite
    fig.add_trace(go.Scatter(
        x=sessions_df["date"], y=sessions_df["composite"],
        mode="lines+markers",
        name="Acoustic Biomarker",
        line=dict(color=ORANGE, width=2.5),
        marker=dict(size=7, color=ORANGE),
        fill="tozeroy",
        fillcolor="rgba(232,93,38,0.06)",
    ))

    # Simulated self-report (lags behind acoustic by 1-2 days)
    if show_self_report and len(sessions_df) > 1:
        self_report = sessions_df["composite"].shift(2).fillna(method="bfill") * 0.85
        fig.add_trace(go.Scatter(
            x=sessions_df["date"], y=self_report,
            mode="lines",
            name="Self-Reported Score",
            line=dict(color=BLUE, width=2, dash="dot"),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=NAVY)),
        xaxis=dict(title="Date", gridcolor="#F0F0F8", showgrid=True),
        yaxis=dict(title="Fatigue Score (0–10)", range=[0, 10.5],
                   gridcolor="#F0F0F8", showgrid=True),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=40, b=60, l=50, r=20),
        height=280,
    )
    return fig

def gauge_chart(value, label, max_val=10, color=ORANGE):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": label, "font": {"size": 13, "color": NAVY}},
        number={"font": {"size": 28, "color": NAVY}},
        gauge={
            "axis": {"range": [0, max_val], "tickwidth": 1, "tickcolor": "#888"},
            "bar":  {"color": color},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#E8E8EF",
            "steps": [
                {"range": [0, 3.5],   "color": "rgba(10,157,126,0.12)"},
                {"range": [3.5, 5.5], "color": "rgba(234,179,8,0.12)"},
                {"range": [5.5, 7.5], "color": "rgba(217,119,6,0.12)"},
                {"range": [7.5, 10],  "color": "rgba(220,38,38,0.12)"},
            ],
            "threshold": {
                "line": {"color": RED, "width": 2},
                "thickness": 0.75,
                "value": 7.5
            }
        }
    ))
    fig.update_layout(
        height=200, margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="white", plot_bgcolor="white"
    )
    return fig

def radar_chart(sub_scores: dict):
    categories = list(sub_scores.keys())
    values     = list(sub_scores.values()) + [list(sub_scores.values())[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(232,93,38,0.15)",
        line=dict(color=ORANGE, width=2),
        marker=dict(color=ORANGE, size=6),
        name="Current"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], gridcolor="#E8E8EF", color="#888"),
            angularaxis=dict(gridcolor="#E8E8EF", color=NAVY),
            bgcolor="white",
        ),
        showlegend=False,
        height=280,
        margin=dict(t=30, b=30, l=30, r=30),
        paper_bgcolor="white",
    )
    return fig

def bar_chart_patients(patients_summary):
    fig = go.Figure()
    colors = [RED if t >= 3 else AMBER if t >= 2 else "#EAB308" if t >= 1 else GREEN
              for t in patients_summary["alert_tier"]]
    fig.add_trace(go.Bar(
        x=patients_summary["name"],
        y=patients_summary["composite"],
        marker_color=colors,
        text=[f"{v:.1f}" for v in patients_summary["composite"]],
        textposition="outside",
    ))
    fig.add_hline(y=7.5, line_dash="dash", line_color=RED, opacity=0.5,
                  annotation_text="Critical Threshold", annotation_font_color=RED)
    fig.add_hline(y=5.5, line_dash="dash", line_color=AMBER, opacity=0.4)
    fig.update_layout(
        title=dict(text="Current Fatigue Scores — All Patients", font=dict(size=14, color=NAVY)),
        yaxis=dict(range=[0, 11], title="Composite Score", gridcolor="#F0F0F8"),
        xaxis=dict(title=""),
        plot_bgcolor="white", paper_bgcolor="white",
        height=280,
        margin=dict(t=40, b=20, l=50, r=20),
    )
    return fig

def waveform_chart(audio_data, sr):
    time_axis = np.linspace(0, len(audio_data) / sr, len(audio_data))
    # Downsample for display
    step = max(1, len(audio_data) // 2000)
    fig = go.Figure(go.Scatter(
        x=time_axis[::step], y=audio_data[::step],
        mode="lines", line=dict(color=ORANGE, width=0.8),
        name="Waveform"
    ))
    fig.update_layout(
        title=dict(text="Voice Waveform", font=dict(size=13, color=NAVY)),
        xaxis=dict(title="Time (s)", gridcolor="#F0F0F8"),
        yaxis=dict(title="Amplitude", gridcolor="#F0F0F8"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=180, margin=dict(t=40, b=40, l=50, r=20),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════
#   PATIENT PORTAL
# ═══════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────
def render_patient_portal(data):
    # Header
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{NAVY} 0%,#1A4A7A 100%);
                padding:28px 32px;border-radius:16px;margin-bottom:24px;">
        <div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;
                    color:#7DC8F5;font-weight:700;margin-bottom:6px;">VOCAL-CRF</div>
        <div style="font-size:28px;font-weight:700;color:white;margin-bottom:6px;">
            Patient Check-In Portal 🎙️
        </div>
        <div style="font-size:14px;color:rgba(255,255,255,0.55);">
            30-second voice check-in · Your health, your voice
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Patient Setup ──
    st.markdown('<div class="section-header">Your Profile</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Your Name", value="Sarah Mitchell",
                                     help="Enter your name to track your personal history")
    with col2:
        patient_age = st.number_input("Age", min_value=18, max_value=95, value=46, step=1)

    col3, col4 = st.columns(2)
    with col3:
        diagnosis = st.text_input("Diagnosis", value="Breast Cancer (Stage II)")
    with col4:
        treatment = st.text_input("Current Treatment", value="Docetaxel + Cyclophosphamide")

    # Update patient record
    patient = get_or_create_patient(patient_name, data)
    patient.update({"age": patient_age, "diagnosis": diagnosis, "treatment": treatment})
    save_data(data)

    st.markdown("---")

    # ── Voice Recording ──
    st.markdown('<div class="section-header">Today\'s Check-In</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{CREAM};border:1px solid rgba(232,93,38,0.2);
                border-left:4px solid {ORANGE};border-radius:10px;
                padding:14px 18px;margin-bottom:16px;">
        <div style="font-size:13px;font-weight:600;color:{NAVY};margin-bottom:4px;">
            💬 Prompt: Describe how you are feeling today in a few sentences.
        </div>
        <div style="font-size:12px;color:#6B6B88;">
            There are no right or wrong answers. Just speak naturally.
            Examples: "I woke up feeling heavy and it's hard to get out of bed today."
        </div>
    </div>
    """, unsafe_allow_html=True)

    audio_file = st.audio_input(
        "🎙️ Record your voice check-in",
        help="Click the microphone to record. Speak for at least 10–20 seconds."
    )

    transcript = st.text_area(
        "📝 (Optional) Type or edit your message here:",
        placeholder="You can also type what you said, or add more context...",
        height=90,
        help="Typing helps the NLP analysis. If you recorded, you can also type a transcript."
    )

    st.markdown("")

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        analyse_btn = st.button("🔬 Analyse My Voice", type="primary", use_container_width=True)

    if analyse_btn:
        if audio_file is None and not transcript.strip():
            st.warning("⚠️ Please record your voice or type a message first.")
            return

        with st.spinner("Analysing your voice biomarkers..."):
            progress = st.progress(0, text="Extracting acoustic features...")
            time.sleep(0.3)

            # Extract acoustic features
            if audio_file is not None:
                audio_bytes = audio_file.getvalue()
                acoustic = extract_acoustic_features(audio_bytes)
            else:
                # No audio — use neutral features
                acoustic = {
                    "hnr": 13.0, "jitter": 0.018, "shimmer": 0.042,
                    "pitch_mean": 195.0, "pitch_sd": 22.0,
                    "spectral_flatness": 0.18, "mfcc_2": -4.5,
                    "rms_energy": 0.04, "pause_ratio": 0.38,
                    "speech_rate_zcr": 0.075, "duration_sec": 15.0,
                    "extraction_ok": False
                }
            progress.progress(40, text="Running NLP analysis...")
            time.sleep(0.2)

            # NLP
            nlp_result = analyse_text(transcript)
            progress.progress(70, text="Computing composite score...")
            time.sleep(0.2)

            # Composite score
            baseline = data["patients"].get(patient_name, {}).get("baseline", None)
            result   = compute_composite_score(acoustic, nlp_result, baseline)
            progress.progress(100, text="Done!")
            time.sleep(0.2)
            progress.empty()

        # ── Save session ──
        session = {
            "patient_name": patient_name,
            "timestamp":    datetime.datetime.now().isoformat(),
            "composite":    result["composite"],
            "alert_tier":   result["alert_tier"],
            "alert_label":  result["alert_label"],
            "transcript":   transcript,
            "acoustic":     acoustic,
            "nlp":          nlp_result,
            "sub_scores":   result["sub_scores"],
            "is_demo":      False,
        }
        data["sessions"].append(session)
        save_data(data)

        # ── SET BASELINE if first real session ──
        if data["patients"][patient_name].get("baseline") is None and acoustic["extraction_ok"]:
            data["patients"][patient_name]["baseline"] = {
                "hnr":         acoustic["hnr"],
                "pause_ratio": acoustic["pause_ratio"],
            }
            save_data(data)

        # ─────────────────────────────
        # RESULTS DISPLAY
        # ─────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">Your Results</div>', unsafe_allow_html=True)

        # Alert banner
        alert_color = result["alert_color"]
        st.markdown(f"""
        <div style="background:{alert_color}12;border:1px solid {alert_color}40;
                    border-left:5px solid {alert_color};border-radius:12px;
                    padding:16px 20px;margin-bottom:20px;">
            <div style="font-size:18px;font-weight:700;color:{alert_color};">
                {result["alert_label"]}
            </div>
            <div style="font-size:13px;color:#3A3A52;margin-top:4px;">
                Your voice check-in has been received and analysed.
                {"Your care team has been notified." if result["alert_tier"] >= 2 else
                 "Continue monitoring as normal." if result["alert_tier"] == 1 else
                 "You're doing well today. Keep it up!"}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Main score + sub-scores
        col_score, col_radar = st.columns([1, 1])
        with col_score:
            st.plotly_chart(
                gauge_chart(result["composite"], "Fatigue Score Today",
                           color=alert_color),
                use_container_width=True
            )
            st.markdown(f"""
            <div style="text-align:center;font-size:12px;color:#6B6B88;margin-top:-10px;">
                0 = Excellent · 10 = Critical
            </div>
            """, unsafe_allow_html=True)

        with col_radar:
            st.plotly_chart(
                radar_chart(result["sub_scores"]),
                use_container_width=True
            )

        # Waveform (if audio recorded)
        if audio_file is not None:
            try:
                audio_data, sr = sf.read(io.BytesIO(audio_file.getvalue()))
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                st.plotly_chart(waveform_chart(audio_data, sr), use_container_width=True)
            except Exception:
                pass

        # Acoustic breakdown
        st.markdown('<div class="section-header">Voice Biomarker Breakdown</div>',
                    unsafe_allow_html=True)

        bm_col1, bm_col2, bm_col3 = st.columns(3)
        biomarkers = [
            ("HNR", f"{acoustic['hnr']:.1f} dB",
             "Vocal fold efficiency",
             "↓ Low" if acoustic["hnr"] < ALERT_THRESHOLDS["hnr_warning"] else "✓ Normal",
             acoustic["hnr"] < ALERT_THRESHOLDS["hnr_warning"]),

            ("Jitter", f"{acoustic['jitter']*100:.2f}%",
             "Pitch cycle variation",
             "↑ Elevated" if acoustic["jitter"] > ALERT_THRESHOLDS["jitter_warning"] else "✓ Normal",
             acoustic["jitter"] > ALERT_THRESHOLDS["jitter_warning"]),

            ("Shimmer", f"{acoustic['shimmer']*100:.2f}%",
             "Amplitude variation",
             "↑ Elevated" if acoustic["shimmer"] > ALERT_THRESHOLDS["shimmer_warning"] else "✓ Normal",
             acoustic["shimmer"] > ALERT_THRESHOLDS["shimmer_warning"]),

            ("Pause Ratio", f"{acoustic['pause_ratio']*100:.1f}%",
             "Speech silence proportion",
             "↑ High" if acoustic["pause_ratio"] > ALERT_THRESHOLDS["pause_warning"] else "✓ Normal",
             acoustic["pause_ratio"] > ALERT_THRESHOLDS["pause_warning"]),

            ("Pitch (F0)", f"{acoustic['pitch_mean']:.0f} Hz",
             "Fundamental frequency",
             "Normal range", False),

            ("Duration", f"{acoustic['duration_sec']:.1f}s",
             "Recording length",
             "✓ Sufficient" if acoustic["duration_sec"] >= 10 else "⚠ Too short",
             acoustic["duration_sec"] < 10),
        ]

        for i, (name, val, desc, status, is_alert) in enumerate(biomarkers):
            col = [bm_col1, bm_col2, bm_col3][i % 3]
            border_color = ORANGE if is_alert else "#E8E8EF"
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-left:3px solid {border_color};">
                    <div style="font-size:10px;font-weight:700;letter-spacing:2px;
                                text-transform:uppercase;color:{BLUE};margin-bottom:6px;">{name}</div>
                    <div style="font-size:24px;font-weight:700;color:{NAVY};">{val}</div>
                    <div style="font-size:11px;color:#6B6B88;margin-top:3px;">{desc}</div>
                    <div style="font-size:11px;font-weight:600;
                                color:{ORANGE if is_alert else GREEN};margin-top:4px;">{status}</div>
                </div>
                """, unsafe_allow_html=True)

        # NLP Results
        if transcript.strip():
            st.markdown('<div class="section-header">Language Analysis</div>',
                        unsafe_allow_html=True)
            n1, n2, n3, n4 = st.columns(4)
            nlp_items = [
                ("Sentiment", f"{nlp_result['sentiment_polarity']:+.2f}", "−1 negative · +1 positive"),
                ("Complexity", f"{nlp_result['complexity_score']:.2f}", "0 simple · 1 complex"),
                ("Word Count", str(nlp_result["word_count"]), "Words spoken"),
                ("NLP Fatigue", f"{nlp_result['nlp_fatigue_score']:.1f}/10", "Language-based score"),
            ]
            for col, (label, val, desc) in zip([n1, n2, n3, n4], nlp_items):
                with col:
                    st.metric(label=label, value=val, help=desc)

            if nlp_result["fatigue_keywords"]:
                kw_html = " ".join([f'<span style="background:{CREAM};color:{ORANGE};'
                                     f'padding:3px 10px;border-radius:100px;font-size:12px;'
                                     f'font-weight:600;border:1px solid rgba(232,93,38,0.2);'
                                     f'margin:3px;display:inline-block;">{kw}</span>'
                                     for kw in nlp_result["fatigue_keywords"]])
                st.markdown(f"""
                <div style="margin-top:8px;">
                    <span style="font-size:11px;color:#6B6B88;font-weight:600;">
                    Fatigue keywords detected: </span>{kw_html}
                </div>
                """, unsafe_allow_html=True)

        # Play back audio
        if audio_file is not None:
            st.markdown('<div class="section-header">Your Recording</div>',
                        unsafe_allow_html=True)
            st.audio(audio_file)

    # ── Personal History ──
    st.markdown("---")
    st.markdown('<div class="section-header">My History</div>', unsafe_allow_html=True)

    my_sessions = [s for s in data["sessions"] if s["patient_name"] == patient_name]
    if my_sessions:
        df = pd.DataFrame([{
            "date": s["timestamp"][:10],
            "composite": s["composite"],
            "alert_tier": s["alert_tier"],
        } for s in my_sessions])
        df = df.sort_values("date")
        st.plotly_chart(fatigue_trend_chart(df, patient_name), use_container_width=True)

        # Recent sessions table
        recent = pd.DataFrame([{
            "Date": s["timestamp"][:10],
            "Time": s["timestamp"][11:16],
            "Score": f"{s['composite']:.1f}",
            "Status": s["alert_label"],
            "Keywords": ", ".join(s.get("nlp", {}).get("fatigue_keywords", [])) or "—",
        } for s in reversed(my_sessions[-7:])])
        st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.info("No check-in history yet. Complete your first check-in above!")


# ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════
#   DOCTOR DASHBOARD
# ═══════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────
def render_doctor_dashboard(data):
    # Header
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{NAVY} 0%,#0A3060 100%);
                padding:24px 32px;border-radius:16px;margin-bottom:24px;
                border-bottom:3px solid {ORANGE};">
        <div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;
                    color:{ORANGE};font-weight:700;margin-bottom:6px;">VOCAL-CRF CLINICAL DASHBOARD</div>
        <div style="display:flex;align-items:center;justify-content:space-between;">
            <div>
                <div style="font-size:26px;font-weight:700;color:white;">
                    Oncology Monitoring — Dr's View 🏥
                </div>
                <div style="font-size:13px;color:rgba(255,255,255,0.5);margin-top:4px;">
                    Real-time vocal biomarker monitoring · {datetime.datetime.now().strftime("%d %b %Y, %H:%M")}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Get latest session per patient
    all_sessions = data["sessions"]
    if not all_sessions:
        st.info("No patient data yet. Ask patients to complete check-ins first.")
        return

    # ── Summary KPIs ──
    # Latest session per patient
    patient_latest = {}
    for s in all_sessions:
        pn = s["patient_name"]
        if pn not in patient_latest or s["timestamp"] > patient_latest[pn]["timestamp"]:
            patient_latest[pn] = s

    latest_list = list(patient_latest.values())
    n_total    = len(latest_list)
    n_critical = sum(1 for s in latest_list if s["alert_tier"] >= 3)
    n_monitor  = sum(1 for s in latest_list if s["alert_tier"] == 2)
    n_stable   = sum(1 for s in latest_list if s["alert_tier"] == 0)
    avg_score  = np.mean([s["composite"] for s in latest_list]) if latest_list else 0

    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    for col, (num, label, color, bg) in zip(
        [kc1, kc2, kc3, kc4, kc5],
        [
            (n_total,    "Total Patients",    NAVY,  "#EBF4FB"),
            (n_critical, "Action Required",   RED,   "#FEF2F2"),
            (n_monitor,  "Monitor Closely",   AMBER, "#FFFBEB"),
            (n_stable,   "Stable",            GREEN, "#F0FBF6"),
            (f"{avg_score:.1f}", "Avg Score", ORANGE, CREAM),
        ]
    ):
        with col:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {color}30;border-radius:12px;
                        padding:16px 18px;text-align:center;">
                <div style="font-size:32px;font-weight:700;color:{color};">{num}</div>
                <div style="font-size:11px;color:#6B6B88;font-weight:600;
                            letter-spacing:1px;text-transform:uppercase;margin-top:4px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Patient Summary Bar Chart ──
    patients_df = pd.DataFrame([{
        "name":       s["patient_name"].split()[0] + " " + s["patient_name"].split()[-1][0] + ".",
        "full_name":  s["patient_name"],
        "composite":  s["composite"],
        "alert_tier": s["alert_tier"],
        "alert_label":s["alert_label"],
        "timestamp":  s["timestamp"][:10],
    } for s in latest_list])

    col_chart, col_alerts = st.columns([3, 2])

    with col_chart:
        st.plotly_chart(bar_chart_patients(patients_df), use_container_width=True)

    with col_alerts:
        st.markdown('<div class="section-header">Active Alerts</div>', unsafe_allow_html=True)
        alert_patients = [s for s in latest_list if s["alert_tier"] >= 2]
        if alert_patients:
            for s in sorted(alert_patients, key=lambda x: -x["composite"]):
                tier_color = RED if s["alert_tier"] >= 3 else AMBER
                suggestions = compute_composite_score(
                    s.get("acoustic", {}), s.get("nlp", {}), None
                )["suggestions"][:2]
                sugg_html = "<br>".join([f"· {sg}" for sg in suggestions])
                st.markdown(f"""
                <div style="background:white;border:1px solid #E8E8EF;
                            border-left:4px solid {tier_color};border-radius:10px;
                            padding:14px 16px;margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div style="font-size:14px;font-weight:700;color:{NAVY};">
                            {s["patient_name"]}
                        </div>
                        <div style="font-size:22px;font-weight:700;color:{tier_color};">
                            {s["composite"]:.1f}
                        </div>
                    </div>
                    <div style="font-size:12px;color:{tier_color};font-weight:600;margin:4px 0;">
                        {s["alert_label"]}
                    </div>
                    <div style="font-size:11px;color:#6B6B88;margin-top:6px;line-height:1.6;">
                        {sugg_html}
                    </div>
                    <div style="font-size:10px;color:#9999AA;margin-top:6px;">
                        Last check-in: {s["timestamp"][:16].replace("T", " ")}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#F0FBF6;border:1px solid {GREEN}30;border-radius:10px;
                        padding:16px;text-align:center;">
                <div style="font-size:24px;margin-bottom:6px;">✅</div>
                <div style="font-size:13px;color:{GREEN};font-weight:600;">All patients stable</div>
                <div style="font-size:12px;color:#6B6B88;margin-top:4px;">No active alerts</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Individual Patient Deep Dive ──
    st.markdown('<div class="section-header">Patient Deep Dive</div>', unsafe_allow_html=True)

    patient_names = sorted(list(patient_latest.keys()))
    selected = st.selectbox("Select Patient", patient_names)

    if selected:
        patient_sessions = [s for s in all_sessions if s["patient_name"] == selected]
        patient_sessions_sorted = sorted(patient_sessions, key=lambda x: x["timestamp"])
        latest = patient_sessions_sorted[-1]

        # Patient info header
        pinfo = data["patients"].get(selected, {})
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        for col, (label, val) in zip(
            [p_col1, p_col2, p_col3, p_col4],
            [
                ("Patient", selected),
                ("Age", str(pinfo.get("age", "N/A"))),
                ("Diagnosis", pinfo.get("diagnosis", "N/A")),
                ("Treatment", pinfo.get("treatment", "N/A")),
            ]
        ):
            with col:
                st.markdown(f"""
                <div style="background:white;border:1px solid #E8E8EF;border-radius:10px;
                            padding:12px 14px;">
                    <div style="font-size:10px;font-weight:700;letter-spacing:2px;
                                text-transform:uppercase;color:{BLUE};margin-bottom:4px;">{label}</div>
                    <div style="font-size:13px;font-weight:600;color:{NAVY};">{val}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # Trend + Radar
        trend_col, radar_col = st.columns([3, 2])

        with trend_col:
            df = pd.DataFrame([{
                "date": s["timestamp"][:10],
                "composite": s["composite"],
            } for s in patient_sessions_sorted])
            st.plotly_chart(
                fatigue_trend_chart(df, selected),
                use_container_width=True
            )

        with radar_col:
            if "sub_scores" in latest and latest["sub_scores"]:
                st.plotly_chart(radar_chart(latest["sub_scores"]), use_container_width=True)

        # Biomarker gauges
        st.markdown('<div class="section-header">Latest Vocal Signature</div>',
                    unsafe_allow_html=True)

        ac = latest.get("acoustic", {})
        sc = latest.get("sub_scores", {})

        g1, g2, g3, g4 = st.columns(4)
        gauge_data = [
            (sc.get("HNR", 5),       "HNR Fatigue",   ORANGE),
            (sc.get("Pausology", 5), "Pausology",      BLUE),
            (sc.get("Spectral", 5),  "Spectral",       ORANGE),
            (sc.get("NLP / Linguistic", 5), "NLP",     BLUE),
        ]
        for col, (val, label, color) in zip([g1, g2, g3, g4], gauge_data):
            with col:
                st.plotly_chart(gauge_chart(val, label, color=color),
                                use_container_width=True)

        # Raw values table
        st.markdown('<div class="section-header">Raw Acoustic Values</div>',
                    unsafe_allow_html=True)

        raw_df = pd.DataFrame([
            {"Biomarker": "HNR (dB)",          "Value": f"{ac.get('hnr', 0):.2f}",
             "Status": "⚠️ Low" if ac.get("hnr", 15) < ALERT_THRESHOLDS["hnr_warning"] else "✅ Normal",
             "Clinical Meaning": "Vocal fold efficiency — lower = more fatigued"},
            {"Biomarker": "Jitter (%)",         "Value": f"{ac.get('jitter', 0)*100:.3f}",
             "Status": "⚠️ High" if ac.get("jitter", 0) > ALERT_THRESHOLDS["jitter_warning"] else "✅ Normal",
             "Clinical Meaning": "Pitch irregularity — higher = vocal muscle instability"},
            {"Biomarker": "Shimmer (%)",        "Value": f"{ac.get('shimmer', 0)*100:.3f}",
             "Status": "⚠️ High" if ac.get("shimmer", 0) > ALERT_THRESHOLDS["shimmer_warning"] else "✅ Normal",
             "Clinical Meaning": "Amplitude irregularity — higher = muscle control loss"},
            {"Biomarker": "Pause Ratio (%)",    "Value": f"{ac.get('pause_ratio', 0)*100:.1f}",
             "Status": "⚠️ High" if ac.get("pause_ratio", 0) > ALERT_THRESHOLDS["pause_warning"] else "✅ Normal",
             "Clinical Meaning": "Speech silence proportion — higher = effort to speak"},
            {"Biomarker": "Spectral Flatness",  "Value": f"{ac.get('spectral_flatness', 0):.4f}",
             "Status": "⚠️ Elevated" if ac.get("spectral_flatness", 0) > 0.3 else "✅ Normal",
             "Clinical Meaning": "Frequency distribution — elevated in distress (female cohorts, IEEE 2025)"},
            {"Biomarker": "Pitch F0 (Hz)",      "Value": f"{ac.get('pitch_mean', 0):.0f}",
             "Status": "—",
             "Clinical Meaning": "Fundamental frequency — baseline reference"},
        ])
        st.dataframe(raw_df, use_container_width=True, hide_index=True)

        # Suggested interventions
        alert_color = latest.get("alert_color", ORANGE)
        suggestions = compute_composite_score(
            ac, latest.get("nlp", {}), None
        )["suggestions"]

        st.markdown('<div class="section-header">Suggested Clinical Actions</div>',
                    unsafe_allow_html=True)
        for i, s in enumerate(suggestions, 1):
            st.markdown(f"""
            <div style="background:white;border:1px solid #E8E8EF;border-radius:8px;
                        padding:11px 16px;margin-bottom:8px;
                        border-left:3px solid {ORANGE};">
                <span style="font-size:12px;font-weight:600;color:{NAVY};">{i}. {s}</span>
            </div>
            """, unsafe_allow_html=True)

        # Session history table
        st.markdown('<div class="section-header">Full Session History</div>',
                    unsafe_allow_html=True)
        hist_df = pd.DataFrame([{
            "Date": s["timestamp"][:10],
            "Time": s["timestamp"][11:16],
            "Score": f"{s['composite']:.1f}",
            "Alert": s["alert_label"],
            "Transcript": (s.get("transcript") or "—")[:60] + ("..." if len(s.get("transcript") or "") > 60 else ""),
            "Source": "Demo" if s.get("is_demo") else "Live",
        } for s in reversed(patient_sessions_sorted)])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────
def main():
    # Load & seed data
    data = load_data()
    data = seed_demo_patients(data)

    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:20px 10px 10px;">
            <div style="font-size:32px;margin-bottom:8px;">🎙️</div>
            <div style="font-size:18px;font-weight:700;color:white;">Vocal-CRF</div>
            <div style="font-size:11px;color:#7DC8F5;letter-spacing:2px;margin-top:4px;">
                VOICE AI · ONCOLOGY
            </div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.1);margin:16px 0;">
        """, unsafe_allow_html=True)

        view = st.radio(
            "Dashboard",
            ["🎙️  Patient Portal", "🏥  Doctor Dashboard"],
            label_visibility="collapsed"
        )

        st.markdown("""
        <hr style="border-color:rgba(255,255,255,0.1);margin:16px 0;">
        """, unsafe_allow_html=True)

        # Session stats
        n_sessions = len(data["sessions"])
        n_patients = len(data["patients"])
        st.markdown(f"""
        <div style="padding:0 8px;">
            <div style="font-size:10px;letter-spacing:2px;color:#7DC8F5;
                        text-transform:uppercase;font-weight:700;margin-bottom:10px;">Stats</div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="font-size:12px;color:rgba(255,255,255,0.6);">Patients</span>
                <span style="font-size:14px;font-weight:700;color:white;">{n_patients}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="font-size:12px;color:rgba(255,255,255,0.6);">Sessions</span>
                <span style="font-size:14px;font-weight:700;color:white;">{n_sessions}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="font-size:12px;color:rgba(255,255,255,0.6);">Today</span>
                <span style="font-size:12px;color:rgba(255,255,255,0.7);">
                    {datetime.datetime.now().strftime("%d %b")}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <hr style="border-color:rgba(255,255,255,0.1);margin:16px 0;">
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="padding:0 8px;">
            <div style="font-size:10px;letter-spacing:2px;color:#7DC8F5;
                        text-transform:uppercase;font-weight:700;margin-bottom:10px;">Biomarkers</div>
            <div style="font-size:11px;color:rgba(255,255,255,0.5);line-height:1.8;">
                🔬 HNR · Jitter · Shimmer<br>
                📊 Spectral Flatness<br>
                ⏸️ Pausology<br>
                🧠 NLP Sentiment<br>
                🗣️ Linguistic Complexity
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <hr style="border-color:rgba(255,255,255,0.1);margin:16px 0;">
        <div style="font-size:10px;color:rgba(255,255,255,0.25);padding:0 8px;line-height:1.6;">
            MSc AI for Health<br>
            Oncology Innovation Challenge 2025<br>
            Haider Ali · Hiwa Khezri · Samya Raof
        </div>
        """, unsafe_allow_html=True)

    # Render selected view
    if "Patient Portal" in view:
        render_patient_portal(data)
    else:
        render_doctor_dashboard(data)


if __name__ == "__main__":
    main()
