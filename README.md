# 🎙️ Vocal-CRF — Voice-Based Fatigue Monitoring
### Speech-Enabled AI for Sex-Specific Adverse Event Detection in Oncology
**MSc AI for Health · Oncology Innovation Challenge 2025**
*Haider Ali · Hiwa Khezri · Samya Raof*

---

## ⚡ Quick Start (5 minutes)

### Step 1 — Install Python
Make sure you have **Python 3.9 or higher**.
Check with: `python --version`

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

If you get errors on Windows with librosa:
```bash
pip install librosa --no-deps
pip install audioread decorator numba
```

If you get errors with praat-parselmouth:
```bash
pip install praat-parselmouth --only-binary=:all:
```

### Step 3 — Download TextBlob language data (one-time)

```bash
python -m textblob.download_corpora
```

### Step 4 — Run the app

```bash
streamlit run vocal_crf_app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 🏗️ What the App Does

### Two Dashboards

#### 🎙️ Patient Portal
- Patient records their voice using the microphone button
- Can also type a transcript for NLP analysis
- Sees their own fatigue score (0–10 gauge)
- Radar chart of all 6 biomarkers
- Waveform visualisation of their recording
- Personal history trend chart

#### 🏥 Doctor Dashboard
- Overview KPIs: total patients, critical alerts, stable count
- Bar chart of all patients' latest scores
- Active alerts panel with suggested clinical actions
- Patient deep-dive: trend chart + radar + biomarker gauges
- Raw acoustic values table
- Full session history per patient

---

## 🔬 The Pipeline (What Happens When You Record)

```
Voice Recording
      │
      ▼
[A] Acoustic Extraction (on-device, via librosa + praat-parselmouth)
      │── HNR (Harmonic-to-Noise Ratio)
      │── Jitter (pitch cycle variation)
      │── Shimmer (amplitude variation)
      │── Spectral Flatness (distress marker, validated IEEE 2025)
      │── Pausology (speech-to-silence ratio)
      │── Pitch F0 mean + SD
      │── RMS Energy
      │
      ▼
[B] NLP Analysis (via TextBlob + custom linguistic scoring)
      │── Sentiment polarity
      │── Linguistic complexity (type-token ratio + word length)
      │── Fatigue keyword detection
      │── NLP fatigue sub-score
      │
      ▼
[C] Composite Score Fusion (weighted multi-modal)
      │── HNR weight:      25%
      │── Pausology weight: 20%
      │── Spectral weight:  15%
      │── NLP weight:       20%
      │── Jitter weight:    10%
      │── Shimmer weight:   10%
      │
      ▼
[D] Alert Tier Assignment
      │── Tier 0: Stable      (score 0.0 – 3.4)
      │── Tier 1: Watch       (score 3.5 – 5.4)
      │── Tier 2: Monitor     (score 5.5 – 7.4) → nurse call triggered
      │── Tier 3: Critical    (score 7.5 – 10)  → urgent action
      │
      ▼
[E] Clinical Dashboard + Persistent Storage (vocal_crf_data.json)
```

---

## 📦 File Structure

```
vocal_crf_app.py         ← Main application (single file)
requirements.txt         ← Python dependencies
vocal_crf_data.json      ← Auto-created on first run (patient data)
README.md                ← This file
```

---

## 🔊 Demo Data

The app automatically generates **3 demo patients** with 14 days of
realistic history on first run:

- **Sarah Mitchell** — Breast Cancer, Docetaxel + Cyclophosphamide
- **Amira Hassan** — Ovarian Cancer, Carboplatin + Paclitaxel
- **Priya Sharma** — Cervical Cancer, Cisplatin + Radiation

Each simulates a realistic treatment cycle (good → fatigue spike on
days 3–5 → recovery), so the doctor dashboard shows meaningful trends
immediately.

---

## 🎯 For the Competition Demo

**Recommended demo flow (5 minutes):**

1. Open Doctor Dashboard → show the overview KPIs and bar chart
2. Click Sarah Mitchell → show her 14-day trend with the fatigue spike
3. Show the radar chart + biomarker gauges + alert suggestions
4. Switch to Patient Portal → record a short voice message live
   *("I'm feeling really heavy and tired today, it's hard to move")*
5. Click Analyse — show the live waveform, gauge, radar, and keywords
6. Switch back to Doctor Dashboard → Sarah's new score appears

**Key talking points while demoing:**
- "The raw audio never leaves the device — only these numbers are sent"
- "The acoustic score dropped BEFORE Sarah would have self-reported"
- "This replaces a 10-screen app with one sentence spoken from a pillow"

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: librosa` | `pip install librosa` |
| `ModuleNotFoundError: parselmouth` | `pip install praat-parselmouth` |
| No microphone button in browser | Use Chrome or Edge, not Firefox |
| `ALSA error` on Linux | Ignore — it's a harmless audio driver warning |
| App won't open | Check terminal for error, ensure port 8501 is free |
| Scores seem wrong | Speak for at least 15 seconds, clearly |
| `textblob` corpus error | Run `python -m textblob.download_corpora` |

---

## 📚 Key References

1. Unger et al., JCO 2022 — 34% higher AE risk in women (23,296 patients)
2. AACR 2025, Du et al. — 69% higher severe fatigue in female survivors
3. Bridge2AI-Voice, NIH 2024 — Training dataset (12,500+ recordings)
4. Bensoussan et al., JAMA Otolaryngol 2024 — Audiomics framework
5. MDPI Sensors 2025 — Spectral Flatness validated in female cohorts
6. PMC Voice for Health 2021 — HNR/Jitter/Shimmer for remote monitoring

---

*Built with Streamlit · librosa · praat-parselmouth · TextBlob · Plotly*
