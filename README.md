# CineQA Studio: AI Cinema Verification & Observability Platform 🎬

> **Agentic Cinema Hackathon (Google Cloud & Grafana Labs Track)**
> An Agentic Quality Control & Observability platform for AI Film Production that eliminates wasted GPU compute and blind re-generation takes using **Google Gemini (Vertex AI)**, **MAPIE (Conformal Prediction)**, and **Grafana Dashboards**.

---

## 🌟 Core Architecture

```
                    ┌──────────────────────────────────────────────────┐
                    │               Director Scene Script              │
                    └────────────────────────┬─────────────────────────┘
                                             │
                       ┌─────────────────────▼─────────────────────┐
                       │   1. Multi-Attribute Claim Extractor      │
                       │   (7 Atomic Dimensions: Count, Action,    │
                       │    Direction, Position, State, Size, Color│
                       └─────────────────────┬─────────────────────┘
                                             │
                       ┌─────────────────────▼─────────────────────┐
                       │    2. Adaptive Frame Sampling Pipeline    │
                       │    (Static: 3 Keyframes | Sequential: 0.4s│
                       └─────────────────────┬─────────────────────┘
                                             │
                       ┌─────────────────────▼─────────────────────┐
                       │    3. Consensus Verification Engine       │
                       │    (5-Step Causal Reasoning + Multi-Vote) │
                       └─────────────────────┬─────────────────────┘
                                             │
                       ┌─────────────────────▼─────────────────────┐
                       │    4. MAPIE 1.5.0 Conformal Risk Engine   │
                       │    (90% Coverage Prediction Intervals)    │
                       └─────────────────────┬─────────────────────┘
                                             │
                       ┌─────────────────────▼─────────────────────┐
                       │    5. Targeted Prompt Remediator & QA     │
                       │    (Negative Prompts, Inpaint Guidance)   │
                       └─────────────────────┬─────────────────────┘
                                             │ (Prometheus & OpenTelemetry)
                       ┌─────────────────────▼─────────────────────┐
                       │    6. Grafana Studio Observability        │
                       │    (Quality Radars, Dollars Saved, Traces)│
                       └───────────────────────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Fill in your GOOGLE_CLOUD_PROJECT (Vertex AI) or GEMINI_API_KEY
```

### 3. Launch Studio UI
```bash
python -m streamlit run ui/app.py
```
*(Or double-click `run_studio.bat` on Windows)*

### 4. Connect Grafana
Import `grafana/dashboards/cineqa_studio_dashboard.json` into your Grafana instance and point the Prometheus data source to `http://localhost:8000`.

---

## 📊 Tech Stack & Compliance
* **LLM & Video Reasoning**: Google Gemini 2.5 Flash / Pro (Google Cloud Vertex AI)
* **Uncertainty Quantification**: MAPIE 1.5.0 (Model Agnostic Prediction Interval Estimator, BSD-3)
* **Observability & Telemetry**: Grafana Labs, Prometheus, OpenTelemetry
* **Computer Vision**: OpenCV, FFmpeg-Python
* **Application Framework**: Streamlit (Python)
