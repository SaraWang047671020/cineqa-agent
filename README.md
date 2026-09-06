# CineQA Studio: Agentic Cinema Quality & Observability Platform 🎬

> **Agentic Cinema Hackathon (Google Cloud & Grafana Labs Track)**  
> Autonomous AI Quality Assurance & Observability for AI Film Production that turns video hallucinations into director-grade cinema using **Google Cloud Vertex AI (Veo 3.1 & Omni)**, **Gemini 2.5/3.6 Multimodal Vision**, **Split-Conformal Prediction (LAC)**, **ClickHouse Cloud**, and **Grafana Dashboards**.

---

## 🌟 Core Architecture & Closed Loop

```
┌───────────────────────────────────────────────────────────────────────┐
│                    1. Director Scene Script / Prompt                  │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
       ┌───────────────────────────▼───────────────────────────┐
       │   Prompt Director & Storyboard Architect              │
       │   (4-Beat Narrative Breakdown & Multi-Option Keyframes)│
       └───────────────────────────┬───────────────────────────┘
                                   │
       ┌───────────────────────────▼───────────────────────────┐
       │   Google Veo 3.1 & Omni Interactions API              │
       │   (Text-to-Video, Image-to-Video, Video-to-Video Edit)│
       └───────────────────────────┬───────────────────────────┘
                                   │
       ┌───────────────────────────▼───────────────────────────┐
       │   2. Multi-Attribute Claim Extractor & Frame Sampling │
       │   (Kinematics, Lighting, Continuity, 7 Dimensions)    │
       └───────────────────────────┬───────────────────────────┘
                                   │
       ┌───────────────────────────▼───────────────────────────┐
       │   3. Gemini Multimodal Consensus Verification Engine   │
       │   (Physical Law Constraints & Causal Video Validation) │
       └───────────────────────────┬───────────────────────────┘
                                   │
       ┌───────────────────────────▼───────────────────────────┐
       │   4. Split-Conformal Prediction (LAC) Risk Engine     │
       │   (90% Distribution-Free Coverage Prediction Interval)│
       └───────────────────────────┬───────────────────────────┘
                                   │
       ┌───────────────────────────▼───────────────────────────┐
       │   5. Targeted Prompt Remediator & V2V Fine-Tuning     │
       │   (Auto-Healing Prompt Mutation & Closed-Loop Rerender│
       └───────────────────────────┬───────────────────────────┘
                                   │ (OTel, Metrics & Logs)
       ┌───────────────────────────▼───────────────────────────┐
       │   6. ClickHouse & Grafana Studio Observability        │
       │   (Real-time Audit Ledger, GPU Hours Saved, Radars)   │
       └───────────────────────────────────────────────────────┘
```

---

## ✨ Key Capabilities

1. **Autonomous Prompt Director**:
   - Converts simple one-sentence ideas into 4-beat cinematic treatments (Establishing, Conflict, Climax, Resolution).
   - Suggests optical lens ratios, cinematic lighting styles, and generates multiple high-fidelity keyframe candidates for seed-frame I2V generation.

2. **Multimodal Physics & Temporal Verification**:
   - Analyzes spatio-temporal video slices with **Gemini 2.5/3.6**.
   - Evaluates 7 atomic dimensions: Count, Action, Direction, Position, State, Size, and Color.
   - Evaluates physical laws: gravity consistency, collision response, momentum, and light source coherence.

3. **Statistically Guaranteed Uncertainty (Split-Conformal Prediction)**:
   - Uses distribution-free conformal prediction ($1 - \alpha = 0.90$) to calculate non-conformity threshold $\hat{q}$.
   - Prevents hallucinations from slipping through by actively flagging high-uncertainty claims for human-in-the-loop review.

4. **Closed-Loop Auto-Remediation & V2V Fine-Tuning**:
   - Pinpoints specific hallucinated frames and mutations.
   - Automatically formulates corrective negative prompts and structured tweak directives.
   - Drives Google Cloud Vertex AI Omni Interactions API in Video-to-Video (V2V) edit mode to heal flaws while preserving visual identity.

5. **Enterprise Observability**:
   - Real-time logging of verification claims and remediation lineage to **ClickHouse Cloud**.
   - Prometheus metrics and Grafana dashboards tracking hallucination frequency, latency, and GPU compute cost savings.

---

## 🚀 Quickstart

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/SaraWang047671020/cineqa-agent.git
cd cineqa-agent
pip install -r requirements.txt
```

### 2. Configure Environment (.env)
Create a `.env` file based on `.env.example`:
```ini
USE_VERTEX_AI=true
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-east5
GEMINI_API_KEY=your-gemini-api-key

# Optional: ClickHouse Cloud for Telemetry
CLICKHOUSE_HOST=your-clickhouse-host
CLICKHOUSE_PORT=8443
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your-clickhouse-password
```

### 3. Run Studio UI Locally
```bash
python -m streamlit run ui/app.py
```
*(Or double-click `run_studio.bat` on Windows)*

---

## 📊 Tech Stack
- **Foundation Models**: Google Veo 3.1 Fast, Google Omni (Interactions API), Google Gemini 2.5 / 3.6 Flash & Pro
- **Cloud Infrastructure**: Google Cloud Vertex AI, Cloud Storage
- **Uncertainty Quantification**: Split-Conformal Prediction (Distribution-Free Non-Conformity Scoring)
- **Database & Telemetry**: ClickHouse Cloud, Prometheus, OpenTelemetry, Grafana
- **Computer Vision**: OpenCV, FFmpeg-Python, PIL
- **Application Framework**: Streamlit (Python)

