# CineQA Studio: Agentic Cinema Quality & Observability Platform 🎬

> **Agentic Cinema Hackathon (Google Cloud Track)**  
> Autonomous AI Quality Assurance & Observability for AI Film Production that turns video hallucinations into director-grade cinema using **Google Cloud Vertex AI (Omni Interactions API)**, **Gemini 2.5/3.6 Multimodal Vision**, **Split-Conformal Prediction (LAC)**, **ClickHouse Cloud**, and **Model Context Protocol (MCP)**.

---

## 🌟 Core Architecture & Closed Loop

```
┌────────────────────────────────────────────────────────────────────────┐
│                      1. Director Scene Seed Idea                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
        ┌───────────────────────────▼───────────────────────────┐
        │   Prompt Director & Adaptive Interview (via MCP)      │◄─── [ClickHouse Cloud]
        │   • Queries historic tweak hotspots via MCP Tool      │     (guidance_events)
        │   • 5-Axis Dynamic Interview (Framing, Action, etc.)  │     Autonomous Axis Priority
        │   • 3x Low-Cost Keyframe Storyboard Generation        │
        └───────────────────────────┬───────────────────────────┘
                                    │ (Confirmed Shot Recipe)
        ┌───────────────────────────▼───────────────────────────┐
        │   Google Gemini Omni Interactions API                 │
        │   (Text-to-Video / Image-to-Video Seed Generation)    │
        └───────────────────────────┬───────────────────────────┘
                                    │ (Video Take 1)
        ┌───────────────────────────▼───────────────────────────┐
        │   2. Multi-Attribute Claim Extractor & Frame Sampling │
        │   (7 Atomic Dimensions: Count, Action, Direction,     │
        │    Position, State, Size, Color)                      │
        └───────────────────────────┬───────────────────────────┘
                                    │
        ┌───────────────────────────▼───────────────────────────┐
        │   3. Gemini Multimodal Consensus Verification Engine   │
        │   (Physical Law Constraints & Temporal Consistency)   │
        └───────────────────────────┬───────────────────────────┘
                                    │
        ┌───────────────────────────▼───────────────────────────┐
        │   4. Split-Conformal Prediction (LAC) Decision Layer  │
        │   (90% Distribution-Free Guarantee: Autonomous /      │
        │    Human Review Escalation)                           │
        └───────────────────────────┬───────────────────────────┘
                                    │ (Flags Raised, Not Auto-Fixed)
        ┌───────────────────────────▼───────────────────────────┐
        │   5. Guided Observation & User-Directed Remediation   │
        │   • Human-readable defect timestamps & frame locations│
        │   • Director inputs adjustment in natural language    │
        │   • Prompt Optimizer formulates technical directives  │
        │   • Omni Incremental Edit via previous_interaction_id │
        └───────────────────────────┬───────────────────────────┘
                                    │ (Take 2: Refined Cinema)
        ┌───────────────────────────▼───────────────────────────┐
        │   6. ClickHouse Observability & Telemetry Store       │
        │   (Real-time ingestion to verification_ledger &       │
        │    guidance_events — feeds back into Step 1 via MCP)  │
        └───────────────────────────────────────────────────────┘
```

---

## ✨ Key Capabilities

1. **Adaptive Prompt Director & MCP Learning Loop**:
   - Interviews the director one dimension at a time across 5 core cinematic axes: Framing & Camera Motion, Action & Blocking, Location & Environment, Lighting & Mood, and Visual Style — with each question's options dynamically conditioned on previous choices.
   - **ClickHouse MCP Autonomous Querying**: Before asking its first question, the agent autonomously queries ClickHouse over Model Context Protocol (`get_axis_priority`) to inspect which dimensions users have most frequently had to remediate, prioritizing interview questions based on real-world error history rather than a hardcoded script.
   - Generates 3 high-fidelity keyframe candidates via Gemini Flash/Imagen for pennies, allowing directors to lock composition, lighting, and style before committing resources to full video generation.

2. **Multimodal Physics & Temporal Verification Engine**:
   - Deconstructs prompts into 7 atomic visual dimensions: **Count, Action, Direction, Position, State, Size, and Color**.
   - Employs Gemini Multimodal Vision across spatio-temporal video slices to verify scene consistency, photometric lighting stability, and kinematic plausibility (gravity, momentum, collision responses, continuity).

3. **Statistically Guaranteed Uncertainty (Split-Conformal Prediction)**:
   - Replaces uncalibrated LLM confidence heuristics with distribution-free Split-Conformal Prediction (`SplitConformalClassifier` with LAC non-conformity scoring, calibrated against 92 human-annotated video samples).
   - Computes conformal critical threshold $\hat{q}$ for statistical confidence $1 - \alpha = 0.90$.
   - Multi-verdict prediction sets ($|\mathcal{C}(X)| \ge 2$) automatically escalate ambiguous cases into human review queues, ensuring autonomous verdicts meet strict empirical safety bounds.

4. **Guided Observation & User-Directed Remediation (Human-in-the-Loop)**:
   - When defects are detected, **CineQA never mutates prompts arbitrarily or auto-heals without consent**. Instead, it surfaces human-readable observations: precise timestamp intervals, spatial frame locations, and verified discrepancies.
   - The director describes their intended tweak in plain natural language. The Prompt Optimizer translates it into precise, technical camera/action directives while strictly preserving all unflagged scene elements.
   - Dispatches fine-tuning to Google Cloud Vertex AI Omni Interactions API with `previous_interaction_id`, enabling true stateful incremental editing that changes only what was asked.

5. **Production Telemetry & Observability**:
   - Real-time streaming of all guidance interactions (`guidance_events`) and claim verification audits (`verification_ledger`) directly to ClickHouse Cloud.
   - Dual-table analytical store powers live studio metrics (Verified Adherence rates, claim pass/fail distributions, defect hot spots) and closes the observability loop back into the Prompt Director's runtime MCP memory.

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

# ClickHouse Cloud for Telemetry & MCP Memory
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
- **Foundation Models**: Google Gemini Omni (Interactions API), Google Gemini 2.5 / 3.6 Flash & Pro
- **Cloud Infrastructure**: Google Cloud Vertex AI, Google Cloud Storage
- **Agent Protocols & Tools**: Model Context Protocol (MCP Server & Client)
- **Uncertainty Quantification**: Split-Conformal Prediction (Distribution-Free Non-Conformity Scoring, LAC)
- **Database & Observability**: ClickHouse Cloud (Dual Core Tables: `guidance_events` & `verification_ledger`)
- **Computer Vision**: OpenCV, ImageIO-FFmpeg, PIL
- **Application Framework**: Streamlit (Python)
