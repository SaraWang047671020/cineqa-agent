# 🎬 CineQA Studio: AI Cinema Quality & Observability Platform
## Accelerated 14-Day Execution Schedule (with 3-Day Buffer)
**Agentic Cinema Blockbuster Hackathon — Google Cloud & Grafana Labs Track**

---

### 1. Executive Summary & Timeline Strategy

* **Hackathon Deadline**: 2026-09-09 17:00 EDT
* **Target Feature Freeze & Completion Date**: **2026-09-06 (Day 14)**
* **Buffer & Rehearsal Window**: **2026-09-07 – 2026-09-09 (3 Days Buffer)**

```
 2026/08/24                                       2026/09/06       2026/09/09 (Deadline)
     │                                                 │                    │
     ▼                                                 ▼                    ▼
┌──────────────────────────────────────────────────────┬────────────────────┐
│ Phase 1 - 5: Accelerated Core Development & Testing  │ 3-Day Buffer Phase │
│ (14 Days: Architecture, MAPIE, Grafana, Demo Video)  │ (Rehearsal & Sub)  │
└──────────────────────────────────────────────────────┴────────────────────┘
```

---

### 2. Comprehensive 14-Day Daily Execution Plan

| Phase | Day & Date | Milestones & Technical Tasks | Module / Status |
| :--- | :--- | :--- | :--- |
| **Phase 1: Environment & Observability** | **Day 1 (08/24)** | Authenticate Google Cloud Vertex AI (`project-aefe3ba2-ab8b-478a-82d`), verify Gemini 2.5 connection. | `config/settings.py` ✅ **COMPLETED** |
| | **Day 2 (08/25)** | Implement Prometheus Metrics Server on Port 8000 & OpenTelemetry tracing pipeline. | `telemetry/metrics.py`, `telemetry/tracer.py` |
| | **Day 3 (08/26)** | Configure Grafana Studio Observability Dashboard; verify live streaming of quality & dollars-saved gauges. | `grafana/dashboards/cineqa_studio_dashboard.json` |
| **Phase 2: Verification Engine & Agent Engine** | **Day 4 (08/27)** | Finalize 7-attribute atomic visual claim extraction (`engine/claims.py`). | 7-Attribute Claim Extractor |
| | **Day 5 (08/28)** | Implement adaptive frame sampling (static 3-frame vs sequential 0.4s) & 5-step causal consensus verifier. | `engine/verify.py`, `engine/pipeline.py` |
| | **Day 6 (08/29)** | Package verification core into `CineQAAgentEngine` compliant with Vertex AI Reasoning Engine specs. | `agents/agent_engine.py` |
| **Phase 3: MAPIE 1.5.0 Benchmark Calibration** | **Day 7 (08/30)** | Ingest 90+ human-reviewed benchmark samples (`template.csv`) to calibrate MAPIE `SplitConformalRegressor`. | `agents/conformal_judge.py`, `eval/` |
| | **Day 8 (08/31)** | Run automated precision & recall benchmarking script (`run_precision_eval.py`) across all 7 visual categories. | Benchmark Evaluation Report |
| **Phase 4: Remediation, Telemetry Replay & UI** | **Day 9 (09/01)** | Implement `agents/remediator.py` for automated prompt surgery, negative prompt generation & inpaint intervals. | Prompt Remediator Agent |
| | **Day 10 (09/02)** | Polish Streamlit Studio UI (`ui/app.py`) with full interactive Verification Ledger and decision cards. | Streamlit Studio UI |
| | **Day 11 (09/03)** | Develop telemetry replay script (`scripts/replay_telemetry.py`) to stream 90+ takes into Grafana. | Grafana Telemetry Replay |
| **Phase 5: Stress Testing & Demo Video** | **Day 12 (09/04)** | Perform end-to-end stress testing; test Vertex AI cloud deployment (`scripts/deploy_agent_engine.py`). | Cloud Deployment Verification |
| | **Day 13 (09/05)** | Record 3-minute Blockbuster Demo Video (Pain point -> Real-world take failure -> Auto-repair -> Grafana Telemetry). | 3-Minute Demo Video |
| | **Day 14 (09/06)** | **🏆 PROJECT COMPLETE**: Finalize open-source repository documentation, architecture diagrams, and Apache 2.0 license. | GitHub Repository Ready |
| **Buffer & Submission Window** | **Day 15 (09/07)** | Upload Demo Video to YouTube/Vimeo with accurate English closed captions. | Public Video URL |
| | **Day 16 (09/08)** | Test hosted live application URL and perform final sanity check on all links and assets. | Live Hosted URL |
| | **Day 17 (09/09)** | Submit project on Devpost targeting the **Grafana Labs Track** before the deadline! 🚀 | **Devpost Official Submission** |

---

### 3. Key Success & Evaluation Metrics

1. **Precision & Robustness**: Minimum 90% accuracy on the 90+ clip human-annotated benchmark dataset.
2. **Observability Integration**: Rich, real-time Grafana Studio dashboards with live Uncertainty Bands and GPU cost savings metrics.
3. **Google Cloud Agent Platform Alignment**: Full compliance with Vertex AI Reasoning Engine container specifications.
