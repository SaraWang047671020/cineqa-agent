import sys
import os
import time
import json
from pathlib import Path
import importlib

# Ensure repository root is in Python sys.path so modules like engine, agents, config can be resolved in any environment
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import engine.storyboard
import engine.generator
_ = importlib.reload(engine.storyboard)
_ = importlib.reload(engine.generator)
_ = importlib.reload(sys.modules['agents.remediator']) if 'agents.remediator' in sys.modules else None
_ = importlib.reload(sys.modules['database.logger']) if 'database.logger' in sys.modules else None
_ = importlib.reload(sys.modules['database.clickhouse_init']) if 'database.clickhouse_init' in sys.modules else None
_ = importlib.reload(sys.modules['engine.verify']) if 'engine.verify' in sys.modules else None
_ = importlib.reload(sys.modules['engine.pipeline']) if 'engine.pipeline' in sys.modules else None
_ = importlib.reload(sys.modules['engine.claims']) if 'engine.claims' in sys.modules else None
_ = importlib.reload(sys.modules['agents.prompt_director']) if 'agents.prompt_director' in sys.modules else None
_ = importlib.reload(sys.modules['agents.mcp_client_wrapper']) if 'agents.mcp_client_wrapper' in sys.modules else None

import streamlit as st

from config.settings import settings
from engine.claims import extract_claims
from telemetry.metrics import TAKES_TOTAL, INSPECTION_DURATION_SECONDS, HUMAN_REVIEWS_TRIGGERED
from engine.pipeline import run_pipeline, stream_pipeline
from engine.generator import generate_video
from agents.remediator import PromptRemediatorAgent

st.set_page_config(page_title="CineQA Studio", layout="wide", page_icon="🎬")

from telemetry.metrics import init_bq_table
try:
    import google.auth
    _, project = google.auth.default()
    if project:
        init_bq_table(project)
except Exception:
    pass




# High-End Dark Cinema Studio Custom CSS
st.markdown("""
<style>
/* Font Imports */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* HIDE STREAMLIT SIDEBAR COMPLETELY */
section[data-testid="stSidebar"], [data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}

/* ELIMINATE STREAMLIT NATIVE WHITE TOP HEADER BAR */
header[data-testid="stHeader"], .stApp > header {
    background-color: transparent !important;
    background: transparent !important;
}

div[data-testid="stDecoration"] {
    display: none !important;
}

/* WIDEN MAIN CONTENT VIEWPORT */
.main .block-container, div[data-testid="stAppViewBlockContainer"] {
    max-width: 96% !important;
    padding-top: 1rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

/* Global Canvas & Layout */
.stApp {
    background: radial-gradient(circle at 50% 0%, #101626 0%, #080B12 60%, #05070B 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* HIGH-CONTRAST PURE WHITE TEXT ON HEADINGS & PARAGRAPHS */
h1, h2, h3, h4, h5, h6, 
p, span, label, li, a, strong, em, b, i,
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown strong, .stMarkdown em,
label[data-testid="stWidgetLabel"], label[data-testid="stWidgetLabel"] p, label[data-testid="stWidgetLabel"] span,
div[data-testid="stCaptionContainer"] p, .stCaption,
div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary span,
div[data-testid="stExpander"] p, div[data-testid="stExpander"] div,
div[data-testid="stCheckbox"] label span, div[data-testid="stCheckbox"] label p,
div[data-testid="stAlert"] *,
div[data-testid="stMetricLabel"] *, div[data-testid="stMetricValue"],
textarea, input {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Studio Hero Header */
.studio-hero {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.85) 0%, rgba(30, 41, 59, 0.5) 100%);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 16px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(12px);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.studio-title {
    font-size: 1.95rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(120deg, #FFFFFF 0%, #BAE6FD 45%, #818CF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent !important;
    margin: 0;
    line-height: 1.2;
}

.studio-subtitle {
    color: #CBD5E1 !important;
    -webkit-text-fill-color: #CBD5E1 !important;
    font-size: 0.88rem;
    letter-spacing: 0.02em;
    margin-top: 4px;
}

.studio-badge {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(139, 92, 246, 0.25) 100%);
    border: 1px solid rgba(129, 140, 248, 0.5);
    color: #C7D2FE !important;
    -webkit-text-fill-color: #C7D2FE !important;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 7px 15px;
    border-radius: 9999px;
    box-shadow: 0 0 16px rgba(99, 102, 241, 0.3);
    white-space: nowrap;
}

/* Step Header */
.step-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
    font-size: 1.15rem;
    font-weight: 700;
    color: #FFFFFF !important;
}

.step-badge {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    padding: 3px 9px;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4);
}

/* Containers / Cards */
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] > div:hover {
    border-color: rgba(99, 102, 241, 0.45) !important;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.5) !important;
}

/* Recommended Option Badge & Note */
.rec-pill {
    background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    padding: 3px 9px;
    border-radius: 9999px;
    display: inline-block;
    margin-left: 8px;
    vertical-align: middle;
    box-shadow: 0 2px 10px rgba(245, 158, 11, 0.45);
}

.director-note-box {
    background: rgba(30, 41, 59, 0.65);
    border-left: 3px solid #818CF8;
    padding: 8px 12px;
    border-radius: 0 8px 8px 0;
    font-size: 0.86rem;
    color: #F1F5F9 !important;
    -webkit-text-fill-color: #F1F5F9 !important;
    margin: 8px 0 12px 0;
}

/* ========================================================= */
/* BUTTONS: COMPACT, SHARP & MODERN (NOT BULKY)              */
/* ========================================================= */
button,
button[data-testid*="stBaseButton"],
div[data-testid="stButton"] button {
    background: #1E293B !important;
    background-color: #1E293B !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 6px !important;
    padding: 3px 12px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    min-height: 1.95rem !important;
    height: auto !important;
    line-height: 1.2 !important;
    outline: none !important;
    transition: all 0.15s ease !important;
}

/* Force all button inner elements to be completely transparent without rogue borders */
button *,
button div,
button p,
button span,
div[data-testid="stButton"] button *,
div[data-testid="stButton"] button div,
div[data-testid="stButton"] button p,
div[data-testid="stButton"] button span {
    background: transparent !important;
    background-color: transparent !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-size: 0.82rem !important;
    line-height: 1.2 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

button:hover,
button[data-testid*="stBaseButton"]:hover,
div[data-testid="stButton"] button:hover {
    background: #2D3D54 !important;
    background-color: #2D3D54 !important;
    border-color: #818CF8 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* PRIMARY BUTTONS - SLEEK ACCENT GRADIENT */
button[kind="primary"],
button[data-testid="stBaseButton-primary"],
div[data-testid="stButton"] button[kind="primary"],
div[data-testid="stButton"] button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
    background-color: #4F46E5 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.28) !important;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.35) !important;
}

button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #4338CA 0%, #4F46E5 100%) !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.55) !important;
}

/* ========================================================= */
/* TEXT INPUTS & TEXTAREAS                                   */
/* ========================================================= */
input, textarea,
div[data-baseweb="input"],
div[data-baseweb="textarea"] {
    background-color: #0B101B !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 6px !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.3) !important;
}

/* ========================================================= */
/* SELECTBOX: CLEAN DARK TRIGGER & WORKING DROPDOWN POPOVER  */
/* ========================================================= */
div[data-baseweb="select"] > div:first-child {
    background-color: #1E293B !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    border-radius: 6px !important;
    min-height: 2rem !important;
    height: 2.1rem !important;
}

div[data-baseweb="select"] div[aria-selected],
div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-size: 0.84rem !important;
}

div[data-baseweb="select"] svg {
    fill: #CBD5E1 !important;
}

/* Dropdown list container in popover */
ul[role="listbox"] {
    background-color: #1E293B !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    border-radius: 6px !important;
    padding: 4px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.8) !important;
}

li[role="option"] {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    padding: 7px 12px !important;
    border-radius: 4px !important;
    font-size: 0.84rem !important;
    cursor: pointer !important;
}

li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background-color: #334155 !important;
    color: #38BDF8 !important;
    -webkit-text-fill-color: #38BDF8 !important;
}

/* ========================================================= */
/* EXPANDERS: CLEAN DARK THEME (NO INVASIVE WILDCARD *)      */
/* ========================================================= */
div[data-testid="stExpander"],
div.stExpander {
    background: #111726 !important;
    background-color: #111726 !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    margin-bottom: 12px !important;
}

div[data-testid="stExpander"] details,
div.stExpander details {
    background: #111726 !important;
    background-color: #111726 !important;
    border: none !important;
}

div[data-testid="stExpander"] summary,
div.stExpander summary {
    background: #111726 !important;
    background-color: #111726 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    padding: 8px 14px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    border: none !important;
}

div[data-testid="stExpander"] summary:hover,
div[data-testid="stExpander"] summary:focus,
div.stExpander summary:hover,
div.stExpander summary:focus {
    background: #161F33 !important;
    background-color: #161F33 !important;
}

div[data-testid="stExpander"] div[data-testid="stExpanderDetails"],
div[data-testid="stExpanderDetails"],
div.stExpander div[data-testid="stExpanderDetails"] {
    background: #111726 !important;
    background-color: #111726 !important;
    color: #FFFFFF !important;
    padding: 10px 14px !important;
    border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}

div[data-testid="stMetricValue"] {
    font-weight: 800 !important;
    color: #38BDF8 !important;
    -webkit-text-fill-color: #38BDF8 !important;
}

/* Video Player */
video {
    border-radius: 10px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.55) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Code */
pre, code {
    background-color: #080C14 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: #38BDF8 !important;
    -webkit-text-fill-color: #38BDF8 !important;
}
</style>
""", unsafe_allow_html=True)

# Studio Header Banner
st.markdown("""
<div class="studio-hero">
    <div>
        <div class="studio-title">🎬 CineQA Studio</div>
        <div class="studio-subtitle">Autonomous Healing Workflow · Multimodal 4-Tier Verification · Hollywood Lineage Engine</div>
    </div>
    <div>
        <span class="studio-badge">⚡ OMNI &amp; VEO PRO</span>
    </div>
</div>
""", unsafe_allow_html=True)

if "take_history" not in st.session_state:
    st.session_state["take_history"] = []
if "extracted_claims" not in st.session_state:
    st.session_state["extracted_claims"] = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state["session_id"] = str(uuid.uuid4())

# --- TOP STUDIO COMMAND TOOLBAR (Moved from sidebar to top) ---
with st.container(border=True):
    col_t_title, col_t_eng, col_t_dur, col_t_chk1, col_t_chk2, col_t_rst = st.columns(
        [1.3, 2.3, 1.2, 1.0, 1.0, 1.2],
        vertical_alignment="center"
    )
    with col_t_title:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.4rem;">⚙️</span>
            <div>
                <div style="font-size: 0.95rem; font-weight: 800; color: #FFFFFF; line-height: 1.1;">STUDIO ENGINE</div>
                <div style="font-size: 0.72rem; color: #CBD5E1;">Global Controls</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_t_eng:
        vid_engine = st.selectbox(
            "Video Engine",
            ["gemini-omni-flash-preview (Gemini Omni)", "veo-3.1-fast-generate-001 (Google Veo)"],
            index=0,
            label_visibility="collapsed"
        )
    with col_t_dur:
        durations = list(range(3, 11)) if "omni" in vid_engine.lower() else [4, 8]
        default_dur_idx = 2 if "omni" in vid_engine.lower() else 0
        vid_duration = st.selectbox(
            "Duration (sec)",
            durations,
            index=default_dur_idx,
            label_visibility="collapsed"
        )
    with col_t_chk1:
        live_veo = st.checkbox("🔥 Live API", value=True)
    with col_t_chk2:
        dry_run = st.checkbox("🧪 Mock Run", value=False)
    with col_t_rst:
        if st.button("🔄 Reset Project", use_container_width=True):
            st.session_state.clear()
            st.rerun()

col1, col2 = st.columns([1, 1.2])

with col1:
    if "director_step" not in st.session_state:
        st.session_state["director_step"] = "input"
        st.session_state["director_answered"] = []
        st.session_state["director_current_q"] = None

    st.markdown('<div class="step-header"><span class="step-badge" style="background: linear-gradient(135deg, #0EA5E9, #2563EB);">DIRECTOR</span><span>Guided Prompt Generation</span></div>', unsafe_allow_html=True)
    step = st.session_state["director_step"]

    if step == "input":
        st.markdown('<div class="step-header"><span class="step-badge">STEP 1</span><span>Input Core Concept</span></div>', unsafe_allow_html=True)
        raw_idea = st.text_area("What scene do you envision?", value=st.session_state.get("director_prompt", ""), placeholder="e.g., A gritty cybernetic detective enters a rain-slicked neon alleyway in Neo-Tokyo...")
        if st.button("🚀 Begin Guided Cinematic Interview", type="primary", use_container_width=True) and raw_idea:
            st.session_state["director_prompt"] = raw_idea
            st.session_state["director_answered"] = []
            st.session_state["director_current_q"] = None
            st.session_state["director_step"] = "options"
            st.rerun()

    elif step == "options":
        st.markdown('<div class="step-header"><span class="step-badge">STEP 2</span><span>Cinematic Dimension Alignment</span></div>', unsafe_allow_html=True)
        
        answered = st.session_state.get("director_answered", [])
        for i, ans in enumerate(answered):
            with st.container(border=True):
                st.markdown(f"✅ **{ans['question']}**\n\n👉 {ans['chosen_label']}")
                if st.button("✏️ Edit", key=f"edit_ans_{i}"):
                    st.session_state["director_answered"] = answered[:i]
                    st.session_state["director_current_q"] = None
                    st.rerun()
        
        if st.session_state.get("director_error"):
            st.error(f"⚠️ 連線至 Google 雲端服務時發生暫時性中斷 (Temporary network/DNS glitch):\n\n`{st.session_state['director_error']}`")
            if st.button("🔄 重試連線 (Retry Connection)", type="primary"):
                st.session_state.pop("director_error", None)
                st.rerun()

        # Enforce max questions at UI level
        elif len(answered) >= 4:
            with st.spinner("Assembling final prompt..."):
                try:
                    from agents.prompt_director import assemble_prompt
                    choices = {a["axis"]: a["chosen_fragment"] for a in answered}
                    final_res = assemble_prompt(st.session_state["director_prompt"], choices)
                    st.session_state["director_final"] = final_res.get("final_prompt")
                    st.session_state["director_breakdown"] = final_res.get("breakdown")
                    st.session_state["director_step"] = "assembled"
                    st.session_state.pop("director_error", None)
                    st.rerun()
                except Exception as e:
                    st.session_state["director_error"] = str(e)
                    st.rerun()
            
        elif st.session_state.get("director_current_q") is None:
            with st.spinner("Adapting next question..."):
                try:
                    from agents.prompt_director import next_question
                    res = next_question(st.session_state["director_prompt"], answered)
                    if res.get("done"):
                        with st.spinner("Assembling final prompt..."):
                            from agents.prompt_director import assemble_prompt
                            choices = {a["axis"]: a["chosen_fragment"] for a in answered}
                            final_res = assemble_prompt(st.session_state["director_prompt"], choices)
                            st.session_state["director_final"] = final_res.get("final_prompt")
                            st.session_state["director_breakdown"] = final_res.get("breakdown")
                            st.session_state["director_step"] = "assembled"
                        st.session_state.pop("director_error", None)
                        st.rerun()
                    else:
                        st.session_state["director_current_q"] = res
                        st.session_state.pop("director_error", None)
                        st.rerun()
                except Exception as e:
                    st.session_state["director_error"] = str(e)
                    st.rerun()
        
        current_q = st.session_state.get("director_current_q")
        if current_q:
            prog = current_q.get("progress", {})
            st.caption(f"Progress: Question {prog.get('asked', len(answered)+1)} / ~{prog.get('expected_total', 4)}")
            
            intent = current_q.get("scene_intent")
            if intent:
                with st.expander("🎬 Cinematographer's Vision", expanded=True):
                    st.markdown(f"""
<div style="background: #111726; border-radius: 6px; padding: 4px; display: grid; gap: 8px; font-size: 0.9rem;">
    <div><span style="color: #38BDF8; font-weight: 600;">👁️ Audience Feel:</span> <span style="color: #FFFFFF;">{intent.get('audience_feeling', '')}</span></div>
    <div><span style="color: #818CF8; font-weight: 600;">🎯 Character Want:</span> <span style="color: #FFFFFF;">{intent.get('character_want', '')}</span></div>
    <div><span style="color: #F59E0B; font-weight: 600;">🌑 Character Hides:</span> <span style="color: #FFFFFF;">{intent.get('character_hides', '')}</span></div>
</div>
""", unsafe_allow_html=True)
            
            st.markdown(f'<div style="background: rgba(14, 165, 233, 0.08); border-left: 3px solid #38BDF8; padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: 0.88rem; color: #BAE6FD; margin: 12px 0 16px 0;">💡 <strong>Strategic Rationale:</strong> {current_q.get("why_this_axis", "")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size: 1.12rem; font-weight: 700; color: #FFFFFF; margin-bottom: 14px;">{current_q.get("question")}</div>', unsafe_allow_html=True)
            axis = current_q.get("axis")
            
            # Sort options: recommended first
            raw_options = current_q.get('options', [])
            sorted_options = sorted(raw_options, key=lambda x: not x.get('recommended', False))
            
            # Log option_shown events in background once per question
            q_shown_key = f"guidance_shown_logged_{len(answered)}"
            if not st.session_state.get(q_shown_key):
                st.session_state[q_shown_key] = True
                def _bg_log_shown(opts=raw_options, ax=axis, a_asked=[a['axis'] for a in answered], s_id=st.session_state.get("session_id"), s_sum=st.session_state.get("director_prompt", "")):
                    try:
                        from database.logger import log_guidance_event
                        for o in opts:
                            log_guidance_event(
                                session_id=s_id,
                                event_type='option_shown',
                                axis=ax,
                                option_label=o.get('label', ''),
                                option_fragment=o.get('prompt_fragment', ''),
                                was_recommended=1 if o.get('recommended') else 0,
                                axes_asked=a_asked,
                                scene_summary=s_sum[:200]
                            )
                    except Exception:
                        pass
                import threading
                threading.Thread(target=_bg_log_shown, daemon=True).start()

            # Use columns or containers for choices instead of plain radio
            # We will use st.button for each choice so they look like distinct cards
            chosen_val = None
            chosen_label = None
            chosen_is_rec = 0
            is_custom = False

            for idx, opt in enumerate(sorted_options):
                is_rec = opt.get("recommended", False)
                with st.container(border=True):
                    rec_badge = ' <span class="rec-pill">⭐ RECOMMENDED</span>' if is_rec else ""
                    st.markdown(f"#### {opt.get('label')}{rec_badge}", unsafe_allow_html=True)
                    director_note = opt.get('director_note', '')
                    if director_note:
                        st.markdown(f'<div class="director-note-box">🎥 <strong>Director Note:</strong> {director_note}</div>', unsafe_allow_html=True)
                    btn_text = "⭐ Select Recommended Option" if is_rec else "Select Option"
                    btn_kind = "primary" if is_rec else "secondary"
                    if st.button(btn_text, key=f"sel_opt_{len(answered)}_{idx}", type=btn_kind, use_container_width=True):
                        chosen_val = opt.get('prompt_fragment')
                        chosen_label = opt.get('label')
                        chosen_is_rec = 1 if is_rec else 0

            st.divider()
            with st.container(border=True):
                st.markdown("#### Custom Input")
                custom_val = st.text_input("Enter custom logic for this dimension", key=f"custom_{len(answered)}")
                if st.button("Use Custom Input", key=f"sel_custom_{len(answered)}"):
                    if custom_val:
                        chosen_val = custom_val
                        chosen_label = f"Custom: {custom_val}"
                        is_custom = True

            if chosen_val:
                # Log option_chosen or freetext in background
                def _bg_log_choice(ev='freetext' if is_custom else 'option_chosen', ax=axis, lbl=chosen_label, frag=chosen_val, rec=chosen_is_rec, a_asked=[a['axis'] for a in answered], s_id=st.session_state.get("session_id"), s_sum=st.session_state.get("director_prompt", "")):
                    try:
                        from database.logger import log_guidance_event
                        log_guidance_event(
                            session_id=s_id,
                            event_type=ev,
                            axis=ax,
                            option_label=lbl,
                            option_fragment=frag,
                            was_recommended=rec,
                            axes_asked=a_asked,
                            scene_summary=s_sum[:200]
                        )
                    except Exception:
                        pass
                import threading
                threading.Thread(target=_bg_log_choice, daemon=True).start()

                st.session_state["director_answered"].append({
                    "axis": axis,
                    "question": current_q.get('question'),
                    "chosen_label": chosen_label,
                    "chosen_fragment": chosen_val
                })
                st.session_state["director_current_q"] = None
                st.rerun()

    elif step == "assembled":
        st.markdown('<div class="step-header"><span class="step-badge" style="background: linear-gradient(135deg, #8B5CF6, #6366F1);">STEP 3</span><span>Review & Polish Assembled Prompt</span></div>', unsafe_allow_html=True)
        final_prompt = st.text_area("Final Prompt", value=st.session_state.get("director_final", ""), height=200)
        st.session_state["director_final"] = final_prompt
        with st.expander("See Breakdown"):
            st.json(st.session_state.get("director_breakdown", {}))
            
        if st.button("Next: Generate Keyframes", type="primary", use_container_width=True):
            with st.spinner("Generating 3 candidate keyframes via Gemini Flash Image..."):
                from engine.storyboard import generate_storyboard
                kfs = []
                for i in range(3):
                    res = generate_storyboard(prompt=final_prompt, use_live_imagen=live_veo)
                    kfs.append(res.get("image_path"))
                st.session_state["director_keyframes"] = kfs
                st.session_state["director_step"] = "keyframe"
                st.rerun()

    elif step == "keyframe":
        st.markdown('<div class="step-header"><span class="step-badge" style="background: linear-gradient(135deg, #EC4899, #8B5CF6);">STEP 4</span><span>Select Opening Anchor Keyframe</span></div>', unsafe_allow_html=True)
        kfs = st.session_state.get("director_keyframes", [])
        num_cols = max(len(kfs), 1)
        cols = st.columns(num_cols)
        for idx, kf_path in enumerate(kfs):
            with cols[idx]:
                is_mock = False
                if kf_path and os.path.exists(kf_path):
                    if os.path.getsize(kf_path) < 35000:
                        is_mock = True
                    st.image(kf_path, use_container_width=True)
                else:
                    st.markdown(f'<div style="background: #111726; border: 1px dashed #EF4444; border-radius: 8px; padding: 20px; text-align: center; color: #F87171;">Keyframe #{idx+1} unavailable</div>', unsafe_allow_html=True)

                if is_mock:
                    st.caption("⚠️ 網路超時備用幀 (Simulated Fallback)")

                c_sel, c_reg = st.columns([1.3, 1])
                with c_sel:
                    btn_type = "primary" if not is_mock else "secondary"
                    if st.button(f"Select #{idx+1}", key=f"sel_{idx}", type=btn_type, use_container_width=True):
                        st.session_state["director_selected_kf"] = kf_path
                        st.session_state["director_step"] = "video"
                        st.rerun()
                with c_reg:
                    if st.button("🔄 Re-roll", key=f"reroll_{idx}", use_container_width=True, help=f"Regenerate candidate #{idx+1}"):
                        with st.spinner(f"Regenerating Keyframe #{idx+1}..."):
                            from engine.storyboard import generate_storyboard
                            res = generate_storyboard(prompt=st.session_state["director_final"], use_live_imagen=live_veo)
                            st.session_state["director_keyframes"][idx] = res.get("image_path")
                            st.rerun()
                    
        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            if st.button("🔄 Regenerate All 3 Candidates", use_container_width=True):
                with st.spinner("Generating 3 new candidates..."):
                    from engine.storyboard import generate_storyboard
                    new_kfs = []
                    for _ in range(3):
                        res = generate_storyboard(prompt=st.session_state["director_final"], use_live_imagen=live_veo)
                        new_kfs.append(res.get("image_path"))
                    st.session_state["director_keyframes"] = new_kfs
                    st.rerun()
        with col_b2:
            if st.button("✏️ Back to Edit Prompt", use_container_width=True):
                st.session_state["director_step"] = "assembled"
                st.rerun()

    elif step == "video":
        st.markdown('<div class="step-header"><span class="step-badge" style="background: linear-gradient(135deg, #10B981, #059669);">STEP 5</span><span>Ready for Omni Production Engine</span></div>', unsafe_allow_html=True)
        st.image(st.session_state["director_selected_kf"], caption="Selected Visual Reference Keyframe", use_container_width=True)
        st.success("Ready to generate video using Gemini Omni Flash Preview!")
        
        if st.button("🎬 Generate Initial Take (Gemini Omni)", type="primary", use_container_width=True):
            with st.spinner("Generating Video..."):
                from engine.generator import generate_video
                final_prompt = "Reference the attached visual image for character appearance, scene environment, and cinematic lighting style. Generate dynamic cinematic motion matching the script:\n\n" + st.session_state["director_final"]

                try:
                    gen_res = generate_video(
                        prompt=final_prompt,
                        negative_prompt="",
                        first_frame_path=st.session_state["director_selected_kf"],
                        duration_seconds=vid_duration,
                        video_engine="gemini-omni-flash-preview",
                        use_live_veo=live_veo
                    )
                    st.session_state["take_history"].append({
                        "take_num": 1,
                        "prompt": st.session_state["director_final"],
                        "video_path": gen_res.get("video_path"),
                        "ledger": None,
                        "remediated_plan": None,
                        "seed": gen_res.get("seed_used"),
                        "interaction_id": gen_res.get("interaction_id")
                    })
                    st.session_state["original_prompt"] = st.session_state["director_final"]
                    
                    from engine.claims import extract_claims
                    claims = extract_claims(st.session_state["director_final"])
                    physics_claim = {
                        "claim_text": "Universal Physics & Topology Sanity: The video MUST maintain strict topological continuity...",
                        "type": "physics_sanity",
                        "verifiable": True,
                        "temporal": "sequential",
                        "tier": "Tier 0 (Foundation)",
                        "reference_source": "system_rule"
                    }
                    claims.insert(0, physics_claim)
                    st.session_state["extracted_claims"] = claims
                    import json as _json
                    _json.dump(claims, open("temp_eval/ui_claims.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                    st.rerun()
                except Exception as e:
                    st.error(f"🚨 Generation Failed:\n\n{str(e)}")



with col2:
    current_prompt = st.session_state.get("original_prompt", st.session_state.get("director_final", ""))
    st.markdown('<div class="step-header"><span class="step-badge" style="background: linear-gradient(135deg, #0284C7, #2563EB);">TIMELINE</span><span>Production Lineage & Fine-Tuning</span></div>', unsafe_allow_html=True)
    if not st.session_state["take_history"]:
        st.info("No takes generated yet. Complete Step 1-5 to generate Take 1.")
        
    for idx, take in enumerate(st.session_state["take_history"]):
        is_latest = (idx == len(st.session_state["take_history"]) - 1)
        with st.expander(f"🎥 Take {take['take_num']}", expanded=is_latest):
            st.video(take["video_path"])
            st.info('🎬 **Prompt Used:** ' + take.get('prompt', ''))
            
            # Conversational Fine-Tuning (Omni Interactions)
            inter_id = take.get("interaction_id")
            with st.container(border=True):
                st.markdown("##### 💬 Conversational Fine-Tuning")
                if inter_id:
                    tweak_mode = st.radio(
                        "Fine-Tuning Impact Mode",
                        options=[
                            "🎬 Reshoot with Tweak (High Impact / Visibly Distinct)",
                            "✂️ Surgical In-Place Edit (Subtle V2V Patch)"
                        ],
                        index=0,
                        key=f"tweak_mode_{idx}",
                        horizontal=True,
                        help="Reshoot creates a clearly distinct take integrating your correction with the storyboard keyframe anchor. Surgical In-Place Edit applies latent video-to-video diffusion directly onto the source footage."
                    )
                    col_t1, col_t2 = st.columns([3, 1], vertical_alignment="bottom")
                    with col_t1:
                        # Check if a suggestion was pasted for this input
                        if f"pending_tweak_input_{idx}" in st.session_state:
                            st.session_state[f"tweak_input_{idx}"] = st.session_state.pop(f"pending_tweak_input_{idx}")
                        tweak_cmd = st.text_area(
                            "Fine-tune this take with one or more instructions",
                            key=f"tweak_input_{idx}",
                            height=75,
                            placeholder="e.g., Change jacket color to bright red; Add heavy pouring rain with lightning reflections in puddles"
                        )
                    with col_t2:
                        tweak_btn = st.button("✨ Apply Tweak", key=f"tweak_btn_{idx}", type="primary", use_container_width=True)

                    if tweak_btn and tweak_cmd.strip():
                        is_reshoot = tweak_mode.startswith("🎬")
                        spinner_msg = (
                            f"Omni reshooting Take {take['take_num']} with high-impact correction..."
                            if is_reshoot
                            else f"Omni performing surgical in-place edit on Take {take['take_num']}..."
                        )
                        with st.spinner(spinner_msg):
                            try:
                                from engine.generator import generate_video
                                original_scene = st.session_state.get("director_final", st.session_state.get("original_prompt", ""))
                                
                                if is_reshoot:
                                    # High-impact reshoot: ground to storyboard keyframe + inject explicit director correction
                                    full_tweak_prompt = (
                                        f"{original_scene}\n\n"
                                        f"[DIRECTOR'S CORRECTION FOR THIS TAKE]:\n"
                                        f"Execute this mandatory change with high visual prominence and bold contrast: {tweak_cmd.strip()}.\n"
                                        f"Ensure this correction is clearly noticeable and prominently featured throughout the footage."
                                    )
                                    res = generate_video(
                                        prompt=full_tweak_prompt,
                                        source_video_path=None,
                                        first_frame_path=st.session_state.get("director_selected_kf"),
                                        video_engine="gemini-omni-flash-preview",
                                        duration_seconds=vid_duration,
                                        use_live_veo=live_veo
                                    )
                                else:
                                    # Surgical V2V edit: clean, assertive prompt without diluting continuity constraints
                                    full_tweak_prompt = (
                                        f"MANDATORY VIDEO MODIFICATION DIRECTIVE:\n"
                                        f"Visibly and noticeably transform the attached video.\n"
                                        f"Execute this change with high visual prominence and physical contrast: {tweak_cmd.strip()}.\n"
                                        f"Make sure this change is immediately distinguishable from the original clip on screen."
                                    )
                                    res = generate_video(
                                        prompt=full_tweak_prompt,
                                        source_video_path=take.get("video_path"),
                                        interaction_id=inter_id,
                                        video_engine="gemini-omni-flash-preview",
                                        duration_seconds=vid_duration,
                                        use_live_veo=live_veo
                                    )
                                new_take = len(st.session_state["take_history"]) + 1
                                st.session_state["take_history"].append({
                                    "take_num": new_take,
                                    "prompt": f"[Tweak from Take {take['take_num']}]: {tweak_cmd}",
                                    "video_path": res["video_path"],
                                    "ledger": None,
                                    "remediated_plan": None,
                                    "seed": res.get("seed_used"),
                                    "interaction_id": res.get("interaction_id")
                                })
                                
                                # If this tweak originated from a pasted suggestion, mark was_applied
                                pasted_sug_id = st.session_state.get(f"last_pasted_sug_id_{idx}")
                                pasted_text = st.session_state.get(f"last_pasted_text_{idx}")
                                if pasted_sug_id and (pasted_text == tweak_cmd.strip()):
                                    def _bg_update_applied(s_id=pasted_sug_id):
                                        try:
                                            from database.logger import update_tweak_suggestion
                                            update_tweak_suggestion(suggestion_id=s_id, was_applied=True)
                                        except Exception:
                                            pass
                                    import threading
                                    threading.Thread(target=_bg_update_applied, daemon=True).start()
                                    
                                # If user sent it, also log was_sent
                                if pasted_sug_id:
                                    def _bg_update_send(s_id=pasted_sug_id):
                                        try:
                                            from database.logger import update_tweak_suggestion
                                            update_tweak_suggestion(
                                                suggestion_id=s_id, 
                                                was_sent=True, 
                                                user_final_text=tweak_cmd.strip()
                                            )
                                        except Exception:
                                            pass
                                    import threading
                                    threading.Thread(target=_bg_update_send, daemon=True).start()

                                # Telemetry tracking for guidance_events: tweak_requested
                                def _bg_log_tweak(cmd=tweak_cmd.strip(), s_id=st.session_state.get("session_id"), s_sum=st.session_state.get("original_prompt", ""), a_asked=[a['axis'] for a in st.session_state.get("director_answered", [])]):
                                    try:
                                        from agents.prompt_director import infer_tweak_axis
                                        from database.logger import log_guidance_event
                                        inferred_axis, confidence = infer_tweak_axis(cmd)
                                        log_guidance_event(
                                            session_id=s_id,
                                            event_type='tweak_requested',
                                            axis=inferred_axis,
                                            tweak_text=cmd,
                                            axes_asked=a_asked,
                                            scene_summary=s_sum[:200],
                                            axis_confidence=confidence
                                        )
                                    except Exception as e:
                                        print(f"[_bg_log_tweak ERROR]: {e}")
                                import threading
                                threading.Thread(target=_bg_log_tweak, daemon=True).start()

                                st.success("Fine-tuned take generated successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Fine-tuning failed: {e}")
                else:
                    st.caption("⚠️ No Omni Interaction ID available for this take (generated by Veo or legacy take), conversational fine-tuning unsupported.")

            # 1. VERIFICATION STAGE
            if take["ledger"] is None:
                if st.button(f"▶️ Run Verification (Take {take['take_num']})", key=f"verify_{idx}", type="primary", use_container_width=True):
                    progress_bar = st.progress(0, text="⏳ Initializing Frame Verification...")
                    status_box = st.status("🕵️ Verifying Video Frames against 4-Tier Critical Path...", expanded=True)
                    start_time = time.time()
                    ledger = []
                    for idx_claim, total, entry in stream_pipeline(
                        scene_text=st.session_state.get("original_prompt", current_prompt),
                        video_path=take["video_path"],
                        frames_dir=str(Path(f"temp_eval/frames_take_{take['take_num']}")),
                        dry_run=dry_run,
                        claims=st.session_state.get("extracted_claims")
                    ):
                        ledger.append(entry)
                        pct = int(((idx_claim + 1) / total) * 100)
                        v_icon = "✅" if entry["verdict"] == "MATCH" else ("❌" if entry["verdict"] == "MISMATCH" else "⚠️")
                        msg = f"{v_icon} **Claim #{idx_claim+1}/{total}**: {entry['claim_text']}\n"
                        msg += f"> **Verdict**: {entry['verdict']}\n"
                        if entry.get("physics_sanity"): msg += f"> **Physics**: {entry['physics_sanity']}\n"
                        if entry.get("spatial_geometry"): msg += f"> **Geometry**: {entry['spatial_geometry']}\n"
                        if entry.get("motion_anchoring"): msg += f"> **Motion**: {entry['motion_anchoring']}\n"
                        if entry.get("causality"): msg += f"> **Causality**: {entry['causality']}\n"
                        if entry.get("physics_laws"): msg += f"> **Physics Laws**: {entry['physics_laws']}\n"
                        status_box.write(msg)
                        progress_bar.progress((idx_claim + 1) / total, text=f"Evaluating Claim {idx_claim+1} of {total} ({pct}% complete)...")
                    
                    progress_bar.progress(1.0, text="✅ Verification Complete!")
                    status_box.update(label=f"✅ Verification Complete ({len(ledger)}/{len(ledger)} Claims)", state="complete", expanded=False)
                    INSPECTION_DURATION_SECONDS.observe(time.time() - start_time)
                    failed = [e for e in ledger if e.get("verdict") == "MISMATCH"]
                    if not failed:
                        TAKES_TOTAL.labels(status="passed", defect_type="none").inc()
                    else:
                        for fc in failed:
                            TAKES_TOTAL.labels(status="failed", defect_type=fc.get("type", "unknown")).inc()
                            
                    take["ledger"] = ledger
                    
                    # Compute metrics for BigQuery
                    matches = sum(1 for r in ledger if r.get("verdict") == "MATCH")
                    pass_rate = (matches / len(ledger) * 100) if ledger else 0
                    avg_set_size = sum(r.get("conformal_set_size", 1) for r in ledger) / len(ledger) if ledger else 1.0
                    
                    # Log to BigQuery (Fire and forget - Truly Async)
                    import threading
                    def _bg_log():
                        try:
                            from database.logger import log_verification_ledger, log_remediation_history_batch
                            log_verification_ledger(scene_id=st.session_state.get("scene_id", "demo_scene_01"), take_num=take["take_num"], ledger=take.get("ledger", []))
                            if take["take_num"] > 1:
                                prev_take = st.session_state["take_history"][idx - 1]
                                log_remediation_history_batch(
                                    prev_ledger=prev_take.get("ledger", []),
                                    curr_ledger=take.get("ledger", []),
                                    plan=prev_take.get("remediated_plan"),
                                    take_num_before=prev_take["take_num"],
                                    take_num_after=take["take_num"],
                                )
                        except Exception as e:
                            print(f"[ClickHouse Logger Error]: {e}")
                    threading.Thread(target=_bg_log, daemon=True).start()
                    st.rerun()
            else:
                # 2. RENDER VERIFICATION RESULTS
                matches = sum(1 for r in take["ledger"] if r.get("verdict") == "MATCH")
                pass_rate = (matches / len(take["ledger"]) * 100) if take["ledger"] else 0
                st.metric("Adherence", f"{pass_rate:.1f}%", delta="Pass" if pass_rate >= 75 else "Needs Repair")
                
                for r in take["ledger"]:
                    icon = "✅" if r.get("verdict") == "MATCH" else "❌"
                    with st.expander(f"{icon} {r['claim_text']}", expanded=True):
                        st.markdown(f"**🧐 Causal Analysis:** {r.get('event_causal_order', 'N/A')}")
                        st.markdown(f"**🎥 Frame Observations:** {r.get('frame_observations', 'N/A')}")
                
                # --- Suggest Tweaks Block (Delta-driven Tweak Recommendations) ---
                failed_items = [r for r in take["ledger"] if r.get("verdict") in ("MISMATCH", "CANNOT_DETERMINE")]
                if failed_items:
                    # Auto-generate suggestions if not already generated for this take
                    if "tweak_suggestions" not in take:
                        with st.spinner("Analyzing verification ledger and synthesizing tweak recommendations..."):
                            try:
                                from agents.prompt_director import suggest_tweaks
                                choices_dict = {}
                                for ans in st.session_state.get("director_answered", []):
                                    choices_dict[ans.get("axis")] = ans.get("chosen_fragment")
                                res_sug = suggest_tweaks(
                                    ledger=take["ledger"],
                                    director_choices=choices_dict,
                                    final_prompt=take.get("prompt", "")
                                )
                                raw_sug = res_sug.get("suggestions", [])
                                import uuid
                                for s in raw_sug:
                                    s["suggestion_id"] = str(uuid.uuid4())[:8]
                                take["tweak_suggestions"] = raw_sug

                                # Optional ClickHouse Logging
                                import threading
                                def _bg_log_sug():
                                    try:
                                        from database.logger import log_tweak_suggestions
                                        log_tweak_suggestions(
                                            suggestions=take["tweak_suggestions"],
                                            session_id=st.session_state.get("scene_id", "demo_session"),
                                            take_num=take["take_num"]
                                        )
                                    except Exception:
                                        pass
                                threading.Thread(target=_bg_log_sug, daemon=True).start()
                            except Exception as e:
                                print(f"[Suggest Tweaks Error]: {e}")
                                take["tweak_suggestions"] = []

                    sugs = take.get("tweak_suggestions", [])
                    if sugs:
                        col_hdr1, col_hdr2 = st.columns([2.5, 1.3], vertical_alignment="center")
                        with col_hdr1:
                            st.markdown("#### 🔧 Detected Issues & Tweak Suggestions")
                            st.caption("Root causes deduplicated and prioritized by impact.")
                        with col_hdr2:
                            if len(sugs) > 1:
                                if st.button("📋 Paste All Suggestions", key=f"paste_all_{idx}", use_container_width=True):
                                    all_instructions = [s.get("tweak_instruction", "").strip() for s in sugs if s.get("tweak_instruction")]
                                    st.session_state[f"pending_tweak_input_{idx}"] = "; ".join(all_instructions)
                                    for s in sugs:
                                        if s.get("suggestion_id"):
                                            def _bg_update_paste_all(s_id=s.get("suggestion_id")):
                                                try:
                                                    from database.logger import update_tweak_suggestion
                                                    update_tweak_suggestion(suggestion_id=s_id, was_pasted=True)
                                                except Exception:
                                                    pass
                                            import threading
                                            threading.Thread(target=_bg_update_paste_all, daemon=True).start()
                                    st.success("All suggestions merged into tweak box!")
                                    st.rerun()

                        current_box_text = st.session_state.get(f"tweak_input_{idx}", "").strip()

                        for s_idx, sug in enumerate(sugs):
                            severity_badge = {
                                "high": "🔴 High Severity",
                                "medium": "🟡 Medium Severity",
                                "low": "🟢 Low / May be intentional"
                            }.get(sug.get("severity", "medium"), "🟡 Medium")

                            tweak_text = sug.get("tweak_instruction", "").strip()
                            already_in_box = tweak_text in current_box_text if current_box_text else False

                            with st.container(border=True):
                                col_sug_text, col_sug_btn = st.columns([3, 1], vertical_alignment="center")
                                with col_sug_text:
                                    ts_badge = f"⏱️ `{sug.get('timestamp_range', 'Whole Clip')}`"
                                    st.markdown(f"**⚠️ Current Defect:** {sug.get('issue', '')} &nbsp; {ts_badge} &nbsp; `{severity_badge}`")
                                    if sug.get("related_claims"):
                                        st.caption(f"Related claims: {', '.join(sug.get('related_claims', []))}")
                                    st.markdown("**🎯 Surgical Tweak Directive:**")
                                    st.code(tweak_text, language="text")
                                with col_sug_btn:
                                    if already_in_box:
                                        btn_label = "✅ In Tweak Box"
                                    elif current_box_text:
                                        btn_label = "➕ Append to Box"
                                    else:
                                        btn_label = "📋 Paste to Box"

                                    if st.button(btn_label, key=f"paste_sug_{idx}_{s_idx}", use_container_width=True):
                                        if current_box_text and not already_in_box:
                                            merged_val = f"{current_box_text}; {tweak_text}"
                                        else:
                                            merged_val = tweak_text
                                        st.session_state[f"pending_tweak_input_{idx}"] = merged_val
                                        st.session_state[f"last_pasted_sug_id_{idx}"] = sug.get("suggestion_id")
                                        st.session_state[f"last_pasted_text_{idx}"] = tweak_text
                                        # Update ClickHouse was_pasted
                                        def _bg_update_paste(s_id=sug.get("suggestion_id")):
                                            try:
                                                from database.logger import update_tweak_suggestion
                                                update_tweak_suggestion(suggestion_id=s_id, was_pasted=True)
                                            except Exception:
                                                pass
                                        import threading
                                        threading.Thread(target=_bg_update_paste, daemon=True).start()
                                        st.success("Tweak updated in input box!")
                                        st.rerun()

                if is_latest:
                    st.divider()
                    if st.button("✂️ Extend Video (Next 4s)", key=f"extend_{idx}"):
                        with st.spinner("Agent synthesizing extension..."):
                            try:
                                ext_prompt = st.session_state.get('original_prompt', current_prompt) + " (Continuing the timeline naturally)"
                                res = generate_video(
                                    prompt=ext_prompt,
                                    source_video_path=take["video_path"],
                                    last_frame_path=st.session_state.get("last_frame_path"),
                                    duration_seconds=vid_duration,
                                    video_engine=vid_engine.split(" ")[0],
                                    use_live_veo=live_veo
                                )
                                new_take = take["take_num"] + 1
                                st.session_state["take_history"].append({
                                    "take_num": new_take,
                                    "prompt": ext_prompt,
                                    "video_path": res["video_path"],
                                    "ledger": None,
                                    "remediated_plan": None,
                                    "interaction_id": res.get("interaction_id")
                                })
                                st.rerun()
                            except Exception as e:
                                st.error(f"🚨 Video Extension Failed:\n\n{str(e)}")
