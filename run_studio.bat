@echo off
title CineQA-Agent Studio Launcher
echo ============================================================
echo Starting CineQA Agent Studio with Google Vertex AI & Grafana
echo ============================================================
cd /d "%~dp0"
python -m streamlit run ui/app.py
pause
