@echo off
title Security Breach Detection Research Project
echo ===============================================
echo   Security Breach Detection - ML Research
echo ===============================================
echo.
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 or 3.12 and tick "Add Python to PATH".
  pause
  exit /b 1
)
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate
echo Installing required packages...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Starting research dashboard...
python -m streamlit run app.py
pause
