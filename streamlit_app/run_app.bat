@echo off
REM Windows batch script to run the Streamlit app
cd /d "%~dp0"
streamlit run app.py
pause

