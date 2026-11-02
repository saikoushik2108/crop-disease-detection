@echo off
echo ========================================================
echo Starting Paddy Disease AI Backend Server on Port 8000...
echo ========================================================
echo.
cd /d "%~dp0"
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
pause

