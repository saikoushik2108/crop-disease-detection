@echo off
echo ========================================================
echo Starting Paddy Disease AI (Backend + Frontend)
echo ========================================================
echo.
cd /d "%~dp0"

echo [1/2] Starting Python FastAPI Backend on http://127.0.0.1:8000 ...
start "Paddy AI Backend" .venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

echo [2/2] Starting Frontend Vite Server on http://localhost:5173 ...
cd frontend
npm run dev

