# Deployment Guide

This document provides instructions on how to deploy and run the Multimodal Paddy Disease AI system.

## 1. Backend Setup (FastAPI)
### Environment
- Python 3.10+
- Install dependencies: `pip install -r requirements.txt`

### Configuration
Create a `.env` file in the root directory:
```env
WEATHER_API_KEY=your_key_here
MODEL_PATH=models/multimodal_best.pt
BACKEND_URL=http://localhost:8000
```

### Execution
```bash
# Run using uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 2. Frontend Setup (React)
### Environment
- Node.js 18+
- npm or yarn

### Execution
```bash
cd frontend
npm install
npm run dev
```
The frontend will typically be available at `http://localhost:5173`.

## 3. Dataset & Model Setup
- **Dataset**: Follow `data/README.md` to place the Paddy Doctor dataset in `data/raw/`.
- **Models**: Follow `models/README.md` to place the trained weights in `models/`.

## 4. Troubleshooting
- **DLL Load Error**: If PyTorch fails to load, ensure the Microsoft Visual C++ Redistributable is installed.
- **API 403**: Ensure the `WEATHER_API_KEY` is valid and the `User-Agent` header is set for Nominatim requests.
