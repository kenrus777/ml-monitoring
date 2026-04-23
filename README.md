# 📊 ML Model Monitoring Dashboard

> Production-grade ML observability — drift detection, performance tracking, automated alerts.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org) [![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com) [![React](https://img.shields.io/badge/React-18-61dafb)](https://react.dev)

## Architecture

```
React Dashboard (PSI Heatmap, Performance Charts, Live Feed, Alerts)
    ↓ REST + WebSocket
FastAPI Backend
    GET /summary              → Dashboard KPIs
    GET /drift/report         → PSI + KS + JS per feature
    GET /drift/timeline       → 30-day heatmap data
    GET /performance/timeline → AUC/F1 over time
    GET /alerts               → Active threshold alerts
    POST /retrain/trigger     → Kick off retraining
    WS  /ws/live              → Real-time prediction stream
    ↓
Monitoring Engine (PSI, KS Test, Jensen-Shannon divergence)
    ↓
Data (PostgreSQL + Redis)
```

## What It Monitors

| Method | Detects | Alert threshold |
|--------|---------|-----------------|
| **PSI** | Feature distribution shift | PSI > 0.2 → retrain |
| **KS Test** | Distribution change | p-value < 0.05 |
| **AUC-ROC delta** | Accuracy drop | > 2% from baseline |
| **Recall drop** | Missing more fraud | > 3% from baseline |

## Quick Start

```bash
# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m app.core.seed_data     # Generate 30 days demo data
uvicorn app.main:app --reload --port 8000
# API docs: http://localhost:8000/docs

# Frontend
cd frontend && npm install && npm run dev
# Dashboard: http://localhost:5173

# Docker
docker-compose up --build
```

## Interview Q&A

**Q: What is PSI and why use it?**
PSI measures feature distribution shift. PSI < 0.1 = stable, 0.1-0.2 = warning, > 0.2 = retrain. Finance standard — single interpretable number per feature.

**Q: Data drift vs concept drift?**
Data drift = input distributions change. Concept drift = feature→target relationship changes. Detect separately with different methods.

**Q: Ground truth delay problem?**
Fraud labels arrive 30-90 days late. Short-term: proxy metrics (score distribution). Long-term: delayed labeled batches.

**Q: When do you retrain?**
Three triggers: (1) Weekly scheduled, (2) AUC drops 2%+ on labeled data, (3) PSI > 0.2 on 3+ key features simultaneously.

## Project Structure

```
ml-monitoring/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── drift.py        ← PSI + KS + JS detection engine
│   │   │   ├── performance.py  ← Rolling AUC/F1 tracker
│   │   │   └── seed_data.py    ← 30-day synthetic data generator
│   │   ├── models/
│   │   │   └── schemas.py      ← Pydantic schemas
│   │   └── main.py             ← FastAPI + WebSocket
│   ├── tests/
│   │   └── test_monitoring.py  ← pytest suite
│   └── requirements.txt
├── frontend/
│   └── src/
│       └── App.jsx             ← React dashboard
├── docker-compose.yml
└── README.md
```
