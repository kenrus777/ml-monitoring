"""FastAPI Application - ML Monitoring Dashboard"""
import uuid
import random
import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from app.core.drift import DriftDetector
from app.core.performance import PerformanceTracker
from app.models.schemas import (
    DriftReport, DashboardSummary, Alert, AlertSeverity, DriftStatus,
    RetrainTrigger, RetrainStatus, LivePrediction, ModelComparison,
)

app = FastAPI(title="ML Monitoring API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(os.environ.get('DATA_DIR', './data'))
FEATURES = [
    "amount", "amount_zscore", "txn_count_1h", "txn_count_24h",
    "hour_of_day", "is_weekend", "is_online", "is_foreign",
    "merchant_risk", "card_age_days", "unique_merch_24h",
]

# Cached dataframes — generated once on first load
_ref_df = None
_prod_df = None


def load():
    global _ref_df, _prod_df
    if _ref_df is not None:
        return _ref_df, _prod_df
    ref_path = DATA_DIR / "reference.parquet"
    prod_path = DATA_DIR / "production.parquet"
    if not ref_path.exists():
        from app.core.seed_data import seed_all
        seed_all()
    _ref_df = pd.read_parquet(ref_path)
    _prod_df = pd.read_parquet(prod_path)
    return _ref_df, _prod_df


@app.on_event("startup")
async def startup_event():
    """Pre-generate data on startup so first request is fast."""
    try:
        load()
        print("Data loaded successfully")
    except Exception as e:
        print(f"Warning: could not pre-load data: {e}")


@app.get("/")
async def root():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/summary", response_model=DashboardSummary)
async def summary():
    ref, prod = load()
    recent = prod[prod["day"] >= 27]
    report = DriftDetector(ref, FEATURES).compute_drift(recent)
    snap = PerformanceTracker().compute_snapshot(recent, "v1.2.0", 72)
    return DashboardSummary(
        model_version="v1.2.0",
        model_status="warning" if report.overall_drift_status != DriftStatus.STABLE else "healthy",
        drift_status=report.overall_drift_status,
        last_drift_check=datetime.utcnow(),
        drifted_features=report.drifted_feature_count,
        auc_7d=snap.auc_roc,
        auc_baseline=PerformanceTracker.BASELINE["auc_roc"],
        auc_delta=(snap.auc_roc - PerformanceTracker.BASELINE["auc_roc"]) if snap.auc_roc else None,
        predictions_today=len(recent),
        fraud_rate_today=round(float((recent["score"] > 0.7).mean()), 4),
        fraud_rate_baseline=0.0017,
        active_alerts=report.drifted_feature_count,
        last_retrain=datetime.utcnow() - timedelta(days=45),
        days_since_retrain=45,
        p99_latency_ms=float(np.percentile(recent["latency_ms"], 99)),
    )


@app.get("/drift/report", response_model=DriftReport)
async def drift_report(days: int = Query(default=7, ge=1, le=30)):
    ref, prod = load()
    recent = prod[prod["day"] >= (30 - days)]
    return DriftDetector(ref, FEATURES).compute_drift(
        recent,
        window_start=datetime.utcnow() - timedelta(days=days),
        window_end=datetime.utcnow(),
    )


@app.get("/drift/timeline")
async def drift_timeline():
    ref, prod = load()
    det = DriftDetector(ref, FEATURES)
    tl = []
    for day in range(0, 30, 3):
        w = prod[(prod["day"] >= day) & (prod["day"] < day + 7)]
        if len(w) < 50:
            continue
        r = det.compute_drift(w)
        tl.append({
            "day": day,
            "date": (datetime.utcnow() - timedelta(days=30 - day)).strftime("%m/%d"),
            "overall_status": r.overall_drift_status.value,
            "features": {
                f.feature_name: {"psi": f.psi, "status": f.drift_status.value}
                for f in r.features
            }
        })
    return {"timeline": tl}


@app.get("/performance/timeline")
async def perf_timeline():
    ref, prod = load()
    tracker = PerformanceTracker()
    snaps = []
    for day in range(7, 30, 3):
        w = prod[(prod["day"] >= day - 7) & (prod["day"] < day)]
        if len(w) < 50:
            continue
        s = tracker.compute_snapshot(w, "v1.2.0", 168)
        snaps.append({
            "day": day,
            "date": (datetime.utcnow() - timedelta(days=30 - day)).strftime("%m/%d"),
            "auc_roc": s.auc_roc,
            "precision": s.precision,
            "recall": s.recall,
            "f1_score": s.f1_score,
            "avg_score": s.avg_prediction_score,
            "high_risk_rate": s.high_risk_rate,
            "sample_count": s.sample_count,
        })
    return {"snapshots": snaps, "baseline": PerformanceTracker.BASELINE}


@app.get("/alerts", response_model=list[Alert])
async def alerts():
    ref, prod = load()
    report = DriftDetector(ref, FEATURES).compute_drift(prod[prod["day"] >= 23])
    return [
        Alert(
            alert_id=str(uuid.uuid4()),
            alert_type="drift",
            severity=AlertSeverity.CRITICAL if f.drift_status == DriftStatus.CRITICAL else AlertSeverity.WARNING,
            title=f"Feature drift: {f.feature_name}",
            description=f"{f.feature_name} PSI={f.psi:.3f}. {'Retrain required.' if f.psi > 0.2 else 'Monitor closely.'}",
            metric_name="psi",
            metric_value=f.psi,
            threshold=0.2 if f.drift_status == DriftStatus.CRITICAL else 0.1,
            model_version="v1.2.0",
            timestamp=datetime.utcnow() - timedelta(hours=random.randint(1, 12)),
        )
        for f in report.features if f.drift_status != DriftStatus.STABLE
    ]


@app.post("/retrain/trigger", response_model=RetrainStatus)
async def retrain(body: RetrainTrigger):
    return RetrainStatus(
        job_id=str(uuid.uuid4()),
        status="queued",
        triggered_at=datetime.utcnow(),
        reason=body.reason
    )


@app.get("/model/comparison", response_model=ModelComparison)
async def comparison():
    ref, prod = load()
    rng = np.random.RandomState(42)
    sa = prod["score"].values
    sb = np.clip(sa + rng.normal(0.01, 0.02, len(sa)), 0, 1)
    return ModelComparison(
        model_a="v1.2.0 (production)",
        model_b="v1.3.0-challenger (shadow)",
        comparison_period_days=7,
        sample_count_a=len(sa),
        sample_count_b=len(sb),
        auc_a=0.987, auc_b=0.991,
        f1_a=0.902, f1_b=0.914,
        score_dist_a=list(np.histogram(sa, bins=20, range=(0, 1))[0].astype(float)),
        score_dist_b=list(np.histogram(sb, bins=20, range=(0, 1))[0].astype(float)),
        winner="v1.3.0-challenger",
        confidence=0.94,
        recommendation="Challenger shows +0.4% AUC. Recommend canary deploy at 10% traffic.",
    )


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    rng = np.random.RandomState()
    try:
        while True:
            import asyncio
            await asyncio.sleep(0.8)
            score = float(rng.beta(1.2, 8))
            risk = "CRITICAL" if score > 0.85 else "HIGH" if score > 0.7 else "MEDIUM" if score > 0.4 else "LOW"
            await websocket.send_json(LivePrediction(
                transaction_id=f"TXN-{uuid.uuid4().hex[:8].upper()}",
                score=round(score, 4),
                risk_level=risk,
                latency_ms=round(float(rng.gamma(2, 4)), 1),
                timestamp=datetime.utcnow(),
                model_version="v1.2.0"
            ).model_dump(mode="json"))
    except WebSocketDisconnect:
        pass
