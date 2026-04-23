from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class DriftStatus(str, Enum):
    STABLE = "stable"
    WARNING = "warning"
    DRIFT = "drift"
    CRITICAL = "critical"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class FeatureDrift(BaseModel):
    feature_name: str
    feature_type: str
    psi: float
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    js_divergence: Optional[float] = None
    drift_status: DriftStatus
    train_mean: Optional[float] = None
    prod_mean: Optional[float] = None
    mean_shift_pct: Optional[float] = None
    train_distribution: list[float] = []
    prod_distribution: list[float] = []
    bins: list[float] = []


class DriftReport(BaseModel):
    report_id: str
    generated_at: datetime
    window_start: datetime
    window_end: datetime
    sample_size: int
    reference_size: int
    overall_drift_status: DriftStatus
    drifted_feature_count: int
    total_feature_count: int
    features: list[FeatureDrift]
    summary: str


class PerformanceSnapshot(BaseModel):
    snapshot_id: str
    model_version: str
    timestamp: datetime
    window_hours: int
    sample_count: int
    labeled_count: int
    auc_roc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    avg_prediction_score: float
    prediction_std: float
    high_risk_rate: float
    data_source: str


class ModelComparison(BaseModel):
    model_a: str
    model_b: str
    comparison_period_days: int
    sample_count_a: int
    sample_count_b: int
    auc_a: Optional[float]
    auc_b: Optional[float]
    f1_a: Optional[float]
    f1_b: Optional[float]
    score_dist_a: list[float]
    score_dist_b: list[float]
    winner: Optional[str]
    confidence: Optional[float]
    recommendation: str


class Alert(BaseModel):
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    metric_value: float
    threshold: float
    model_version: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class RetrainTrigger(BaseModel):
    reason: str
    triggered_by: str
    drift_features: list[str] = []
    performance_drop: Optional[float] = None


class RetrainStatus(BaseModel):
    job_id: str
    status: str
    triggered_at: datetime
    completed_at: Optional[datetime] = None
    reason: str
    new_model_version: Optional[str] = None


class LivePrediction(BaseModel):
    transaction_id: str
    score: float
    risk_level: str
    latency_ms: float
    timestamp: datetime
    model_version: str


class DashboardSummary(BaseModel):
    model_version: str
    model_status: str
    drift_status: DriftStatus
    last_drift_check: datetime
    drifted_features: int
    auc_7d: Optional[float]
    auc_baseline: Optional[float]
    auc_delta: Optional[float]
    predictions_today: int
    fraud_rate_today: float
    fraud_rate_baseline: float
    active_alerts: int
    last_retrain: Optional[datetime]
    days_since_retrain: Optional[int]
    p99_latency_ms: float
