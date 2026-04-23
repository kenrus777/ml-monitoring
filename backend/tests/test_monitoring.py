import pytest, numpy as np, pandas as pd
from datetime import datetime
from app.core.drift import DriftDetector, DriftConfig
from app.core.performance import PerformanceTracker
from app.models.schemas import DriftStatus, PerformanceSnapshot

@pytest.fixture
def ref():
    rng = np.random.RandomState(42)
    return pd.DataFrame({"amount":rng.lognormal(4.5,1.2,1000),"txn_count_1h":rng.poisson(2.1,1000).astype(float),"is_online":rng.binomial(1,0.35,1000).astype(float),"merchant_risk":rng.beta(1.5,20,1000)})

@pytest.fixture
def stable(ref):
    rng = np.random.RandomState(99)
    return pd.DataFrame({"amount":rng.lognormal(4.5,1.2,500),"txn_count_1h":rng.poisson(2.1,500).astype(float),"is_online":rng.binomial(1,0.35,500).astype(float),"merchant_risk":rng.beta(1.5,20,500)})

@pytest.fixture
def drifted():
    rng = np.random.RandomState(77)
    return pd.DataFrame({"amount":rng.lognormal(5.5,1.8,500),"txn_count_1h":rng.poisson(6.0,500).astype(float),"is_online":rng.binomial(1,0.75,500).astype(float),"merchant_risk":rng.beta(3.0,10,500)})

FEATS = ["amount","txn_count_1h","is_online","merchant_risk"]

class TestDrift:
    def test_stable_no_critical(self, ref, stable):
        r = DriftDetector(ref, FEATS).compute_drift(stable)
        assert r.overall_drift_status in [DriftStatus.STABLE, DriftStatus.WARNING]
        assert r.total_feature_count == 4

    def test_drifted_triggers_alert(self, ref, drifted):
        r = DriftDetector(ref, FEATS).compute_drift(drifted)
        assert r.overall_drift_status in [DriftStatus.WARNING, DriftStatus.DRIFT, DriftStatus.CRITICAL]
        assert r.drifted_feature_count > 0

    def test_psi_non_negative(self, ref, drifted):
        r = DriftDetector(ref, FEATS).compute_drift(drifted)
        for f in r.features:
            assert f.psi >= 0.0

    def test_has_summary(self, ref, stable):
        r = DriftDetector(ref, ["amount"]).compute_drift(stable)
        assert isinstance(r.summary, str) and len(r.summary) > 5

class TestPerformance:
    def test_no_labels_returns_proxy(self):
        rng = np.random.RandomState(42)
        preds = pd.DataFrame({"score": rng.beta(1.2, 8, 200)})
        snap = PerformanceTracker().compute_snapshot(preds, "v1.0.0")
        assert snap.auc_roc is None
        assert snap.data_source == "proxy"

    def test_degraded_metrics_alert(self):
        tracker = PerformanceTracker()
        bad = PerformanceSnapshot(
            snapshot_id="t", model_version="v1", timestamp=datetime.utcnow(),
            window_hours=24, sample_count=1000, labeled_count=500,
            auc_roc=0.920, precision=0.800, recall=0.700, f1_score=0.750,
            avg_prediction_score=0.14, prediction_std=0.18, high_risk_rate=0.04,
            data_source="labeled")
        alerts = tracker.check_performance_alerts(bad)
        assert len(alerts) > 0
        assert any(a["severity"] == "critical" for a in alerts)

    def test_good_metrics_no_alert(self):
        tracker = PerformanceTracker()
        good = PerformanceSnapshot(
            snapshot_id="t", model_version="v1", timestamp=datetime.utcnow(),
            window_hours=24, sample_count=1000, labeled_count=500,
            auc_roc=0.985, precision=0.918, recall=0.881, f1_score=0.899,
            avg_prediction_score=0.14, prediction_std=0.18, high_risk_rate=0.04,
            data_source="labeled")
        assert len(tracker.check_performance_alerts(good)) == 0
