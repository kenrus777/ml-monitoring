"""
Performance Tracker — rolling AUC/F1/Precision/Recall over time.
Handles the delayed label problem: labels arrive 30-90 days late in fraud detection.
Short-term: proxy metrics. Long-term: delayed labeled batches.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from typing import Optional
import uuid

from app.models.schemas import PerformanceSnapshot


class PerformanceTracker:
    BASELINE = {
        "auc_roc": 0.987, "precision": 0.921,
        "recall": 0.884, "f1_score": 0.902,
        "fraud_rate": 0.0017, "avg_score": 0.142,
    }

    def compute_snapshot(self, predictions, model_version, window_hours=24):
        scores = predictions["score"].values
        labels = predictions.get("label")
        auc = precision = recall = f1 = None
        labeled_count = 0

        if labels is not None:
            labeled = predictions.dropna(subset=["label"])
            labeled_count = len(labeled)
            if labeled_count >= 50:
                y_true = labeled["label"].astype(int).values
                y_score = labeled["score"].values
                y_pred = (y_score >= 0.5).astype(int)
                try:
                    auc = float(roc_auc_score(y_true, y_score))
                    precision = float(precision_score(y_true, y_pred, zero_division=0))
                    recall = float(recall_score(y_true, y_pred, zero_division=0))
                    f1 = float(f1_score(y_true, y_pred, zero_division=0))
                except Exception:
                    pass

        return PerformanceSnapshot(
            snapshot_id=str(uuid.uuid4()),
            model_version=model_version,
            timestamp=datetime.utcnow(),
            window_hours=window_hours,
            sample_count=len(predictions),
            labeled_count=labeled_count,
            auc_roc=round(auc, 4) if auc else None,
            precision=round(precision, 4) if precision else None,
            recall=round(recall, 4) if recall else None,
            f1_score=round(f1, 4) if f1 else None,
            avg_prediction_score=round(float(np.mean(scores)), 4),
            prediction_std=round(float(np.std(scores)), 4),
            high_risk_rate=round(float(np.mean(scores > 0.7)), 4),
            data_source="labeled" if labeled_count > 50 else "proxy",
        )

    def check_performance_alerts(self, snapshot):
        if snapshot.auc_roc is None:
            return []
        alerts = []
        for attr, baseline, threshold, label in [
            ("auc_roc",   self.BASELINE["auc_roc"],   0.02, "AUC-ROC"),
            ("precision", self.BASELINE["precision"], 0.05, "Precision"),
            ("recall",    self.BASELINE["recall"],    0.03, "Recall"),
            ("f1_score",  self.BASELINE["f1_score"],  0.04, "F1 Score"),
        ]:
            current = getattr(snapshot, attr)
            if current is None:
                continue
            drop = baseline - current
            if drop > threshold:
                severity = "critical" if drop > threshold * 1.5 else "warning"
                alerts.append({
                    "alert_type": "performance", "severity": severity,
                    "title": f"{label} degraded",
                    "description": f"{label} dropped from {baseline:.3f} to {current:.3f} ({drop:.1%}).",
                    "metric_name": attr, "metric_value": current,
                    "threshold": baseline - threshold, "model_version": snapshot.model_version,
                })
        return alerts

    def compute_score_distribution(self, scores, n_bins=20):
        hist, edges = np.histogram(scores, bins=n_bins, range=(0, 1))
        p = np.percentile(scores, [5, 25, 50, 75, 95])
        return {
            "histogram": hist.tolist(),
            "bin_edges": [round(e, 3) for e in edges.tolist()],
            "mean": round(float(np.mean(scores)), 4),
            "std": round(float(np.std(scores)), 4),
            "p50": round(float(p[2]), 4), "p95": round(float(p[4]), 4),
            "high_risk_pct": round(float(np.mean(scores > 0.7) * 100), 2),
        }
