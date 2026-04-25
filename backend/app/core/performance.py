"""Performance Tracker - rolling AUC/F1 with delayed label handling"""
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
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
        labels = predictions.get("label") if hasattr(predictions, 'get') else predictions["label"] if "label" in predictions.columns else None
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
                    if len(np.unique(y_true)) > 1:
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
