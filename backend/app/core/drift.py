"""Drift Detection Engine - PSI + KS + Jensen-Shannon"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from dataclasses import dataclass
from datetime import datetime
import uuid

from app.models.schemas import FeatureDrift, DriftReport, DriftStatus


@dataclass
class DriftConfig:
    psi_warning: float = 0.1
    psi_critical: float = 0.2
    ks_pvalue_threshold: float = 0.05
    n_bins: int = 10


class DriftDetector:
    def __init__(self, reference_data, feature_columns, categorical_columns=None, config=None):
        self.reference = reference_data
        self.features = feature_columns
        self.categoricals = set(categorical_columns or [])
        self.config = config or DriftConfig()

    def compute_drift(self, production_data, window_start=None, window_end=None):
        feature_reports = []
        for feature in self.features:
            if feature not in self.reference.columns or feature not in production_data.columns:
                continue
            ref_vals = self.reference[feature].dropna()
            prod_vals = production_data[feature].dropna()
            if len(prod_vals) < 30:
                continue
            feature_reports.append(
                self._compute_feature_drift(feature, ref_vals, prod_vals, feature in self.categoricals)
            )

        drifted = [f for f in feature_reports if f.drift_status != DriftStatus.STABLE]
        critical = [f for f in feature_reports if f.drift_status == DriftStatus.CRITICAL]

        if critical:
            overall = DriftStatus.CRITICAL
        elif len(drifted) >= 3:
            overall = DriftStatus.DRIFT
        elif drifted:
            overall = DriftStatus.WARNING
        else:
            overall = DriftStatus.STABLE

        return DriftReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.utcnow(),
            window_start=window_start or datetime.utcnow(),
            window_end=window_end or datetime.utcnow(),
            sample_size=len(production_data),
            reference_size=len(self.reference),
            overall_drift_status=overall,
            drifted_feature_count=len(drifted),
            total_feature_count=len(feature_reports),
            features=feature_reports,
            summary=self._build_summary(feature_reports, overall),
        )

    def _compute_feature_drift(self, feature, ref, prod, is_categorical):
        if is_categorical:
            psi = self._psi_categorical(ref, prod)
            ks_stat, ks_pval = None, None
            js_div = self._js_categorical(ref, prod)
            bins, ref_dist, prod_dist = [], [], []
        else:
            psi, bins, ref_dist, prod_dist = self._psi_numerical(ref, prod)
            ks_stat, ks_pval = stats.ks_2samp(ref.values, prod.values)
            js_div = self._js_numerical(ref, prod)

        if psi >= self.config.psi_critical:
            status = DriftStatus.CRITICAL
        elif psi >= self.config.psi_warning:
            status = DriftStatus.WARNING
        elif ks_pval is not None and ks_pval < self.config.ks_pvalue_threshold:
            status = DriftStatus.WARNING
        else:
            status = DriftStatus.STABLE

        ref_mean = float(ref.mean()) if not is_categorical else None
        prod_mean = float(prod.mean()) if not is_categorical else None
        mean_shift = None
        if ref_mean and prod_mean and ref_mean != 0:
            mean_shift = abs(prod_mean - ref_mean) / abs(ref_mean) * 100

        return FeatureDrift(
            feature_name=feature,
            feature_type="categorical" if is_categorical else "numerical",
            psi=round(psi, 4),
            ks_statistic=round(ks_stat, 4) if ks_stat else None,
            ks_pvalue=round(ks_pval, 4) if ks_pval else None,
            js_divergence=round(js_div, 4),
            drift_status=status,
            train_mean=round(ref_mean, 4) if ref_mean else None,
            prod_mean=round(prod_mean, 4) if prod_mean else None,
            mean_shift_pct=round(mean_shift, 2) if mean_shift else None,
            train_distribution=[round(v, 4) for v in ref_dist],
            prod_distribution=[round(v, 4) for v in prod_dist],
            bins=[round(b, 4) for b in bins],
        )

    def _psi_numerical(self, ref, prod):
        bins = np.percentile(ref, np.linspace(0, 100, self.config.n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 3:
            return 0.0, [], [], []
        eps = 1e-6
        ref_pct = (np.histogram(ref, bins=bins)[0] / len(ref)) + eps
        prod_pct = (np.histogram(prod, bins=bins)[0] / len(prod)) + eps
        psi = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
        return psi, bins.tolist(), ref_pct.tolist(), prod_pct.tolist()

    def _psi_categorical(self, ref, prod):
        eps = 1e-6
        return float(sum(
            ((prod == c).mean() + eps - (ref == c).mean() - eps) *
            np.log(((prod == c).mean() + eps) / ((ref == c).mean() + eps))
            for c in set(ref.unique()) | set(prod.unique())
        ))

    def _js_numerical(self, ref, prod):
        bins = np.linspace(min(ref.min(), prod.min()), max(ref.max(), prod.max()), self.config.n_bins + 1)
        rh = np.histogram(ref, bins=bins, density=True)[0] + 1e-10
        ph = np.histogram(prod, bins=bins, density=True)[0] + 1e-10
        return float(jensenshannon(rh / rh.sum(), ph / ph.sum()))

    def _js_categorical(self, ref, prod):
        cats = sorted(set(ref.unique()) | set(prod.unique()))
        rp = np.array([(ref == c).mean() + 1e-10 for c in cats])
        pp = np.array([(prod == c).mean() + 1e-10 for c in cats])
        return float(jensenshannon(rp / rp.sum(), pp / pp.sum()))

    def _build_summary(self, features, overall):
        drifted = [f for f in features if f.drift_status != DriftStatus.STABLE]
        if not drifted:
            return "All features stable. No retraining required."
        top = ", ".join(f.feature_name for f in sorted(drifted, key=lambda f: f.psi, reverse=True)[:3])
        return (f"{len(drifted)}/{len(features)} features drifted. Highest PSI: {top}. "
                f"Status: {overall.value.upper()}. "
                f"{'Retraining recommended.' if overall == DriftStatus.CRITICAL else 'Monitor closely.'}")
