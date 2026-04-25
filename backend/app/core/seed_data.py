"""
Synthetic Data Seeder
Run: python backend/app/core/seed_data.py
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid
import os
from pathlib import Path

# DATA_DIR env var lets us override at runtime
DATA_DIR = Path(os.environ.get('DATA_DIR', './data'))


def generate_reference_data(n=10000, seed=42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "amount":           rng.lognormal(mean=4.5, sigma=1.2, size=n),
        "amount_zscore":    rng.normal(0, 1, n),
        "txn_count_1h":     rng.poisson(2.1, n).astype(float),
        "txn_count_24h":    rng.poisson(8.5, n).astype(float),
        "hour_of_day":      rng.randint(0, 24, n).astype(float),
        "is_weekend":       rng.binomial(1, 0.28, n).astype(float),
        "is_online":        rng.binomial(1, 0.35, n).astype(float),
        "is_foreign":       rng.binomial(1, 0.08, n).astype(float),
        "merchant_risk":    rng.beta(1.5, 20, n),
        "card_age_days":    rng.gamma(shape=3, scale=100, size=n),
        "unique_merch_24h": rng.poisson(3.2, n).astype(float),
        "label":            rng.binomial(1, 0.0017, n),
    })


def generate_production_data(n_days=30, samples_per_day=500, drift_start_day=14, seed=99):
    rng = np.random.RandomState(seed)
    records = []
    base_date = datetime.utcnow() - timedelta(days=n_days)
    for day in range(n_days):
        ts_base = base_date + timedelta(days=day)
        df = max(0, (day - drift_start_day) / (n_days - drift_start_day))
        for _ in range(samples_per_day):
            ts = ts_base + timedelta(
                hours=int(rng.randint(0, 24)),
                minutes=int(rng.randint(0, 60))
            )
            score = float(np.clip(rng.beta(1.2, 8) + df * rng.normal(0, 0.05), 0, 1))
            has_label = day < (n_days - 14)
            records.append({
                "record_id": str(uuid.uuid4()),
                "timestamp": ts.isoformat(),
                "day": day,
                "amount": round(float(rng.lognormal(4.5 + df * 0.8, 1.2 + df * 0.3)), 2),
                "amount_zscore": float(rng.normal(df * 0.5, 1)),
                "txn_count_1h": float(rng.poisson(2.1)),
                "txn_count_24h": float(rng.poisson(8.5 + df * 3)),
                "hour_of_day": float(ts.hour),
                "is_weekend": float(ts.weekday() >= 5),
                "is_online": float(rng.binomial(1, min(0.35 + df * 0.3, 0.95))),
                "is_foreign": float(rng.binomial(1, min(0.08 + df * 0.12, 0.5))),
                "merchant_risk": float(rng.beta(1.5, 20)),
                "card_age_days": float(rng.gamma(3, 100)),
                "unique_merch_24h": float(rng.poisson(3.2 + df * 2)),
                "score": round(score, 4),
                "label": int(rng.binomial(1, min(0.0017 * (1 + df * 0.5), 0.01))) if has_label else None,
                "model_version": "v1.2.0",
                "latency_ms": round(float(rng.gamma(2, 4)), 1),
                "drift_factor": round(df, 3),
            })
    return records


def seed_all():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Seeding data to: {DATA_DIR.resolve()}")

    print("Generating reference data (10,000 samples)...")
    ref = generate_reference_data(10000)
    ref.to_parquet(DATA_DIR / "reference.parquet", index=False)
    print(f"  Done: {len(ref)} rows")

    print("Generating 30-day production data...")
    prod = pd.DataFrame(generate_production_data())
    prod.to_parquet(DATA_DIR / "production.parquet", index=False)
    print(f"  Done: {len(prod)} rows")

    print("\nDone!")


if __name__ == "__main__":
    seed_all()
