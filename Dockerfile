# ── Stage 1: build the React frontend ───────────────────────────────────────
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend + pre-built frontend ─────────────────────────────
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

# Copy Vite build output → /app/static (served by FastAPI StaticFiles)
COPY --from=frontend-builder /app/frontend/dist /app/static

# Generate seed data at build time so first request is instant
RUN mkdir -p /app/data && python -c "
import sys
sys.path.insert(0, '/app')
from app.core.seed_data import generate_reference_data, generate_production_data
import pandas as pd
print('Generating reference data...')
ref = generate_reference_data(10000)
ref.to_parquet('/app/data/reference.parquet', index=False)
print('Generating production data...')
prod = pd.DataFrame(generate_production_data())
prod.to_parquet('/app/data/production.parquet', index=False)
print(f'Done: {len(ref)} reference + {len(prod)} production rows.')
"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
