FROM python:3.11-slim
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Generate seed data into /app/data
RUN mkdir -p /app/data && python -c "
import sys
sys.path.insert(0, '/app')
from app.core.seed_data import generate_reference_data, generate_production_data
import pandas as pd
from pathlib import Path
print('Generating reference data...')
ref = generate_reference_data(10000)
ref.to_parquet('/app/data/reference.parquet', index=False)
print('Generating production data...')
import pandas as pd
prod = pd.DataFrame(generate_production_data())
prod.to_parquet('/app/data/production.parquet', index=False)
print(f'Done! {len(ref)} reference + {len(prod)} production rows.')
"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
