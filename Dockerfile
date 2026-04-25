FROM python:3.11-slim
WORKDIR /app

# Copy and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code ONLY (no data/ folder — generated at runtime)
COPY backend/ .

# Create data dir and generate seed data during build
RUN mkdir -p /app/data && python -m app.core.seed_data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
