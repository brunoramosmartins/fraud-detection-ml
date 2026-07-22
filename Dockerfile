FROM python:3.10-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY pyproject.toml ./
COPY src ./src
COPY app ./app
COPY configs ./configs

# Editable install: dependencies come from pyproject.toml, while the code
# keeps running from /app so PROJECT_ROOT-relative paths (artifacts, configs)
# resolve correctly (see src/utils/config.py).
RUN pip install --no-cache-dir -e . \
 && apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Copy only the model artifacts needed for serving.
# In a production setup, replace this COPY with a volume mount or a startup
# script that pulls the artifact from object storage (e.g. S3 / GCS).
COPY artifacts/models ./artifacts/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
