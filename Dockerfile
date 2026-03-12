FROM python:3.10-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app
COPY configs ./configs

# Copy only the model artifacts needed for serving.
# In a production setup, replace this COPY with a volume mount or a startup
# script that pulls the artifact from object storage (e.g. S3 / GCS).
COPY artifacts/models ./artifacts/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
