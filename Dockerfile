
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[api]"

COPY artifacts/cloud-run/baseline_model.joblib ./artifacts/cloud-run/baseline_model.joblib

EXPOSE 8000

CMD ["uvicorn", "houseprice_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
