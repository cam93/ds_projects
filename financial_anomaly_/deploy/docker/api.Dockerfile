FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml .
COPY src ./src
COPY apps ./apps
COPY models ./models
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "fraud_detector.api:app", "--host", "0.0.0.0", "--port", "8000"]
