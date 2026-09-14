FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml .
COPY src ./src
COPY apps ./apps
COPY models ./models
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

CMD ["python", "apps/stream_processor/main.py"]
