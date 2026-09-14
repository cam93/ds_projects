FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml .
COPY src ./src
COPY apps ./apps
RUN pip install --no-cache-dir .

CMD ["python", "apps/traffic_generator/main.py"]
