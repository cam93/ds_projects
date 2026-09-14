FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml .
COPY src ./src
COPY apps ./apps
RUN pip install --no-cache-dir .

EXPOSE 8001
CMD ["uvicorn", "apps.audit_store.main:app", "--host", "0.0.0.0", "--port", "8001"]
