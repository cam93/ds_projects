FROM python:3.10-slim-trixie@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY constraints.txt pyproject.toml ./
RUN pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.0' && pip install --no-cache-dir -c constraints.txt setuptools wheel pip==26.2.1
COPY src ./src
COPY apps ./apps
RUN pip install --no-cache-dir --no-build-isolation -c constraints.txt . && \
    groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app && \
    mkdir -p /var/lib/fraud-detector /state && chown -R app:app /var/lib/fraud-detector /state
RUN pip uninstall -y pip setuptools wheel
USER 10001:10001
CMD ["uvicorn", "apps.audit_store.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1", "--limit-concurrency", "64", "--no-proxy-headers"]
