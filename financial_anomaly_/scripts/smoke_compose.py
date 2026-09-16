import json
import os
import secrets
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

sys.path[:0] = ["src", "scripts"]
import torch
from train import export_artifact

from fraud_detector.features.handbook import FEATURE_NAMES

root = Path.cwd()
project = "fraud-compose-check-" + uuid.uuid4().hex[:8]
with tempfile.TemporaryDirectory(prefix="fraud-compose-") as temp:
    d = Path(temp)
    d.chmod(0o755)
    (d / "models").mkdir()
    (d / "data").mkdir()
    for key in ["api_key", "audit_write_key", "audit_read_key", "metrics_key", "grafana_password"]:
        (d / key).write_text(secrets.token_hex(32))
        (d / key).chmod(0o444)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    export_artifact(
        model,
        {n: 0.0 for n in FEATURE_NAMES},
        {n: 1.0 for n in FEATURE_NAMES},
        0.5,
        {},
        d / "models/fraud_model.pt",
    )
    event = {
        "transaction_id": "1",
        "customer_id": "1",
        "terminal_id": "1",
        "amount": 10.0,
        "transactions_last_hour": 0,
        "customer_history_days": 0.0,
        "hour_of_day": 12,
        "timestamp": "2024-01-01T12:00:00Z",
    }
    (d / "data/replay.jsonl").write_text(json.dumps(event) + "\n")
    import hashlib

    env = {
        **os.environ,
        "MODEL_SHA256": hashlib.sha256((d / "models/fraud_model.pt").read_bytes()).hexdigest(),
    }
    override = f"""services:
  api:
    image: fraud-detector-review:local
    environment:
      APP_ENV: development
    volumes: ['{d}/models:/models:ro']
    ports: !reset []
    healthcheck:
      interval: 1s
  audit-store:
    image: fraud-detector-review:local
    command: [uvicorn, apps.audit_store.main:app, --host, 0.0.0.0, --port, '8001']
    healthcheck:
      interval: 1s
  traffic-generator:
    image: fraud-detector-review:local
    command: [python, -m, apps.traffic_generator.main]
    volumes: ['{d}/data:/data:ro']
  prometheus:
    ports: !reset []
  grafana:
    ports: !reset []
secrets:
""" + "".join(
        f"  {key}: {{file: {d}/{key}}}\n"
        for key in [
            "api_key",
            "audit_write_key",
            "audit_read_key",
            "metrics_key",
            "grafana_password",
        ]
    )
    (d / "override.yml").write_text(override)
    cmd = [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        str(root / "docker-compose.yml"),
        "-f",
        str(d / "override.yml"),
    ]

    def run(*args, check=True):
        r = subprocess.run([*cmd, *args], env=env, text=True, capture_output=True, check=False)
        if check and r.returncode:
            print(r.stdout, r.stderr)
            raise RuntimeError("Compose command failed")
        return r

    try:
        run(
            "up",
            "-d",
            "--no-build",
            "--wait",
            "--wait-timeout",
            "90",
            "api",
            "audit-store",
            "prometheus",
            "grafana",
        )
        run("--profile", "replay", "run", "--rm", "--no-deps", "traffic-generator")
        code = """
import httpx,time
from pathlib import Path
with httpx.Client(timeout=5) as client:
 for attempt in range(30):
  try:
   prom=client.get('http://prometheus:9090/api/v1/targets').json()
   if prom['data']['activeTargets'] and all(t['health']=='up' for t in prom['data']['activeTargets']):break
  except Exception:pass
  time.sleep(1)
 else:raise RuntimeError('Prometheus target did not become healthy')
 assert client.get('http://grafana:3000/api/health').json()['database']=='ok'
 assert client.get('http://grafana:3000/api/search').status_code==401
 r=client.get('http://prometheus:9090/api/v1/query',params={'query':'fraud_predictions_total'})
 assert r.json()['data']['result'],r.text
 print('Authenticated metrics scraping, private Grafana, and dataset replay verified')
"""
        print(run("exec", "-T", "api", "python", "-c", code).stdout)
        print("Isolated full Compose verification passed")
    except Exception:
        r = run("logs", "--tail", "20", check=False)
        print(r.stdout, r.stderr)
        raise
    finally:
        run("down", "--volumes", "--remove-orphans", check=False)
