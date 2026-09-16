import argparse
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

from fraud_detector.features.adaptive import AdaptiveState
from fraud_detector.features.handbook import FEATURE_NAMES, FeatureState, calculate_features
from fraud_detector.schemas import AdaptiveFeatures, BehavioralFeatures, Transaction

root = Path.cwd()
parser = argparse.ArgumentParser(
    description="Verify an isolated demo stack and remove it afterward"
)
parser.add_argument("--model", type=Path, help="Optional portable JSON candidate to verify")
args = parser.parse_args()
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
    model_name = "fraud_model.pt"
    if args.model:
        if args.model.suffix != ".json":
            parser.error("--model requires a portable .json artifact")
        model_name = "candidate.json"
        (d / "models" / model_name).write_bytes(args.model.read_bytes())
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
    if args.model:
        parsed = Transaction.model_validate(event)
        features = calculate_features(
            customer_id=parsed.customer_id,
            terminal_id=parsed.terminal_id,
            amount=parsed.amount,
            timestamp=parsed.timestamp,
            state=FeatureState(),
        )
        delay = json.loads(args.model.read_text()).get("feedback_delay_days", 7)
        adaptive = AdaptiveState(delay).observe(
            customer_id=parsed.customer_id,
            terminal_id=parsed.terminal_id,
            amount=parsed.amount,
            timestamp=parsed.timestamp,
            outcome=0,
        )
        event["adaptive_features"] = AdaptiveFeatures.from_features(adaptive, delay).model_dump()
        event["behavioral_features"] = BehavioralFeatures.from_features(features).model_dump()
    (d / "data/replay.jsonl").write_text(json.dumps(event) + "\n")
    import hashlib

    env = {
        **os.environ,
        "MODEL_SHA256": hashlib.sha256((d / "models" / model_name).read_bytes()).hexdigest(),
    }
    override = f"""services:
  api:
    image: fraud-detector-review:local
    environment:
      APP_ENV: development
      MODEL_PATH: /models/{model_name}
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
  simulator:
    image: fraud-detector-review:local
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
        run(
            "--profile",
            "simulation",
            "run",
            "--rm",
            "--no-deps",
            "simulator",
            "python",
            "-m",
            "fraud_detector.simulator",
            "stream",
            "--seed",
            "17",
            "--url",
            "http://api:8000",
            "--state",
            "/state/live.db",
            "--api-key-file",
            "/run/secrets/api_key",
            "--count",
            "12",
            "--rate",
            "10",
        )
        run(
            "--profile",
            "simulation",
            "run",
            "--rm",
            "--no-deps",
            "simulator",
            "python",
            "-m",
            "fraud_detector.simulator",
            "stream",
            "--seed",
            "17",
            "--url",
            "http://api:8000",
            "--state",
            "/state/live.db",
            "--api-key-file",
            "/run/secrets/api_key",
            "--count",
            "3",
            "--rate",
            "10",
        )
        run(
            "--profile",
            "simulation",
            "run",
            "--rm",
            "--no-deps",
            "simulator",
            "python",
            "-c",
            "import sqlite3; c=sqlite3.connect('/state/live.db'); "
            "assert c.execute('SELECT count(*) FROM events WHERE delivered_at IS NOT NULL').fetchone()[0]==15; "
            "assert c.execute('SELECT count(*) FROM events WHERE delivered_at IS NULL').fetchone()[0]==0; "
            "print('Live simulator delivered and journaled 15 distinct transactions across container restarts')",
        )
        # Start the long-lived producer so Prometheus can scrape its authenticated quality metrics.
        run("--profile", "simulation", "up", "-d", "--no-build", "--no-deps", "simulator")
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
 assert client.get('http://simulator:8002/metrics').status_code==401
 quality=client.get('http://simulator:8002/metrics',headers={'Authorization':'Bearer '+Path('/run/secrets/metrics_key').read_text().strip()})
 assert quality.status_code==200 and 'fraud_demo_quality' in quality.text,quality.text
 assert 'customer_id' not in quality.text and 'transaction_id' not in quality.text
 assert client.get('http://grafana:3000/api/health').json()['database']=='ok'
 home=client.get('http://grafana:3000/api/dashboards/home')
 assert home.status_code==200,home.text
 home_data=home.json()
 if 'redirectUri' in home_data:
  home_uid=home_data['redirectUri'].split('/')[2]
  home_data=client.get('http://grafana:3000/api/dashboards/uid/'+home_uid).json()
 assert home_data.get('dashboard',{}).get('title')=='Realtime Fraud Detector',home_data
 dashboard=client.get('http://grafana:3000/api/dashboards/uid/fraud-detector')
 assert dashboard.status_code==200,dashboard.text
 assert dashboard.json()['meta']['canSave'] is False
 assert client.post('http://grafana:3000/api/dashboards/db',json={'dashboard':{'title':'Unauthorized demo edit','panels':[]}}).status_code in (401,403)
 assert client.post('http://grafana:3000/api/datasources',json={'name':'Unauthorized source','type':'prometheus','url':'http://prometheus:9090','access':'proxy'}).status_code in (401,403)
 assert client.get('http://grafana:3000/api/admin/settings').status_code in (401,403)
 r=client.get('http://prometheus:9090/api/v1/query',params={'query':'fraud_predictions_total'})
 assert r.json()['data']['result'],r.text
 q=client.get('http://prometheus:9090/api/v1/query',params={'query':'sum(fraud_demo_quality)'})
 assert q.json()['data']['result'],q.text
 print('Authenticated quality and API metrics scraping, anonymous read-only Grafana, replay and live simulation verified')
"""
        print(run("exec", "-T", "api", "python", "-c", code).stdout)
        print("Isolated full Compose verification passed")
    except Exception:
        r = run("logs", "--tail", "20", check=False)
        print(r.stdout, r.stderr)
        raise
    finally:
        run("down", "--volumes", "--remove-orphans", check=False)
