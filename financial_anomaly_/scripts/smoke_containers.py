"""Exercise isolated containers with a test-only model; never touches the deployed stack."""

import os
import secrets
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import torch
from train import export_artifact

from fraud_detector.features.handbook import FEATURE_NAMES


def docker(*args, check=True):
    result = subprocess.run(["docker", *args], text=True, capture_output=True, check=check)
    return result.stdout.strip()


def main():
    suffix = uuid4().hex[:10]
    network, ledger, api = [f"fraud-smoke-{suffix}-{x}" for x in ("net", "ledger", "api")]
    image = os.getenv("SMOKE_IMAGE", "fraud-detector-review:local")
    with tempfile.TemporaryDirectory(prefix="fraud-smoke-") as directory:
        folder = Path(directory)
        folder.chmod(0o755)
        for name in ("api_key", "audit_write_key", "audit_read_key", "metrics_key"):
            (folder / name).write_text(secrets.token_hex(32))
            (folder / name).chmod(0o444)
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
        export_artifact(
            model,
            {n: 0.0 for n in FEATURE_NAMES},
            {n: 1.0 for n in FEATURE_NAMES},
            0.5,
            {},
            folder / "model.pt",
        )
        try:
            docker("network", "create", "--internal", network)
            shared = [
                "--network",
                network,
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges:true",
                "--memory",
                "512m",
                "--pids-limit",
                "128",
                "--tmpfs",
                "/tmp:rw,size=64m",
                "-v",
                f"{folder}:/fixture:ro",
            ]
            docker(
                "run",
                "-d",
                "--name",
                ledger,
                *shared,
                "--tmpfs",
                "/var/lib/fraud-detector:rw,uid=10001,gid=10001",
                "-e",
                "SQLITE_PATH=/var/lib/fraud-detector/audit.db",
                "-e",
                "APP_ENV=development",
                "-e",
                "AUDIT_WRITE_KEY_FILE=/fixture/audit_write_key",
                "-e",
                "AUDIT_READ_KEY_FILE=/fixture/audit_read_key",
                image,
                "uvicorn",
                "apps.audit_store.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8001",
            )
            docker(
                "run",
                "-d",
                "--name",
                api,
                *shared,
                "-e",
                "APP_ENV=development",
                "-e",
                "MODEL_PATH=/fixture/model.pt",
                "-e",
                f"AUDIT_SERVICE_URL=http://{ledger}:8001",
                "-e",
                "API_KEY_FILE=/fixture/api_key",
                "-e",
                "AUDIT_WRITE_KEY_FILE=/fixture/audit_write_key",
                "-e",
                "METRICS_KEY_FILE=/fixture/metrics_key",
                image,
            )
            for _ in range(30):
                ready = subprocess.run(
                    [
                        "docker",
                        "exec",
                        api,
                        "python",
                        "-c",
                        "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')",
                    ],
                    capture_output=True,
                    check=False,
                )
                if ready.returncode == 0:
                    break
                time.sleep(1)
            else:
                raise RuntimeError("Containers did not become ready")
            check = """
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import httpx
key=Path('/fixture/api_key').read_text()
event={'transaction_id':'smoke:1','source_transaction_id':'1','customer_id':'1','terminal_id':'1','amount':10.0,'transactions_last_hour':0,'customer_history_days':0.0,'hour_of_day':12,'timestamp':'2024-01-01T12:00:00Z'}
with httpx.Client(base_url='http://localhost:8000',headers={'Authorization':'Bearer '+key},timeout=10) as client:
    assert client.post('/predict',json=event,headers={'Authorization':''}).status_code==401
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses=list(pool.map(lambda _:client.post('/predict',json=event),range(16)))
    assert all(r.status_code==200 for r in responses),[(r.status_code,r.text) for r in responses]
    assert all(r.json()==responses[0].json() for r in responses)
    event['amount']=20
    assert client.post('/predict',json=event).status_code==409
    assert client.post('/predict',content=b'x'*17000).status_code==413
print('HTTP auth, concurrent idempotency, conflicting input, and body limits verified')
"""
            print(docker("exec", api, "python", "-c", check))
            print(
                docker(
                    "exec",
                    ledger,
                    "python",
                    "-c",
                    "import sqlite3; c=sqlite3.connect('/var/lib/fraud-detector/audit.db'); assert c.execute('SELECT count(*) FROM predictions').fetchone()[0]==1; assert c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'; print('Exactly one durable ledger record verified')",
                )
            )
            print("Isolated Docker smoke test passed")
        except Exception:
            for container in (api, ledger):
                print(docker("logs", "--tail", "30", container, check=False))
            raise
        finally:
            for container in (api, ledger):
                docker("rm", "-f", container, check=False)
            docker("network", "rm", network, check=False)


if __name__ == "__main__":
    main()
