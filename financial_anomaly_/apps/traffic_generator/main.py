"""Generate continuous synthetic transaction traffic for local demonstrations."""

import json
import os
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from uuid import uuid4


def build_transaction() -> dict[str, object]:
    return {
        "transaction_id": str(uuid4()),
        "amount": round(random.uniform(2.0, 2500.0), 2),
        "account_age_days": random.randint(1, 3650),
        "transactions_last_hour": random.randint(0, 20),
        "is_international": random.choice([False, False, False, True]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    target_url = os.getenv("TRAFFIC_TARGET_URL", "http://localhost:8000")
    interval = float(os.getenv("TRAFFIC_INTERVAL_SECONDS", "0.25"))
    endpoint = f"{target_url.rstrip('/')}/predict"
    while True:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(build_transaction()).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5):
                pass
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
        time.sleep(interval)


if __name__ == "__main__":
    main()
