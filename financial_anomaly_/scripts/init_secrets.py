"""Create local secret files without overwriting existing credentials."""

import os
import secrets
from pathlib import Path

if __name__ == "__main__":
    directory = Path("secrets")
    directory.mkdir(mode=0o700, exist_ok=True)
    for name in ("api_key", "audit_write_key", "audit_read_key", "metrics_key", "grafana_password"):
        path = directory / name
        if not path.exists():
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            with os.fdopen(fd, "w") as output:
                output.write(secrets.token_hex(32) + "\n")
    print("Secret files ready. Values were not printed. Keep the directory private.")
