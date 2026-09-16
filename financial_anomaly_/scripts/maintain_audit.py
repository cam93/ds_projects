"""Consistent SQLite backup, integrity checks, and optional retention pruning."""

import argparse
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def backup_database(database: Path, backup: Path):
    if not database.is_file() or backup.exists():
        raise ValueError("Source must exist and backup destination must be new")
    backup.parent.mkdir(parents=True, exist_ok=True)
    with (
        closing(sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)) as source,
        closing(sqlite3.connect(backup)) as target,
    ):
        source.backup(target)
        if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Backup integrity check failed")
    backup.chmod(0o600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("backup", type=Path)
    parser.add_argument(
        "--prune-before",
        help="UTC ingestion-time cutoff; removes old audit/idempotency records after backup",
    )
    args = parser.parse_args()
    backup_database(args.database, args.backup)
    if args.prune_before:
        cutoff = datetime.fromisoformat(args.prune_before.replace("Z", "+00:00"))
        if cutoff.tzinfo is None or cutoff >= datetime.now(timezone.utc):
            raise ValueError("Retention cutoff must be timezone-aware and in the past")
        with sqlite3.connect(args.database, timeout=5) as connection:
            cursor = connection.execute(
                "DELETE FROM predictions WHERE created_at < ?",
                (cutoff.astimezone(timezone.utc).isoformat(),),
            )
            print(f"Pruned {cursor.rowcount} records after successful backup")
    print("Backup integrity verified")


if __name__ == "__main__":
    main()
