"""Run schema migration once with a migration credential before rolling out ledger replicas."""

import os

import psycopg
from psycopg import sql

from apps.audit_store.postgres import SCHEMA, database_url, validate_connection_url


def main():
    url = validate_connection_url(database_url())
    with psycopg.connect(url, connect_timeout=10) as connection:
        connection.execute("SELECT pg_advisory_xact_lock(882401)")
        connection.execute(SCHEMA)
        role = os.getenv("DATABASE_RUNTIME_ROLE")
        if role:
            connection.execute(
                sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(role))
            )
            connection.execute(
                sql.SQL("GRANT SELECT, INSERT ON predictions TO {}").format(sql.Identifier(role))
            )
    print("PostgreSQL schema migration completed")


if __name__ == "__main__":
    main()
