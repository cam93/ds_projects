import pytest

from apps.audit_store.postgres import validate_connection_url
from scripts.render_kubernetes import render


def manifest():
    return render(
        "registry.example.com/api@sha256:" + "a" * 64,
        "registry.example.com/ledger@sha256:" + "b" * 64,
        "c" * 64,
        "fraud.example.com",
        "private-ca",
        "public-ca",
        "eg",
        "s3://example-backups/fraud",
        "ghcr.io/cloudnative-pg/postgresql@sha256:" + "d" * 64,
    )


def test_ha_manifest_enforces_replication_tls_and_private_storage():
    items = manifest()["items"]
    deployments = [x for x in items if x["kind"] == "Deployment"]
    assert len(deployments) == 2
    for deployment in deployments:
        assert deployment["spec"]["replicas"] >= 3
        pod = deployment["spec"]["template"]["spec"]
        assert pod["automountServiceAccountToken"] is False
        assert pod["topologySpreadConstraints"][0]["minDomains"] == 3
        assert all(v.get("hostPath") is None for v in pod["volumes"])
        assert "--ssl-certfile" in pod["containers"][0]["command"]
    db = next(x for x in items if x["kind"] == "Cluster")
    assert db["spec"]["instances"] == 3
    assert db["spec"]["postgresql"]["synchronous"]["dataDurability"] == "required"
    assert any(x["kind"] == "ScheduledBackup" for x in items)
    assert any(x["kind"] == "BackendTLSPolicy" for x in items)
    assert any(x["kind"] == "NetworkPolicy" for x in items)


def test_production_database_requires_verified_tls(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(ValueError, match="verify-full"):
        validate_connection_url("postgresql://host/db?sslmode=require")
    assert validate_connection_url("postgresql://host/db?sslmode=verify-full")
