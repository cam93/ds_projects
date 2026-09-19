"""Render an HA Kubernetes release; does not connect to or mutate a cluster."""

import argparse
import json
import re
from pathlib import Path


def render(
    api_image,
    ledger_image,
    model_hash,
    hostname,
    internal_issuer,
    public_issuer,
    gateway_class,
    backup_uri,
    postgres_image,
):
    for image in (api_image, ledger_image, postgres_image):
        if not re.fullmatch(r"[a-zA-Z0-9./:_-]+@sha256:[0-9a-f]{64}", image):
            raise ValueError("Every image must be pinned by registry digest")
    if not re.fullmatch(r"[0-9a-f]{64}", model_hash):
        raise ValueError("Model SHA256 is required")
    if not re.fullmatch(r"[a-z0-9.-]+", hostname) or "." not in hostname:
        raise ValueError("A DNS hostname is required")
    if not backup_uri.startswith("s3://"):
        raise ValueError("An off-cluster S3 backup destination is required")
    ns = "fraud"
    items = []

    def obj(api, kind, name, spec=None, **extra):
        item = {
            "apiVersion": api,
            "kind": kind,
            "metadata": {"name": name, "namespace": ns},
            **extra,
        }
        if spec is not None:
            item["spec"] = spec
        items.append(item)
        return item

    def env(name, value):
        return {"name": name, "value": value}

    def secret_volume(name, secret):
        return {"name": name, "secret": {"secretName": secret, "defaultMode": 0o440}}

    def mount(name, path):
        return {"name": name, "mountPath": path, "readOnly": True}

    obj(
        "v1",
        "Namespace",
        ns,
        metadata={"name": ns, "labels": {"pod-security.kubernetes.io/enforce": "restricted"}},
    )
    obj("v1", "ServiceAccount", "fraud", automountServiceAccountToken=False)
    for name, port, image, max_replicas in [
        ("api", 8000, api_image, 10),
        ("audit-store", 8001, ledger_image, 6),
    ]:
        labels = {"app": name, "app.kubernetes.io/part-of": "fraud"}
        app_env = [env("APP_ENV", "production")]
        if name == "api":
            app_env += [
                env("MODEL_PATH", "/models/fraud_model.json"),
                env("MODEL_SHA256", model_hash),
                env("AUDIT_SERVICE_URL", "https://audit-store.fraud.svc.cluster.local:8001"),
                env("AUDIT_CA_FILE", "/run/internal-ca/ca.crt"),
                env("API_KEY_FILE", "/run/credentials/api_key"),
                env("METRICS_KEY_FILE", "/run/credentials/metrics_key"),
            ]
        else:
            app_env += [
                env("DATABASE_URL_FILE", "/run/database/url"),
                env("AUDIT_READ_KEY_FILE", "/run/credentials/audit_read_key"),
            ]
        app_env += [env("AUDIT_WRITE_KEY_FILE", "/run/credentials/audit_write_key")]
        volumes = [
            secret_volume("credentials", f"fraud-{name}-credentials"),
            secret_volume("tls", f"fraud-{name}-tls"),
            {"name": "tmp", "emptyDir": {"sizeLimit": "64Mi"}},
            {"name": "internal-ca", "configMap": {"name": "fraud-internal-ca"}},
        ]
        mounts = [
            mount("credentials", "/run/credentials"),
            mount("tls", "/run/tls"),
            mount("internal-ca", "/run/internal-ca"),
            {"name": "tmp", "mountPath": "/tmp"},
        ]
        if name == "audit-store":
            volumes += [
                secret_volume("database", "fraud-database"),
                secret_volume("db-ca", "fraud-db-ca"),
            ]
            mounts += [mount("database", "/run/database"), mount("db-ca", "/run/db-ca")]
        pod = {
            "serviceAccountName": "fraud",
            "automountServiceAccountToken": False,
            "terminationGracePeriodSeconds": 40,
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": 10001,
                "runAsGroup": 10001,
                "fsGroup": 10001,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "topologySpreadConstraints": [
                {
                    "maxSkew": 1,
                    "minDomains": 3,
                    "topologyKey": "topology.kubernetes.io/zone",
                    "whenUnsatisfiable": "DoNotSchedule",
                    "labelSelector": {"matchLabels": labels},
                }
            ],
            "affinity": {
                "podAntiAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": [
                        {
                            "labelSelector": {"matchLabels": labels},
                            "topologyKey": "kubernetes.io/hostname",
                        }
                    ]
                }
            },
            "volumes": volumes,
            "containers": [
                {
                    "name": name,
                    "image": image,
                    "imagePullPolicy": "IfNotPresent",
                    "args": [
                        "uvicorn",
                        "fraud_detector.api:app" if name == "api" else "apps.audit_store.main:app",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        str(port),
                        "--workers",
                        "1",
                        "--limit-concurrency",
                        "64",
                        "--no-proxy-headers",
                        "--ssl-keyfile",
                        "/run/tls/tls.key",
                        "--ssl-certfile",
                        "/run/tls/tls.crt",
                    ],
                    "command": [],
                    "env": app_env,
                    "ports": [{"name": "https", "containerPort": port}],
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "readOnlyRootFilesystem": True,
                        "capabilities": {"drop": ["ALL"]},
                    },
                    "resources": {
                        "requests": {"cpu": "250m", "memory": "256Mi"},
                        "limits": {"cpu": "2", "memory": "1Gi"},
                    },
                    "volumeMounts": mounts,
                    "lifecycle": {
                        "preStop": {
                            "exec": {"command": ["python", "-c", "import time; time.sleep(5)"]}
                        }
                    },
                    "startupProbe": {
                        "httpGet": {"path": "/ready", "port": "https", "scheme": "HTTPS"},
                        "periodSeconds": 5,
                        "failureThreshold": 24,
                        "timeoutSeconds": 5,
                    },
                    "readinessProbe": {
                        "httpGet": {"path": "/ready", "port": "https", "scheme": "HTTPS"},
                        "periodSeconds": 5,
                        "timeoutSeconds": 5,
                    },
                    "livenessProbe": {
                        "httpGet": {"path": "/health", "port": "https", "scheme": "HTTPS"},
                        "periodSeconds": 15,
                        "timeoutSeconds": 3,
                    },
                }
            ],
        }
        # Explicit command avoids depending on image CMD/ENTRYPOINT composition.
        pod["containers"][0]["command"] = pod["containers"][0].pop("args")
        obj(
            "apps/v1",
            "Deployment",
            name,
            {
                "replicas": 3,
                "revisionHistoryLimit": 3,
                "strategy": {"rollingUpdate": {"maxUnavailable": 0, "maxSurge": 1}},
                "selector": {"matchLabels": labels},
                "template": {"metadata": {"labels": labels}, "spec": pod},
            },
        )
        obj(
            "v1",
            "Service",
            name,
            {"selector": labels, "ports": [{"name": "https", "port": port, "targetPort": "https"}]},
        )
        obj(
            "policy/v1",
            "PodDisruptionBudget",
            name,
            {"minAvailable": 2, "selector": {"matchLabels": labels}},
        )
        obj(
            "autoscaling/v2",
            "HorizontalPodAutoscaler",
            name,
            {
                "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": name},
                "minReplicas": 3,
                "maxReplicas": max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {"type": "Utilization", "averageUtilization": 60},
                        },
                    }
                ],
                "behavior": {"scaleDown": {"stabilizationWindowSeconds": 300}},
            },
        )
        obj(
            "cert-manager.io/v1",
            "Certificate",
            f"fraud-{name}",
            {
                "secretName": f"fraud-{name}-tls",
                "dnsNames": [f"{name}.fraud.svc.cluster.local"],
                "issuerRef": {"kind": "ClusterIssuer", "name": internal_issuer},
            },
        )
    obj(
        "networking.k8s.io/v1",
        "NetworkPolicy",
        "fraud-default-deny",
        {
            "podSelector": {"matchLabels": {"app.kubernetes.io/part-of": "fraud"}},
            "policyTypes": ["Ingress", "Egress"],
        },
    )

    def peer(labels):
        return {"podSelector": {"matchLabels": labels}}

    def namespace(name):
        return {"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": name}}}

    dns = {
        "to": [namespace("kube-system")],
        "ports": [{"protocol": "UDP", "port": 53}, {"protocol": "TCP", "port": 53}],
    }
    obj(
        "networking.k8s.io/v1",
        "NetworkPolicy",
        "api",
        {
            "podSelector": {"matchLabels": {"app": "api"}},
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [
                {
                    "from": [namespace("envoy-gateway-system"), namespace("monitoring")],
                    "ports": [{"port": 8000}],
                }
            ],
            "egress": [dns, {"to": [peer({"app": "audit-store"})], "ports": [{"port": 8001}]}],
        },
    )
    obj(
        "networking.k8s.io/v1",
        "NetworkPolicy",
        "ledger",
        {
            "podSelector": {"matchLabels": {"app": "audit-store"}},
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [{"from": [peer({"app": "api"})], "ports": [{"port": 8001}]}],
            "egress": [
                dns,
                {"to": [peer({"cnpg.io/cluster": "fraud-db"})], "ports": [{"port": 5432}]},
            ],
        },
    )
    obj(
        "networking.k8s.io/v1",
        "NetworkPolicy",
        "migration",
        {
            "podSelector": {"matchLabels": {"app": "migration"}},
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [],
            "egress": [
                dns,
                {"to": [peer({"cnpg.io/cluster": "fraud-db"})], "ports": [{"port": 5432}]},
            ],
        },
    )
    obj(
        "cert-manager.io/v1",
        "Certificate",
        "fraud-public",
        {
            "secretName": "fraud-public-tls",
            "dnsNames": [hostname],
            "issuerRef": {"kind": "ClusterIssuer", "name": public_issuer},
        },
    )
    obj(
        "gateway.networking.k8s.io/v1",
        "Gateway",
        "fraud",
        {
            "gatewayClassName": gateway_class,
            "listeners": [
                {
                    "name": "https",
                    "protocol": "HTTPS",
                    "port": 443,
                    "hostname": hostname,
                    "tls": {"mode": "Terminate", "certificateRefs": [{"name": "fraud-public-tls"}]},
                }
            ],
        },
    )
    obj(
        "gateway.networking.k8s.io/v1",
        "HTTPRoute",
        "fraud",
        {
            "parentRefs": [{"name": "fraud"}],
            "hostnames": [hostname],
            "rules": [
                {
                    "matches": [{"path": {"type": "Exact", "value": "/predict"}, "method": "POST"}],
                    "backendRefs": [{"name": "api", "port": 8000}],
                    "timeouts": {"request": "15s", "backendRequest": "12s"},
                }
            ],
        },
    )
    obj(
        "gateway.networking.k8s.io/v1",
        "BackendTLSPolicy",
        "fraud",
        {
            "targetRefs": [{"group": "", "kind": "Service", "name": "api"}],
            "validation": {
                "hostname": "api.fraud.svc.cluster.local",
                "caCertificateRefs": [
                    {"group": "", "kind": "ConfigMap", "name": "fraud-internal-ca"}
                ],
            },
        },
    )
    obj(
        "monitoring.coreos.com/v1",
        "ServiceMonitor",
        "fraud-api",
        {
            "selector": {"matchLabels": {"app": "api"}},
            "endpoints": [
                {
                    "port": "https",
                    "path": "/metrics",
                    "scheme": "https",
                    "interval": "15s",
                    "authorization": {
                        "type": "Bearer",
                        "credentials": {"name": "fraud-api-credentials", "key": "metrics_key"},
                    },
                    "tlsConfig": {
                        "serverName": "api.fraud.svc.cluster.local",
                        "ca": {"configMap": {"name": "fraud-internal-ca", "key": "ca.crt"}},
                    },
                }
            ],
        },
    )
    obj(
        "monitoring.coreos.com/v1",
        "PrometheusRule",
        "fraud-service",
        {
            "groups": [
                {
                    "name": "fraud-service",
                    "rules": [
                        {
                            "alert": "FraudAPIUnavailable",
                            "expr": 'sum(up{namespace="fraud",service="api"}) < 2 or absent(up{namespace="fraud",service="api"})',
                            "for": "2m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "Fewer than two fraud API replicas are scrapeable"
                            },
                        },
                        {
                            "alert": "FraudAuditFailures",
                            "expr": 'sum(increase(fraud_audit_failures_total{namespace="fraud"}[5m])) > 0',
                            "for": "1m",
                            "labels": {"severity": "critical"},
                            "annotations": {"summary": "Prediction persistence is failing"},
                        },
                        {
                            "alert": "FraudHighLatency",
                            "expr": 'histogram_quantile(0.95, sum(rate(fraud_prediction_latency_seconds_bucket{namespace="fraud"}[5m])) by (le)) > 1',
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {"summary": "Fraud API p95 exceeds one second"},
                        },
                    ],
                }
            ]
        },
    )
    # ServiceMonitor selectors match Service labels, not pod selectors.
    for item in items:
        if item["kind"] == "Service":
            item["metadata"]["labels"] = {"app": item["metadata"]["name"]}
    obj(
        "postgresql.cnpg.io/v1",
        "Cluster",
        "fraud-db",
        {
            "instances": 3,
            "imageName": postgres_image,
            "enableSuperuserAccess": False,
            "storage": {"size": "100Gi"},
            "walStorage": {"size": "20Gi"},
            "resources": {
                "requests": {"cpu": "1", "memory": "2Gi"},
                "limits": {"cpu": "4", "memory": "4Gi"},
            },
            "affinity": {
                "enablePodAntiAffinity": True,
                "podAntiAffinityType": "required",
                "topologyKey": "topology.kubernetes.io/zone",
            },
            "bootstrap": {
                "initdb": {
                    "database": "fraud",
                    "owner": "fraud_owner",
                    "secret": {"name": "fraud-db-owner"},
                }
            },
            "managed": {
                "roles": [
                    {
                        "name": "fraud_app",
                        "ensure": "present",
                        "login": True,
                        "superuser": False,
                        "passwordSecret": {"name": "fraud-db-app"},
                    }
                ]
            },
            "postgresql": {
                "parameters": {"max_connections": "200"},
                "synchronous": {"method": "any", "number": 1, "dataDurability": "required"},
                "pg_hba": ["hostnossl all all all reject"],
            },
            "plugins": [
                {
                    "name": "barman-cloud.cloudnative-pg.io",
                    "isWALArchiver": True,
                    "parameters": {"barmanObjectName": "fraud-backups"},
                }
            ],
        },
    )
    obj(
        "barmancloud.cnpg.io/v1",
        "ObjectStore",
        "fraud-backups",
        {
            "configuration": {
                "destinationPath": backup_uri,
                "s3Credentials": {
                    "accessKeyId": {"name": "fraud-backup-credentials", "key": "ACCESS_KEY_ID"},
                    "secretAccessKey": {
                        "name": "fraud-backup-credentials",
                        "key": "ACCESS_SECRET_KEY",
                    },
                },
                "wal": {"compression": "gzip"},
            },
            "retentionPolicy": "30d",
        },
    )
    obj(
        "postgresql.cnpg.io/v1",
        "ScheduledBackup",
        "fraud-daily",
        {
            "schedule": "0 0 2 * * *",
            "backupOwnerReference": "self",
            "cluster": {"name": "fraud-db"},
            "method": "plugin",
            "pluginConfiguration": {"name": "barman-cloud.cloudnative-pg.io"},
        },
    )
    obj(
        "batch/v1",
        "Job",
        "fraud-schema-migration",
        {
            "backoffLimit": 3,
            "template": {
                "metadata": {"labels": {"app": "migration", "app.kubernetes.io/part-of": "fraud"}},
                "spec": {
                    "restartPolicy": "Never",
                    "automountServiceAccountToken": False,
                    "serviceAccountName": "fraud",
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 10001,
                        "runAsGroup": 10001,
                        "fsGroup": 10001,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "volumes": [
                        secret_volume("database", "fraud-migration-database"),
                        secret_volume("db-ca", "fraud-db-ca"),
                    ],
                    "containers": [
                        {
                            "name": "migration",
                            "image": ledger_image,
                            "command": ["python", "-m", "apps.audit_store.migrate"],
                            "env": [
                                env("APP_ENV", "production"),
                                env("DATABASE_URL_FILE", "/run/database/url"),
                                env("DATABASE_RUNTIME_ROLE", "fraud_app"),
                            ],
                            "volumeMounts": [
                                mount("database", "/run/database"),
                                mount("db-ca", "/run/db-ca"),
                            ],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "128Mi"},
                                "limits": {"cpu": "1", "memory": "512Mi"},
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "readOnlyRootFilesystem": True,
                                "capabilities": {"drop": ["ALL"]},
                            },
                        }
                    ],
                },
            },
        },
    )
    obj(
        "gateway.envoyproxy.io/v1alpha1",
        "EnvoyProxy",
        "fraud-proxy",
        {
            "provider": {
                "type": "Kubernetes",
                "kubernetes": {
                    "envoyDeployment": {
                        "replicas": 3,
                        "patch": {
                            "type": "StrategicMerge",
                            "value": {
                                "spec": {
                                    "template": {
                                        "spec": {
                                            "topologySpreadConstraints": [
                                                {
                                                    "maxSkew": 1,
                                                    "minDomains": 3,
                                                    "topologyKey": "topology.kubernetes.io/zone",
                                                    "whenUnsatisfiable": "DoNotSchedule",
                                                    "labelSelector": {
                                                        "matchLabels": {
                                                            "gateway.envoyproxy.io/owning-gateway-name": "fraud"
                                                        }
                                                    },
                                                }
                                            ]
                                        }
                                    }
                                }
                            },
                        },
                    }
                },
            }
        },
    )
    obj(
        "policy/v1",
        "PodDisruptionBudget",
        "fraud-proxy",
        {
            "minAvailable": 2,
            "selector": {
                "matchLabels": {
                    "gateway.envoyproxy.io/owning-gateway-name": "fraud",
                    "gateway.envoyproxy.io/owning-gateway-namespace": "fraud",
                }
            },
        },
        metadata={"name": "fraud-proxy", "namespace": "envoy-gateway-system"},
    )
    for item in items:
        if item["kind"] == "Gateway":
            item["spec"]["infrastructure"] = {
                "parametersRef": {
                    "group": "gateway.envoyproxy.io",
                    "kind": "EnvoyProxy",
                    "name": "fraud-proxy",
                }
            }
        if item["kind"] == "Deployment":
            item["metadata"]["annotations"] = {"reloader.stakater.com/auto": "true"}
            pod = item["spec"]["template"]["spec"]
            affinity = pod["affinity"]["podAntiAffinity"].pop(
                "requiredDuringSchedulingIgnoredDuringExecution"
            )
            pod["affinity"]["podAntiAffinity"][
                "preferredDuringSchedulingIgnoredDuringExecution"
            ] = [{"weight": 100, "podAffinityTerm": term} for term in affinity]
    obj(
        "networking.k8s.io/v1",
        "NetworkPolicy",
        "postgres",
        {
            "podSelector": {"matchLabels": {"cnpg.io/cluster": "fraud-db"}},
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [
                {
                    "from": [
                        peer({"app": "audit-store"}),
                        peer({"app": "migration"}),
                        peer({"cnpg.io/cluster": "fraud-db"}),
                    ],
                    "ports": [{"port": 5432}],
                },
                {
                    "from": [namespace("cnpg-system"), peer({"cnpg.io/cluster": "fraud-db"})],
                    "ports": [{"port": 8000}],
                },
                {"from": [namespace("monitoring")], "ports": [{"port": 9187}]},
            ],
            "egress": [
                dns,
                {
                    "to": [peer({"cnpg.io/cluster": "fraud-db"})],
                    "ports": [{"port": 5432}, {"port": 8000}],
                },
                {"to": [{"ipBlock": {"cidr": "0.0.0.0/0"}}], "ports": [{"port": 443}]},
            ],
        },
    )
    return {"apiVersion": "v1", "kind": "List", "items": items}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "api-image",
        "ledger-image",
        "model-hash",
        "hostname",
        "internal-issuer",
        "public-issuer",
        "gateway-class",
        "backup-uri",
        "postgres-image",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(render(**args), indent=2) + "\n")
    print(f"Rendered deployment to {output}; review and server-side validate before applying")


if __name__ == "__main__":
    main()
