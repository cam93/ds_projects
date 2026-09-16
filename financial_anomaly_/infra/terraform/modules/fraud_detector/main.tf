terraform {
  required_providers {
    docker = { source = "kreuzwerker/docker", version = "~> 3.0" }
  }
}
variable "environment" { type = string }
variable "project_root" { type = string }
variable "model_sha256" {
  type = string
  validation {
    condition     = can(regex("^[0-9a-f]{64}$", var.model_sha256))
    error_message = "Supply the SHA256 of an approved model."
  }
}
resource "docker_network" "backend" {
  name     = "fraud-${var.environment}"
  internal = true
}
resource "docker_volume" "audit" { name = "fraud-audit-${var.environment}" }
resource "docker_volume" "replay" { name = "fraud-replay-${var.environment}" }
resource "docker_volume" "prometheus" { name = "fraud-prometheus-${var.environment}" }
resource "docker_volume" "grafana" { name = "fraud-grafana-${var.environment}" }
locals {
  services = {
    api = {
      dockerfile = "api.Dockerfile"
      env        = ["APP_ENV=production", "MODEL_PATH=/models/fraud_model.pt", "MODEL_SHA256=${var.model_sha256}", "AUDIT_SERVICE_URL=http://audit-store:8001", "API_KEY_FILE=/run/secrets/api_key", "AUDIT_WRITE_KEY_FILE=/run/secrets/audit_write_key", "METRICS_KEY_FILE=/run/secrets/metrics_key"]
      secrets    = ["api_key", "audit_write_key", "metrics_key"]
    }
    audit-store = {
      dockerfile = "audit-store.Dockerfile"
      env        = ["APP_ENV=development", "SQLITE_PATH=/var/lib/fraud-detector/audit.db", "AUDIT_WRITE_KEY_FILE=/run/secrets/audit_write_key", "AUDIT_READ_KEY_FILE=/run/secrets/audit_read_key"]
      secrets    = ["audit_write_key", "audit_read_key"]
    }
    traffic-generator = {
      dockerfile = "traffic-generator.Dockerfile"
      env        = ["TRAFFIC_TARGET_URL=http://api:8000", "TRAFFIC_DATA_PATH=/data/replay.jsonl", "TRAFFIC_CHECKPOINT_PATH=/state/checkpoint.json", "API_KEY_FILE=/run/secrets/api_key"]
      secrets    = ["api_key"]
    }
  }
}
resource "docker_image" "service" {
  for_each = local.services
  name     = "fraud-${var.environment}-${each.key}"
  build {
    context    = var.project_root
    dockerfile = "deploy/docker/${each.value.dockerfile}"
  }
  triggers = { source = sha256(join("", [for file in fileset(var.project_root, "{src,apps,deploy}/**") : filesha256("${var.project_root}/${file}")])) }
}
resource "docker_container" "service" {
  for_each      = local.services
  name          = "fraud-${var.environment}-${each.key}"
  image         = docker_image.service[each.key].image_id
  env           = each.value.env
  user          = "10001:10001"
  read_only     = true
  security_opts = ["no-new-privileges:true"]
  capabilities { drop = ["ALL"] }
  memory     = 1024
  cpu_period = 100000
  cpu_quota  = 200000
  restart    = each.key == "traffic-generator" ? "no" : "unless-stopped"
  tmpfs      = { "/tmp" = "rw,noexec,nosuid,size=64m" }
  log_opts   = { "max-size" = "10m", "max-file" = "3" }
  networks_advanced {
    name    = docker_network.backend.name
    aliases = [each.key]
  }
  dynamic "mounts" {
    for_each = toset(each.value.secrets)
    content {
      type      = "bind"
      source    = "${var.project_root}/secrets/${mounts.value}"
      target    = "/run/secrets/${mounts.value}"
      read_only = true
    }
  }
  dynamic "mounts" {
    for_each = each.key == "api" ? [{ source = "${var.project_root}/models/artifacts", target = "/models" }] : each.key == "traffic-generator" ? [{ source = "${var.project_root}/data/processed/handbook", target = "/data" }] : []
    content {
      type      = "bind"
      source    = mounts.value.source
      target    = mounts.value.target
      read_only = true
    }
  }
  dynamic "mounts" {
    for_each = each.key == "audit-store" ? [{ source = docker_volume.audit.name, target = "/var/lib/fraud-detector" }] : each.key == "traffic-generator" ? [{ source = docker_volume.replay.name, target = "/state" }] : []
    content {
      type   = "volume"
      source = mounts.value.source
      target = mounts.value.target
    }
  }
  dynamic "ports" {
    for_each = each.key == "api" ? [8000] : []
    content {
      internal = ports.value
      external = ports.value
      ip       = "127.0.0.1"
    }
  }
  dynamic "healthcheck" {
    for_each = each.key == "traffic-generator" ? [] : [each.key == "api" ? 8000 : 8001]
    content {
      test         = ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:${healthcheck.value}/ready')"]
      interval     = "15s"
      timeout      = "5s"
      retries      = 3
      start_period = "30s"
    }
  }
}
resource "docker_image" "prometheus" { name = "prom/prometheus:v3.14.0@sha256:5ce7540c3c00ef4ab0c9d2c995c6a5b9c421f44b4a115d97a2c7af3b1c21cbb0" }
resource "docker_image" "grafana" { name = "grafana/grafana:13.2.2@sha256:ac461fb352abc50da10a51c7d02462e9c05488f11f53f14b3ad79a8145f638a0" }
resource "docker_container" "prometheus" {
  name          = "fraud-${var.environment}-prometheus"
  image         = docker_image.prometheus.image_id
  read_only     = true
  security_opts = ["no-new-privileges:true"]
  capabilities { drop = ["ALL"] }
  memory  = 1024
  restart = "unless-stopped"
  command = ["--config.file=/etc/prometheus/prometheus.yml", "--storage.tsdb.path=/prometheus", "--storage.tsdb.retention.time=15d", "--storage.tsdb.retention.size=2GB"]
  networks_advanced {
    name    = docker_network.backend.name
    aliases = ["prometheus"]
  }
  dynamic "mounts" {
    for_each = { "/etc/prometheus/prometheus.yml" = "monitoring/prometheus/prometheus.yml", "/etc/prometheus/alerts.yml" = "monitoring/prometheus/alerts.yml", "/run/secrets/metrics_key" = "secrets/metrics_key" }
    content {
      type      = "bind"
      source    = "${var.project_root}/${mounts.value}"
      target    = mounts.key
      read_only = true
    }
  }
  mounts {
    type   = "volume"
    source = docker_volume.prometheus.name
    target = "/prometheus"
  }
  ports {
    internal = 9090
    external = 9090
    ip       = "127.0.0.1"
  }
}
resource "docker_container" "grafana" {
  name          = "fraud-${var.environment}-grafana"
  image         = docker_image.grafana.image_id
  read_only     = true
  security_opts = ["no-new-privileges:true"]
  capabilities { drop = ["ALL"] }
  memory  = 512
  restart = "unless-stopped"
  env     = ["GF_AUTH_ANONYMOUS_ENABLED=false", "GF_USERS_ALLOW_SIGN_UP=false", "GF_SECURITY_ADMIN_PASSWORD__FILE=/run/secrets/grafana_password"]
  tmpfs   = { "/tmp" = "rw,noexec,nosuid,size=64m" }
  networks_advanced {
    name    = docker_network.backend.name
    aliases = ["grafana"]
  }
  dynamic "mounts" {
    for_each = { "/etc/grafana/provisioning" = "monitoring/grafana/provisioning", "/var/lib/grafana/dashboards" = "monitoring/grafana/dashboards", "/run/secrets/grafana_password" = "secrets/grafana_password" }
    content {
      type      = "bind"
      source    = "${var.project_root}/${mounts.value}"
      target    = mounts.key
      read_only = true
    }
  }
  mounts {
    type   = "volume"
    source = docker_volume.grafana.name
    target = "/var/lib/grafana"
  }
  ports {
    internal = 3000
    external = 3000
    ip       = "127.0.0.1"
  }
}
output "container_names" {
  value = concat([for c in docker_container.service : c.name], [docker_container.prometheus.name, docker_container.grafana.name])
}
