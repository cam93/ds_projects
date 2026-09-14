terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

variable "environment" {
  description = "Deployment environment name."
  type        = string
}

variable "project_root" {
  description = "Absolute path to the project root."
  type        = string
}

resource "docker_network" "fraud_detector" {
  name = "fraud-detector-${var.environment}"
}

resource "docker_volume" "audit_data" {
  name = "fraud-detector-audit-${var.environment}"
}

locals {
  services = {
    traffic_generator = {
      dockerfile = "traffic-generator.Dockerfile"
      command    = ["python", "apps/traffic_generator/main.py"]
    }
    api = {
      dockerfile = "api.Dockerfile"
      command    = ["uvicorn", "fraud_detector.api:app", "--host", "0.0.0.0", "--port", "8000"]
    }
    audit_store = {
      dockerfile = "audit-store.Dockerfile"
      command    = ["uvicorn", "apps.audit_store.main:app", "--host", "0.0.0.0", "--port", "8001"]
    }
  }
}

resource "docker_image" "service" {
  for_each = local.services
  name     = "fraud-detector-${var.environment}-${each.key}"
  build {
    context    = var.project_root
    dockerfile = "deploy/docker/${each.value.dockerfile}"
  }
}

resource "docker_container" "service" {
  for_each = local.services
  name     = "fraud-detector-${var.environment}-${each.key}"
  image    = docker_image.service[each.key].image_id
  command  = each.value.command
  networks_advanced { name = docker_network.fraud_detector.name }
  env = [
    "AUDIT_SERVICE_URL=http://fraud-detector-${var.environment}-audit_store:8001",
    "TRAFFIC_TARGET_URL=http://fraud-detector-${var.environment}-api:8000",
    "SQLITE_PATH=/var/lib/fraud-detector/audit.db",
  ]
  dynamic "mounts" {
    for_each = each.key == "audit_store" ? [1] : []
    content {
      type   = "volume"
      target = "/var/lib/fraud-detector"
      source = docker_volume.audit_data.name
    }
  }
}

resource "docker_container" "prometheus" {
  name  = "fraud-detector-${var.environment}-prometheus"
  image = "prom/prometheus:v2.55.1"
  networks_advanced { name = docker_network.fraud_detector.name }
  mounts {
    type   = "bind"
    source = "${var.project_root}/monitoring/prometheus/prometheus.yml"
    target = "/etc/prometheus/prometheus.yml"
  }
  ports {
    internal = 9090
    external = 9090
  }
}

resource "docker_container" "grafana" {
  name  = "fraud-detector-${var.environment}-grafana"
  image = "grafana/grafana:11.3.1"
  networks_advanced { name = docker_network.fraud_detector.name }
  mounts {
    type   = "bind"
    source = "${var.project_root}/monitoring/grafana"
    target = "/etc/grafana"
  }
  ports {
    internal = 3000
    external = 3000
  }
}

output "container_names" {
  value = concat(
    [for container in docker_container.service : container.name],
    [docker_container.prometheus.name, docker_container.grafana.name],
  )
}
