# Project-root entry point for the local development stack.
# Reuse the original state file; never create a second owner for the same resources.
terraform {
  required_version = ">= 1.8.0"
  backend "local" {
    path = "infra/terraform/environments/dev/terraform.tfstate"
  }
}
variable "model_sha256" {
  type        = string
  default     = null
  description = "Optional explicit pin; defaults to the local artifact SHA256 for development."
}
variable "enable_simulator" {
  type        = bool
  default     = true
  description = "Continuously generate synthetic demo traffic; explicit replay takes precedence."
}
variable "enable_replay" {
  type        = bool
  default     = false
  description = "Opt into finite dataset replay; the application stack starts without it."
}
module "fraud_detector" {
  source           = "./infra/terraform/modules/fraud_detector"
  environment      = "dev"
  project_root     = abspath(path.root)
  model_sha256     = var.model_sha256 != null ? var.model_sha256 : filesha256("${path.root}/models/artifacts/fraud_model.json")
  enable_replay    = var.enable_replay
  enable_simulator = var.enable_simulator
}
output "api_url" { value = "http://localhost:8000" }
output "grafana_url" { value = "http://localhost:3000" }
output "prometheus_url" { value = "http://localhost:9090" }
