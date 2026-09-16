terraform {
  required_version = ">= 1.8.0"
}

variable "model_sha256" { type = string }

module "fraud_detector" {
  model_sha256 = var.model_sha256
  source       = "../../modules/fraud_detector"
  environment  = "dev"
  project_root = abspath("${path.root}/../../../..")
}
