terraform {
  required_version = ">= 1.8.0"
}

variable "model_sha256" {
  type    = string
  default = null
}
variable "enable_replay" {
  type    = bool
  default = false
}
locals { project_root = abspath("${path.root}/../../../..") }

module "fraud_detector" {
  model_sha256  = var.model_sha256 != null ? var.model_sha256 : filesha256("${local.project_root}/models/artifacts/fraud_model.pt")
  enable_replay = var.enable_replay
  source        = "../../modules/fraud_detector"
  environment   = "dev"
  project_root  = local.project_root
}
