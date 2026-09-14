terraform {
  required_version = ">= 1.8.0"
}

module "fraud_detector" {
  source       = "../../modules/fraud_detector"
  environment  = "dev"
  project_root = "../../../../"
}
