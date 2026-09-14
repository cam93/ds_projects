variable "environment" {
  description = "Deployment environment name."
  type        = string
}

output "environment" {
  value = var.environment
}
