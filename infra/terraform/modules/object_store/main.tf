  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
/* Minimal S3/GCS/Blob abstraction module
   This module is intentionally tiny: the overlay decides which provider to configure (aws, google, azurerm).
   The module exposes a simple output tuple that the application expects:
     - bucket_name
     - endpoint (optional; for S3-compatible endpoints)
*/
terraform {
  required_version = ">= 1.0.0"
}

variable "name" {
  type = string
}

variable "region" {
  type    = string
  default = "us-east-1"
}

output "bucket_name" {
  value = var.name
}

output "endpoint" {
  value = ""
}
