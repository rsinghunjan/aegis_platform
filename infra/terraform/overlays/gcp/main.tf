  1
  2
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
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
/* GCP overlay skeleton */
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = var.project
  region  = var.region
}

variable "project" {
  type = string
}

variable "region" {
  type    = string
  default = "us-central1"
}

module "object_store" {
  source = "../../modules/object_store"
  name   = "aegis-models"
  region = var.region
}

module "database" {
  source  = "../../modules/database"
  db_name = "aegis_prod"
}

output "bucket_name" {
  value = module.object_store.bucket_name
}
output "database_url" {
  value = module.database.database_url
}
