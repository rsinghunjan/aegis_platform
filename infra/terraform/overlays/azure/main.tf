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
/* Azure overlay skeleton */
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features = {}
}

variable "location" {
  type    = string
  default = "eastus"
}

module "object_store" {
  source = "../../modules/object_store"
  name   = "aegis-models"
  region = var.location
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
