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
/* AWS overlay example (skeleton).
   Provide your actual provider config and resources here.
   This file shows how to call the canonical modules and map outputs.
*/
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.region
  # credential configuration should rely on CI OIDC or environment variables; do not hardcode
}

variable "region" {
  type    = string
  default = "us-east-1"
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
