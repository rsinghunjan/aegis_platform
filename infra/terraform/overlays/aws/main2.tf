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
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
/*
AWS overlay for Aegis staging.

This is a practical but minimal example that creates:
 - S3 bucket for model artifacts
 - RDS Postgres (single AZ example - for production consider multi-AZ)
 - Outputs: bucket name and a placeholder DATABASE_URL (you should use actual credentials and endpoint)

Notes:
- You must provide appropriate provider configuration and credentials to run this.
- Security groups, subnet groups and networking are simplified; coordinate with infra team.
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
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "db_username" {
  type    = string
  default = "aegis"
}
variable "db_password" {
  type    = string
  default = "change-me"
  sensitive = true
}

# S3 bucket
resource "aws_s3_bucket" "aegis_models" {
  bucket = var.bucket_name
  acl    = "private"
  force_destroy = true
  tags = {
    Name = "aegis-models"
    Env  = "staging"
  }
}

# RDS Postgres (recommended: replace with managed DB module and subnet groups)
resource "aws_db_instance" "aegis_postgres" {
  allocated_storage    = 20
  engine               = "postgres"
  engine_version       = "14.9"
  instance_class       = "db.t3.medium"
  name                 = "aegis"
  username             = var.db_username
  password             = var.db_password
  skip_final_snapshot  = true
  publicly_accessible  = true
}

output "bucket_name" {
  value = aws_s3_bucket.aegis_models.id
}

output "database_url" {
  value = format("postgresql://%s:%s@%s:%d/%s", aws_db_instance.aegis_postgres.username, aws_db_instance.aegis_postgres.password, aws_db_instance.aegis_postgres.address, aws_db_instance.aegis_postgres.port, aws_db_instance.aegis_postgres.name)
}
infra/terraform/overlays/aws/main.tf
