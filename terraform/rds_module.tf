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
 42
 43
 44
 45
 46
 47
# Terraform example for a managed AWS RDS Postgres with PITR (snapshot retention and automated backups).
# Adapt provider, VPC/Subnet IDs and DB parameter groups for your environment.
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" { type = string }
variable "db_identifier" { type = string }
variable "db_subnet_group_name" { type = string }
variable "vpc_security_group_ids" { type = list(string) }

resource "aws_db_instance" "aegis" {
  identifier             = var.db_identifier
  engine                 = "postgres"
  instance_class         = "db.r6g.large" # example, pick appropriate class
  allocated_storage      = 100
  storage_type           = "gp2"
  username               = var.db_username
  password               = var.db_password
  db_subnet_group_name   = var.db_subnet_group_name
  vpc_security_group_ids = var.vpc_security_group_ids
  backup_retention_period = 7         # automated snapshots retention (days)
  apply_immediately       = true
  deletion_protection     = true
  multi_az               = true       # multi-AZ for HA; consider Read Replicas for scaling
  skip_final_snapshot     = false
  performance_insights_enabled = true
  tags = {
    Name = "aegis-db"
  }
}

# To enable PITR, ensure automated backups are enabled (backup_retention_period > 0)
output "rds_endpoint" {
  value = aws_db_instance.aegis.address
}

# IMPORTANT: store credentials securely (use Terraform variables or remote state with encryption)
