# infra/rds.tf (example Terraform snippet for AWS RDS with automated backups & PITR)
# NOTE: adapt VPC, subnet_group, kms_key, encryption and provider config to your infra.
resource "aws_db_instance" "aegis" {
  allocated_storage    = 100
  engine               = "postgres"
  engine_version       = "15.3"
  instance_class       = "db.r5.large"
  name                 = "aegis"
  username             = var.db_username
  password             = var.db_password
  parameter_group_name = "default.postgres15"
  backup_retention_period = 7
  backup_window        = "03:00-04:00"
  multi_az             = true
  storage_encrypted    = true
  kms_key_id           = var.kms_key_id
  auto_minor_version_upgrade = true

  # Enable PITR / continuous backups by setting backup_retention_period > 0 in RDS
  # For Aurora use cluster resource with cluster-level backups/PITR
  tags = var.tags
}

# Example read replica
resource "aws_db_instance" "aegis_replica" {
  replicate_source_db = aws_db_instance.aegis.id
  instance_class      = "db.r5.large"
  publicly_accessible = false
  # ...
}
