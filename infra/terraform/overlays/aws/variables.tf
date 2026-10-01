
variable "region" {
  type    = string
  default = "us-east-1"
}

variable "bucket_name" {
  type    = string
  default = "aegis-staging-models-12345"
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
