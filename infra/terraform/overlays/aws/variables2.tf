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
variable "region" {
  type    = string
  default = "us-east-1"
}

variable "project_prefix" {
  type    = string
  default = "aegis"
}

variable "env" {
  type    = string
  default = "staging"
}

variable "github_org" {
  type = string
}

variable "github_repo" {
  type = string
}

variable "branch_filter" {
  type    = string
  default = "main"
}

variable "staging_bucket" {
  type = string
}
infra/terraform/overlays/aws/variables.tf
