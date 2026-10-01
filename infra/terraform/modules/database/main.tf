
/* Minimal DB module (returns a DATABASE_URL string)
   The overlay is expected to provide the actual managed service.
*/
variable "db_name" {
  type = string
}

output "database_url" {
  value = "postgres://postgres:postgres@localhost:5432/${var.db_name}"
}
