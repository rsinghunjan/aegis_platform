
# Least-privilege Vault policy for Aegis Transit sign/verify (use in AppRole/Role)
path "transit/sign/aegis-model-sign" {
  capabilities = ["update"]
}
path "transit/verify/aegis-model-sign" {
  capabilities = ["update"]
}
path "transit/keys/aegis-model-sign" {
  capabilities = ["read"]
}
