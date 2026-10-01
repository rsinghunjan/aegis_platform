  7
# Minimal policy for transit signing
path "transit/sign/aegis-cosign" {
  capabilities = ["update"]
}
path "transit/keys/aegis-cosign" {
  capabilities = ["read"]
}
