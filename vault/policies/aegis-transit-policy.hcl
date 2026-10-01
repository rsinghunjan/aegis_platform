
# Minimal Vault policy that allows only signing via transit for the Aegis key.
# Keep this file under vault/policies and load via `vault policy write aegis-transit-policy vault/policies/aegis-transit-policy.hcl`

path "transit/sign/aegis-cosign" {
  capabilities = ["update"]
}

# Allow reading metadata about the key if needed (optional)
path "transit/keys/aegis-cosign" {
  capabilities = ["read","list"]
}
