
# Vault KV v2 policy for the Aegis service account
path "secret/data/aegis/keys/*" {
  capabilities = ["read", "list"]
}

path "secret/data/aegis/keys/model_sign/*" {
  capabilities = ["read", "list", "create", "update"]
}

path "secret/data/aegis/secrets/*" {
  capabilities = ["read", "list"]
}
