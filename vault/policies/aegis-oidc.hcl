
 
# Least-privilege Vault policy for CI OIDC role that rotates model_sign keys and reads active pointer.
# Scope this policy to only the needed KV paths.

path "secret/data/aegis/keys/model_sign/*" {
  capabilities = ["create", "update", "read", "list"]
}

path "secret/data/aegis/keys/model_sign/active" {
  capabilities = ["create", "update", "read"]
}

# Optionally allow read of JWT secret (if CI needs to verify or rotate JWT secret)
path "secret/data/aegis/secrets/jwt" {
  capabilities = ["read"]
}
