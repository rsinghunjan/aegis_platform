
```markdown
  - verify the artifact using the same Vault key or an external public key for cosign.
- promote_mlflow_run already calls sign_model_artifact when sign_key is provided.

6) Secrets & permissions
- Use GitHub OIDC -> Vault roles to avoid long-lived VAULT_TOKEN secrets.
- Limit the Vault Transit key role to only sign artifacts (separate keys/roles per environment).
- Audit Vault key usage and rotate keys per policy.

7) Audit & traceability
- The promotion helper annotates MLflow runs with aegis.registry.* tags (registered, model_name, model_version, signed paths).
- Persist verification results in the registry if your registry supports it (extend api.registry.add_validation_result).

If you'd like, I can:
- A) open a PR adding the signed promotion workflow and verify script into your repo and update the training workflow to call it,
- B) add a small runtime verification wrapper that your model loader calls to refuse unsigned artifacts,
- C) add cosign keyless sign example in the training CI and add a cosign verify-blob step to the promote workflow.

Which option would you like me to implement next?
