102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
"""
            return json.dumps(data)
        return str(data)

    def write_secret(self, key: str, value: Dict[str, Any]) -> bool:
        """
        Write (create/overwrite) a secret under KV v2 at base/key.
        `value` should be a dict of fields.
        """
        path = self._kv2_path(key)
        try:
            # Try to infer mount/secret path; use generic kv v2 write interface
            if "/data/" in path:
                secret_rel = path.split("/data/", 1)[1]
                self.client.secrets.kv.v2.create_or_update_secret(path=secret_rel, secret=value)
            else:
                # if base is like "secret" or "secret/data/aegis" handle accordingly
                if self.base.startswith("secret/data/"):
                    secret_rel = self.base.split("secret/data/", 1)[1].rstrip("/") + "/" + key.lstrip("/")
                    self.client.secrets.kv.v2.create_or_update_secret(path=secret_rel, secret=value)
                else:
                    # best-effort: write at key
                    self.client.secrets.kv.v2.create_or_update_secret(path=f"{self.base}/{key}", secret=value)
            logger.info("Vault: wrote secret (key=%s) under base", key)
            return True
        except Exception:
            logger.exception("Vault write_secret failed for %s", key)
            return False

    def generate_api_key(self, length: int = 40) -> str:
        """
        Generate a cryptographically secure API key (url-safe characters).
        """
        alphabet = string.ascii_letters + string.digits + "-_"
        return "".join(secrets.choice(alphabet) for _ in range(length))

    def rotate_client_api_key(self, client_id: str, path_prefix: str = "clients") -> Optional[str]:
        """
        Generate a new key for client_id, store it at base/clients/<client_id>, and return the new key
        (do not log the key). Caller must distribute the new key to the client securely.
        """
        new_key = self.generate_api_key()
        # store under client record
        secret_path = f"{path_prefix}/{client_id}"
        payload = {"api_key": new_key, "rotated_at": int(os.time()) if hasattr(os, "time") else None}
        # avoid logging new_key
        ok = self.write_secret(secret_path, payload)
        if not ok:
            return None
        # Return the new key for the caller to transmit to the client (out of band)
        return new_key
aegis_multimodal_ai_system/secrets/vault_provider.py
