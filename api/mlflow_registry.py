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
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
"""

    Returns a metadata dict:
      {
        "run_id": run_id,
        "artifact_local_path": "/tmp/...",
        "signed_paths": ["/tmp/....sig.json"],
        "registered": True/False,
        "model_name": model_name,
        "version": version_str_or_generated
      }

    Raises Exception on unrecoverable failures (e.g., download failure).
    """
    tmpdir = tempfile.mkdtemp(prefix="aegis_mlflow_promote_")
    signed_paths = []
    registered = False
    try:
        downloaded = _download_mlflow_artifact(run_id, artifact_path, tmpdir)
        basename = os.path.basename(artifact_path)
        candidate = _find_artifact_file(downloaded, basename)
        if not candidate:
            raise FileNotFoundError(f"Could not locate artifact file in downloaded path: {downloaded}")

        # optionally sign the artifact file
        if sign_key:
            try:
                from api.model_signing import sign_model_artifact
                sig = sign_model_artifact(candidate, sign_key)
                signed_paths.append(sig)
                logger.info("Signed artifact: %s -> %s", candidate, sig)
            except Exception:
                logger.exception("Signing failed; continuing without signature (call may still register if allowed)")

        # determine version string if not provided
        ver = version or f"mlflow-{run_id}"

        # attempt registry registration
        registered = _register_with_registry(model_name, ver, candidate, registry_module)

        # annotate the MLflow run with registry data (best-effort)
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            tags = {
                "aegis.registry.registered": str(bool(registered)),
                "aegis.registry.model_name": model_name,
                "aegis.registry.model_version": ver,
            }
            if signed_paths:
                tags["aegis.artifact.signed"] = ",".join(signed_paths)
            for k, v in tags.items():
                client.set_tag(run_id, k, v)
            logger.info("Annotated MLflow run %s with registry tags", run_id)
        except Exception:
            logger.exception("Failed to annotate MLflow run with registry tags (non-fatal)")

        return {
            "run_id": run_id,
            "artifact_local_path": candidate,
            "signed_paths": signed_paths,
            "registered": registered,
            "model_name": model_name,
            "version": ver,
        }
    finally:
        # Keep the tmpdir for debugging on CI by default. If you prefer to clean up, uncomment the next line.
        # shutil.rmtree(tmpdir, ignore_errors=True)
        pass
api/mlflow_registry.py
