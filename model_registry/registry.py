 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
 98
 99
100
101
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
152
153
154
import hashlib
            self._models[self._key(name, version)] = meta
        logger.info("Registered local model %s version %s at %s", name, version, meta.path)
        return meta

    def fetch_from_s3(self, name: str, version: str, s3_key: Optional[str] = None, checksum: Optional[str] = None) -> ModelMetadata:
        """
        Download an artifact from S3 to local cache and register it.
        s3_key can be provided or we fallback to prefix/{name}/{version}/artifact.tar
        """
        if not _has_boto3:
            raise RuntimeError("boto3 not available; cannot fetch from s3")
        if not (self.s3_bucket or os.getenv("MODEL_REGISTRY_S3_BUCKET")):
            raise RuntimeError("S3 bucket not configured for model registry")

        s3 = boto3.client("s3")
        key = s3_key or f"{self.s3_prefix}/{name}/{version}/model.tar"
        local_dir = MODELS_DIR / name / version
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / Path(key).name

        try:
            logger.info("Downloading s3://%s/%s -> %s", self.s3_bucket, key, local_path)
            s3.download_file(self.s3_bucket, key, str(local_path))
        except ClientError as e:
            logger.exception("S3 download failed: %s", e)
            raise

        if checksum and not self._verify_checksum(local_path, checksum):
            # remove possibly corrupted file
            try:
                local_path.unlink()
            except Exception:
                pass
            raise ValueError("Checksum mismatch after S3 download")

        meta = ModelMetadata(name=name, version=version, path=str(local_path.resolve()), checksum=checksum, loaded_at=time.time())
        with self._lock:
            self._models[self._key(name, version)] = meta
        return meta

    def _verify_checksum(self, path: Path, expected_sha256: str) -> bool:
        """
        Compute sha256 of file at path and compare to expected (hex string).
        Works for files. If path is directory, returns False.
        """
        if not path.exists() or not path.is_file():
            logger.debug("Checksum verify: path missing or not a file: %s", path)
            return False
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        computed = h.hexdigest()
        ok = computed.lower() == expected_sha256.lower()
        logger.debug("Checksum computed=%s expected=%s ok=%s", computed[:8], (expected_sha256 or "")[:8], ok)
        return ok

    def get(self, name: str, version: str) -> Optional[ModelMetadata]:
        with self._lock:
            return self._models.get(self._key(name, version))

    def list_models(self) -> Dict[str, Dict]:
        with self._lock:
            return {k: v.as_dict() for k, v in self._models.items()}

    def health(self) -> Dict:
        """
        Return registry health summary: number of models, last load times, disk space hints.
        """
        with self._lock:
            models = {k: v.as_dict() for k, v in self._models.items()}
        stat = MODELS_DIR.stat()
        return {
            "model_count": len(models),
            "models": models,
            "models_dir": str(MODELS_DIR),
            "models_dir_mtime": stat.st_mtime,
            "checked_at": time.time(),
        }
