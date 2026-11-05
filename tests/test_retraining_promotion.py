import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from api.model_signing import SigningError, sign_model_artifact, verify_model_artifact
from services.feedback_loop import export_retraining_manifest, verify_retraining_manifest


@pytest.fixture
def keys(monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setenv(
        "AEGIS_MODEL_SIGN_PRIVATE_KEY",
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode(),
    )
    monkeypatch.setenv(
        "AEGIS_MODEL_SIGN_PUBLIC_KEY",
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode(),
    )


def manifest():
    signals = [{"run_id_sha256": "a" * 64, "run_sha256": "b" * 64, "ratings": [5, 3], "verification_failures": 0}]
    return export_retraining_manifest(signals)


def test_sign_and_verify_roundtrip_detects_tampering(tmp_path, keys):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"weights")
    sig = sign_model_artifact(str(artifact), "k")
    assert verify_model_artifact(str(artifact), sig)
    artifact.write_bytes(b"tampered")
    assert not verify_model_artifact(str(artifact), sig)


def test_signing_fails_closed_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("AEGIS_MODEL_SIGN_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("AEGIS_MODEL_SIGN_PUBLIC_KEY", raising=False)
    artifact = tmp_path / "m.bin"
    artifact.write_bytes(b"x")
    with pytest.raises(SigningError):
        sign_model_artifact(str(artifact))
    assert not verify_model_artifact(str(artifact))


def test_manifest_requires_matching_approved_hash():
    m = manifest()
    assert verify_retraining_manifest(m, m["manifest_sha256"].upper()) == m["manifest_sha256"]
    with pytest.raises(ValueError):
        verify_retraining_manifest(m, "0" * 64)
    forged = json.loads(json.dumps(m))
    forged["rows"][0]["mean_rating"] = 5.0
    with pytest.raises(ValueError):
        verify_retraining_manifest(forged, m["manifest_sha256"])
    with pytest.raises(ValueError):
        verify_retraining_manifest({**m, "schema": "other"}, m["manifest_sha256"])


def test_registry_refuses_unsigned_registration(tmp_path, monkeypatch):
    from unittest import mock

    from api import mlflow_registry

    monkeypatch.delenv("AEGIS_MODEL_SIGN_PRIVATE_KEY", raising=False)
    monkeypatch.setenv("AEGIS_MODEL_REGISTRY_DIR", str(tmp_path / "reg"))
    art = tmp_path / "model.bin"
    art.write_bytes(b"x")
    with mock.patch.object(mlflow_registry, "_download_mlflow_artifact", return_value=str(art)):
        with pytest.raises(SigningError):
            mlflow_registry.promote_run_to_registry("r", "model.bin", "m", "1", sign_key="k")
    assert not (tmp_path / "reg").exists()


def test_signed_registration_and_cli_manifest_gate(tmp_path, monkeypatch, keys):
    import subprocess
    import sys
    from unittest import mock

    from api import mlflow_registry

    monkeypatch.setenv("AEGIS_MODEL_REGISTRY_DIR", str(tmp_path / "reg"))
    art = tmp_path / "model.bin"
    art.write_bytes(b"weights")
    with mock.patch.object(mlflow_registry, "_download_mlflow_artifact", return_value=str(art)):
        res = mlflow_registry.promote_run_to_registry("r", "model.bin", "m", "1", sign_key="k")
    meta = json.loads((tmp_path / "reg" / "m" / "1" / "model_signature.json").read_text())
    assert res["registered"] and meta["sha256"] and meta["signatures"] == ["model.bin.sig"]
    assert verify_model_artifact(res["artifact_local_path"], res["signed_paths"][0])
    with mock.patch.object(mlflow_registry, "_download_mlflow_artifact", return_value=str(art)):
        with pytest.raises(FileExistsError):
            mlflow_registry.promote_run_to_registry("r", "model.bin", "m", "1", sign_key="k")
    with pytest.raises(ValueError):
        mlflow_registry._register("../x", "1", str(art), [], {})

    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps(manifest()))
    proc = subprocess.run(
        [sys.executable, "scripts/promote_mlflow_run.py", "--run-id", "r", "--artifact-path", "model.bin",
         "--model-name", "m2", "--manifest", str(bad), "--approved-manifest-sha256", "0" * 64],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0 and "approved value" in proc.stderr
    assert not (tmp_path / "reg" / "m2").exists()
