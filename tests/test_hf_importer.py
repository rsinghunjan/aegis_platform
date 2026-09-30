"""
Unit tests for HF importer. Uses monkeypatch to avoid network calls.

Run: pytest tests/test_hf_importer.py -q
"""
import os
import tempfile
import json
from unittest import mock

import pytest

from api.hf_importer import import_from_hf

# Mock snapshot_download to return a prepared directory
@pytest.fixture
def fake_hf_repo(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    # create minimal repo files
    (repo_dir / "config.json").write_text(json.dumps({"dummy": True}))
    (repo_dir / "pytorch_model.bin").write_text("fake-weights")
    return str(repo_dir)


def test_import_from_hf_local(tmp_path, monkeypatch, fake_hf_repo):
    # monkeypatch snapshot_download to return our fake repo path
    monkeypatch.setattr("api.hf_importer.snapshot_download", lambda repo_id, cache_dir, resume_download: fake_hf_repo)

    # monkeypatch sign_model_artifact to avoid Vault dependency
    monkeypatch.setattr("api.hf_importer.sign_model_artifact", lambda p, k: p + ".sig.json")

    # call import_from_hf without upload or registry
    res = import_from_hf(repo_id="fake/repo", model_name="fake-model", version="v1", sign_key="key", upload_s3=False, register=False)
    assert res["repo_id"] == "fake/repo"
    assert res["artifact_path"].endswith(".tar.gz")
    assert res["signature_path"].endswith(".sig.json")
    assert res["registered"] is False


def test_import_upload_s3_raises_without_bucket(monkeypatch, fake_hf_repo):
    monkeypatch.setattr("api.hf_importer.snapshot_download", lambda repo_id, cache_dir, resume_download: fake_hf_repo)
    with pytest.raises(ValueError):
        import_from_hf(repo_id="fake/repo", model_name="fake-model", upload_s3=True, register=False)
tests/test_hf_importer.py
