import json

import pytest

from scripts.deployment_evidence import EvidenceError, append_evidence, verify_records


def test_evidence_is_append_only_hash_chained_and_verifiable(tmp_path):
    evidence_path = tmp_path / "evidence.jsonl"

    first = append_evidence(
        evidence_path, "staging_deployed", {"image": "sha256:abc"}, "2026-01-01T00:00:00Z"
    )
    second = append_evidence(
        evidence_path, "production_approved", {"actor": "release-manager"},
        "2026-01-01T00:01:00Z",
    )
    records = [json.loads(line) for line in evidence_path.read_text().splitlines()]

    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert second["previous_sha256"] == first["sha256"]
    assert verify_records(records)


def test_evidence_rejects_tampering_before_append(tmp_path):
    evidence_path = tmp_path / "evidence.jsonl"
    append_evidence(evidence_path, "staging_deployed", {"image": "sha256:abc"})
    record = json.loads(evidence_path.read_text())
    record["details"]["image"] = "sha256:tampered"
    evidence_path.write_text(json.dumps(record) + "\n")

    with pytest.raises(EvidenceError, match="failed verification"):
        append_evidence(evidence_path, "production_approved", {})


def test_evidence_rejects_non_json_values(tmp_path):
    with pytest.raises(EvidenceError, match="JSON-compatible"):
        append_evidence(tmp_path / "evidence.jsonl", "bad_event", {"value": float("nan")})
