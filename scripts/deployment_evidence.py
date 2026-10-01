"""Append and verify hash-chained deployment evidence records."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    pass


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise EvidenceError("evidence must contain JSON-compatible values") from exc


def verify_records(records: list[dict[str, Any]]) -> bool:
    previous_sha256 = None
    for record in records:
        if not isinstance(record, dict) or "sha256" not in record:
            return False
        unsigned = {key: value for key, value in record.items() if key != "sha256"}
        if unsigned.get("previous_sha256") != previous_sha256:
            return False
        digest = hashlib.sha256(_canonical_json(unsigned).encode()).hexdigest()
        if record["sha256"] != digest:
            return False
        previous_sha256 = digest
    return True


def append_evidence(
    path: Path, event: str, details: dict[str, Any], timestamp: str | None = None
) -> dict[str, Any]:
    if not event.strip():
        raise EvidenceError("event must not be empty")
    if not isinstance(details, dict):
        raise EvidenceError("details must be a JSON object")

    records: list[dict[str, Any]] = []
    if path.exists():
        try:
            records = [json.loads(line) for line in path.read_text().splitlines()]
        except (json.JSONDecodeError, OSError) as exc:
            raise EvidenceError("existing evidence file is invalid") from exc
        if not verify_records(records):
            raise EvidenceError("existing evidence chain failed verification")

    unsigned = {
        "sequence": len(records) + 1,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "event": event.strip(),
        "details": details,
        "previous_sha256": records[-1]["sha256"] if records else None,
    }
    record = {
        **unsigned,
        "sha256": hashlib.sha256(_canonical_json(unsigned).encode()).hexdigest(),
    }
    _canonical_json(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as evidence_file:
        evidence_file.write(_canonical_json(record) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--details", default="{}")
    args = parser.parse_args()
    try:
        details = json.loads(args.details)
        record = append_evidence(args.file, args.event, details)
    except (json.JSONDecodeError, EvidenceError) as exc:
        parser.error(str(exc))
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
