from services.regression_sets import failure_candidates, merge_cases

GOOD = {
    "query": "What does Aegis sign?",
    "context": "Aegis signs model artifacts before promotion.",
    "expected_facts": ["signs model artifacts"],
    "source_request_sha256": "abc",
}


def test_failure_candidates_are_hash_only_and_exclude_successes():
    rows = [
        {"request_id": "1", "request_sha256": "h", "status": "failed", "error_type": "X",
         "provider": "p", "created_at": "t", "tenant_id": "t"},
        {"request_id": "2", "request_sha256": "h2", "status": "succeeded", "provider": "p",
         "created_at": "t"},
    ]
    result = failure_candidates(rows)
    assert [r["request_id"] for r in result] == ["1"]
    assert "tenant_id" not in result[0]


def test_merge_validates_dedups_and_screens_pii():
    merged, rejected = merge_cases([], [GOOD, GOOD])
    assert len(merged) == 1 and rejected[0]["reason"] == "duplicate"

    pii = {**GOOD, "query": "mail bob@example.com", "source_request_sha256": "p"}
    missing_fact = {**GOOD, "expected_facts": ["not present"], "source_request_sha256": "m"}
    empty = {**GOOD, "expected_facts": [], "source_request_sha256": "e"}
    _, rejected = merge_cases([], [pii, missing_fact, empty])
    assert [r["id"] for r in rejected] == ["p", "m", "e"]
