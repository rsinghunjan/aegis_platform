"""Lightweight data quality checks, inspired by Great Expectations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Expectation:
    """A single column-level data quality expectation."""

    column: str
    check: Callable[[Any], bool]
    description: str

    def evaluate(self, value: Any) -> bool:
        try:
            return bool(self.check(value))
        except Exception:
            return False


@dataclass
class ExpectationResult:
    column: str
    description: str
    success: bool
    failed_values: List[Any] = field(default_factory=list)


@dataclass
class DataQualityReport:
    total_rows: int
    results: List[ExpectationResult]

    @property
    def success(self) -> bool:
        return all(r.success for r in self.results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "success": self.success,
            "results": [
                {
                    "column": r.column,
                    "description": r.description,
                    "success": r.success,
                    "failed_values": r.failed_values[:10],
                }
                for r in self.results
            ],
        }


def expect_not_null(column: str) -> Expectation:
    return Expectation(column, lambda v: v is not None, f"{column} should not be null")


def expect_in_set(column: str, allowed: set) -> Expectation:
    return Expectation(column, lambda v: v in allowed, f"{column} should be in {sorted(allowed)}")


def expect_between(column: str, minimum: float, maximum: float) -> Expectation:
    return Expectation(
        column,
        lambda v: v is not None and minimum <= v <= maximum,
        f"{column} should be between {minimum} and {maximum}",
    )


def run_expectations(
    rows: List[Dict[str, Any]], expectations: List[Expectation]
) -> DataQualityReport:
    results: List[ExpectationResult] = []
    for expectation in expectations:
        failed_values = []
        for row in rows:
            value = row.get(expectation.column)
            if not expectation.evaluate(value):
                failed_values.append(value)
        results.append(
            ExpectationResult(
                column=expectation.column,
                description=expectation.description,
                success=not failed_values,
                failed_values=failed_values,
            )
        )
    return DataQualityReport(total_rows=len(rows), results=results)
