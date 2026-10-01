from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from aegis_policy.contracts import EngineDecisionRecord, PolicyDecision, obligations_equal


@dataclass(frozen=True)
class FinalDecision:
    allow: bool
    reason: str
    disagree: bool
    engine_records: Sequence[EngineDecisionRecord]


def _materially_equal(a: PolicyDecision, b: PolicyDecision) -> bool:
    """
    Material equality for deny-on-disagree:
    - allow must match
    - obligations must match (strict)
    Labels and reason can differ without forcing deny, but you may choose otherwise.
    """
    if a.allow != b.allow:
        return False
    return obligations_equal(a.obligations, b.obligations)


def deny_on_disagree(engine_records: Sequence[EngineDecisionRecord]) -> FinalDecision:
    """
    Implements deny-on-disagree across policy engines.
    - Any deny -> deny.
    - If decisions disagree materially -> deny.
    """
    if not engine_records:
        return FinalDecision(allow=False, reason="no_policy_engines", disagree=True, engine_records=engine_records)

    # Any deny => deny.
    for r in engine_records:
        if not r.decision.allow:
            return FinalDecision(
                allow=False,
                reason=f"denied_by_{r.engine}",
                disagree=False,
                engine_records=engine_records,
            )

    # Compare all against the first.
    base = engine_records[0].decision
    for r in engine_records[1:]:
        if not _materially_equal(base, r.decision):
            return FinalDecision(
                allow=False,
                reason="policy_disagree",
                disagree=True,
                engine_records=engine_records,
            )

    return FinalDecision(allow=True, reason="allowed", disagree=False, engine_records=engine_records)
