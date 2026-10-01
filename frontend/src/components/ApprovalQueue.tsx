import { useEffect, useState } from "react";
import { aegisApi, ApprovalSummary } from "../api/client";

/** Shows pending approvals and lets an operator approve/deny them. */
export function ApprovalQueue() {
  const [approvals, setApprovals] = useState<ApprovalSummary[]>([]);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    try {
      setApprovals(await aegisApi.listApprovals());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleDecision(id: string, decision: "approve" | "deny") {
    setBusyId(id);
    try {
      await aegisApi.decideApproval(id, decision);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
    }
  }

  if (error) {
    return <div role="alert">Failed to load approvals: {error}</div>;
  }

  return (
    <section aria-label="approval queue">
      <h2>Pending Approvals</h2>
      <ul>
        {approvals.map((approval) => (
          <li key={approval.id}>
            <span>
              Run {approval.run_id} ({approval.risk_level}) — {approval.status}
            </span>
            <button
              disabled={busyId === approval.id}
              onClick={() => handleDecision(approval.id, "approve")}
            >
              Approve
            </button>
            <button
              disabled={busyId === approval.id}
              onClick={() => handleDecision(approval.id, "deny")}
            >
              Deny
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
