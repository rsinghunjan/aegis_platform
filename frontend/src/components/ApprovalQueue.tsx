import { useEffect, useState } from "react";
import { aegisApi, ApprovalSummary } from "../api/client";

/** Shows pending approvals and lets an operator approve/deny them. */
export function ApprovalQueue({ tenantId }: { tenantId: string }) {
  const [approvals, setApprovals] = useState<ApprovalSummary[]>([]);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    try {
      if (!tenantId) {
        setApprovals([]);
        setError("Set a tenant ID to load approvals.");
        return;
      }
      setApprovals(await aegisApi.listApprovals(tenantId));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  useEffect(() => {
    refresh();
  }, [tenantId]);

  async function handleDecision(approvalId: string, decision: "approve" | "deny") {
    setBusyId(approvalId);
    try {
      await aegisApi.decideApproval(tenantId, approvalId, decision);
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
          <li key={approval.approval_id}>
            <span>
              Run {approval.run_id} — {approval.tool_name}: {approval.status}
            </span>
            <button
              disabled={busyId === approval.approval_id}
              onClick={() => handleDecision(approval.approval_id, "approve")}
            >
              Approve
            </button>
            <button
              disabled={busyId === approval.approval_id}
              onClick={() => handleDecision(approval.approval_id, "deny")}
            >
              Deny
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
