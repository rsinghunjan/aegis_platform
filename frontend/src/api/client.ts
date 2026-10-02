/**
 * Minimal typed client for the Aegis operator read-only API
 * (`/operator/agent/*`) exposed by `production.py`.
 */

export interface AgentRunSummary {
  run_id: string;
  status: string;
  risk: string;
  goal_hash: string;
  result_hash: string | null;
  created_at: string;
  updated_at: string;
}

export interface ApprovalSummary {
  approval_id: string;
  run_id: string;
  tenant_id: string;
  tool_name: string;
  status: string;
  created_at: string;
  expires_at: string | null;
}

export interface EvidenceEntry {
  evidence_id: string;
  run_id: string;
  tenant_id: string;
  kind: string;
  sha256: string;
  previous_sha256: string | null;
  created_at: string;
}

export interface Page<T> {
  items: T[];
  offset: number;
  limit: number;
}

export interface AIAnswer {
  answer: string;
  model: string;
  provider: string;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  citations: Array<{
    document_id: string;
    chunk_index: number;
    score: number;
  }>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`request to ${path} failed (${response.status}): ${body}`);
  }
  return (await response.json()) as T;
}

export const aegisApi = {
  ingestKnowledge: (tenantId: string, document: string) =>
    request<{ document_id: string; chunks_indexed: number; embedding_provider: string }>(
      "/ai/knowledge",
      { method: "POST", body: JSON.stringify({ tenant_id: tenantId, document }) },
    ),
  answerAI: (tenantId: string, query: string) =>
    request<AIAnswer>("/ai/answer", {
      method: "POST",
      body: JSON.stringify({ tenant_id: tenantId, query }),
    }),
  listRuns: (tenantId: string) =>
    request<Page<AgentRunSummary>>(
      `/operator/agent/runs?tenant_id=${encodeURIComponent(tenantId)}`,
    ),
  listApprovals: (tenantId: string) =>
    request<ApprovalSummary[]>(
      `/agent/approvals?tenant_id=${encodeURIComponent(tenantId)}&status=pending`,
    ),
  decideApproval: (
    tenantId: string,
    approvalId: string,
    decision: "approve" | "deny",
  ) =>
    request<{ approval: ApprovalSummary; status: string; run_id: string }>(
      `/agent/approvals/${encodeURIComponent(approvalId)}/decision`,
      {
        method: "POST",
        body: JSON.stringify({ tenant_id: tenantId, decision }),
      },
    ),
  listEvidence: (tenantId: string, runId: string) =>
    request<EvidenceEntry[]>(
      `/agent/runs/${encodeURIComponent(runId)}/evidence?tenant_id=${encodeURIComponent(tenantId)}`,
    ),
};
