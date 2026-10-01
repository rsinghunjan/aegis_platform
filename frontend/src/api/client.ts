/**
 * Minimal typed client for the Aegis operator read-only API
 * (`/operator/agent/*`) exposed by `production.py`.
 */

export interface AgentRunSummary {
  id: string;
  goal: string;
  status: string;
  tenant_id: string;
  created_at: string;
}

export interface ApprovalSummary {
  id: string;
  run_id: string;
  status: string;
  risk_level: string;
  requested_at: string;
}

export interface EvidenceEntry {
  id: string;
  run_id: string;
  step_index: number;
  hash: string;
  previous_hash: string | null;
  created_at: string;
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
  listRuns: () => request<AgentRunSummary[]>("/operator/agent/runs"),
  getRun: (runId: string) => request<AgentRunSummary>(`/operator/agent/runs/${runId}`),
  listApprovals: () => request<ApprovalSummary[]>("/operator/agent/approvals"),
  decideApproval: (approvalId: string, decision: "approve" | "deny") =>
    request<ApprovalSummary>(`/operator/agent/approvals/${approvalId}`, {
      method: "POST",
      body: JSON.stringify({ decision }),
    }),
  listEvidence: (runId: string) =>
    request<EvidenceEntry[]>(`/operator/agent/runs/${runId}/evidence`),
};
