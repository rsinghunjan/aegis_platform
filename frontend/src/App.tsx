import { useState } from "react";
import { RunMonitor } from "./components/RunMonitor";
import { ApprovalQueue } from "./components/ApprovalQueue";
import { EvidenceExplorer } from "./components/EvidenceExplorer";
import { AnalyticsPanel } from "./components/AnalyticsPanel";
import { AIWorkbench } from "./components/AIWorkbench";

type Tab = "workflows" | "runs" | "approvals" | "evidence" | "analytics";

export default function App() {
  const [tab, setTab] = useState<Tab>("workflows");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [tenantId, setTenantId] = useState(import.meta.env.VITE_AEGIS_TENANT_ID ?? "");

  return (
    <main>
      <h1>Aegis AI Operations</h1>
      <p>Production operations for governed AI, ML, and LLM workflows.</p>
      <label>
        Tenant ID:
        <input
          value={tenantId}
          onChange={(event) => setTenantId(event.target.value)}
          required
        />
      </label>
      <nav>
        <button onClick={() => setTab("runs")}>AI Runs</button>
        <button onClick={() => setTab("workflows")}>AI Workflows</button>
        <button onClick={() => setTab("approvals")}>Governance</button>
        <button onClick={() => setTab("evidence")}>AI Evidence</button>
        <button onClick={() => setTab("analytics")}>Run Analytics</button>
      </nav>

      {tab === "workflows" && <AIWorkbench tenantId={tenantId} />}
      {tab === "runs" && <RunMonitor tenantId={tenantId} />}
      {tab === "approvals" && <ApprovalQueue tenantId={tenantId} />}
      {tab === "evidence" && (
        <div>
          <label>
            Run ID:
            <input
              value={selectedRunId}
              onChange={(e) => setSelectedRunId(e.target.value)}
            />
          </label>
          {selectedRunId && (
            <EvidenceExplorer tenantId={tenantId} runId={selectedRunId} />
          )}
        </div>
      )}
      {tab === "analytics" && <AnalyticsPanel tenantId={tenantId} />}
    </main>
  );
}
