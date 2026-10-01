import { useState } from "react";
import { RunMonitor } from "./components/RunMonitor";
import { ApprovalQueue } from "./components/ApprovalQueue";
import { EvidenceExplorer } from "./components/EvidenceExplorer";
import { AnalyticsPanel } from "./components/AnalyticsPanel";

type Tab = "runs" | "approvals" | "evidence" | "analytics";

export default function App() {
  const [tab, setTab] = useState<Tab>("runs");
  const [selectedRunId, setSelectedRunId] = useState("");

  return (
    <main>
      <h1>Aegis AI Operations</h1>
      <p>Production operations for governed AI, ML, and LLM workflows.</p>
      <nav>
        <button onClick={() => setTab("runs")}>AI Runs</button>
        <button onClick={() => setTab("approvals")}>Governance</button>
        <button onClick={() => setTab("evidence")}>AI Evidence</button>
        <button onClick={() => setTab("analytics")}>Run Analytics</button>
      </nav>

      {tab === "runs" && <RunMonitor />}
      {tab === "approvals" && <ApprovalQueue />}
      {tab === "evidence" && (
        <div>
          <label>
            Run ID:
            <input
              value={selectedRunId}
              onChange={(e) => setSelectedRunId(e.target.value)}
            />
          </label>
          {selectedRunId && <EvidenceExplorer runId={selectedRunId} />}
        </div>
      )}
      {tab === "analytics" && <AnalyticsPanel />}
    </main>
  );
}
