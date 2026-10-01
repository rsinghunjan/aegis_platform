import { useEffect, useState } from "react";
import { aegisApi, AgentRunSummary } from "../api/client";

/** Lists agent runs with their current status, polling for live updates. */
export function RunMonitor() {
  const [runs, setRuns] = useState<AgentRunSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const data = await aegisApi.listRuns();
        if (!cancelled) {
          setRuns(data);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    }

    load();
    const interval = setInterval(load, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (error) {
    return <div role="alert">Failed to load runs: {error}</div>;
  }

  return (
    <section aria-label="agent run monitor">
      <h2>Agent Runs</h2>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Goal</th>
            <th>Status</th>
            <th>Tenant</th>
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.id}>
              <td>{run.id}</td>
              <td>{run.goal}</td>
              <td>{run.status}</td>
              <td>{run.tenant_id}</td>
              <td>{run.created_at}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
