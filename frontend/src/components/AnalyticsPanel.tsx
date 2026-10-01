import { useEffect, useMemo, useState } from "react";
import { aegisApi, AgentRunSummary } from "../api/client";

/** Simple aggregate analytics derived from the run list (counts by status). */
export function AnalyticsPanel() {
  const [runs, setRuns] = useState<AgentRunSummary[]>([]);

  useEffect(() => {
    aegisApi.listRuns().then(setRuns).catch(() => setRuns([]));
  }, []);

  const statusCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const run of runs) {
      counts[run.status] = (counts[run.status] ?? 0) + 1;
    }
    return counts;
  }, [runs]);

  return (
    <section aria-label="run analytics">
      <h2>Run Analytics</h2>
      <ul>
        {Object.entries(statusCounts).map(([status, count]) => (
          <li key={status}>
            {status}: {count}
          </li>
        ))}
      </ul>
    </section>
  );
}
