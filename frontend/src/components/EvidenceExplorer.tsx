import { useEffect, useState } from "react";
import { aegisApi, EvidenceEntry } from "../api/client";

/** Visualizes the evidence hash chain for a single run, highlighting breaks. */
export function EvidenceExplorer({ runId }: { runId: string }) {
  const [entries, setEntries] = useState<EvidenceEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    aegisApi
      .listEvidence(runId)
      .then(setEntries)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [runId]);

  if (error) {
    return <div role="alert">Failed to load evidence: {error}</div>;
  }

  function isChainBroken(index: number): boolean {
    if (index === 0) return false;
    return entries[index].previous_hash !== entries[index - 1].hash;
  }

  return (
    <section aria-label="evidence chain explorer">
      <h2>Evidence Chain for Run {runId}</h2>
      <ol>
        {entries.map((entry, index) => (
          <li key={entry.id} style={{ color: isChainBroken(index) ? "red" : undefined }}>
            step {entry.step_index}: {entry.hash}
            {isChainBroken(index) && " (chain break detected)"}
          </li>
        ))}
      </ol>
    </section>
  );
}
