import { useEffect, useState } from "react";
import { aegisApi, EvidenceEntry } from "../api/client";

/** Visualizes the evidence hash chain for a single run, highlighting breaks. */
export function EvidenceExplorer({
  tenantId,
  runId,
}: {
  tenantId: string;
  runId: string;
}) {
  const [entries, setEntries] = useState<EvidenceEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!tenantId) {
      setError("Set a tenant ID to load run evidence.");
      setEntries([]);
      return;
    }
    aegisApi
      .listEvidence(tenantId, runId)
      .then(setEntries)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [tenantId, runId]);

  if (error) {
    return <div role="alert">Failed to load evidence: {error}</div>;
  }

  function isChainBroken(index: number): boolean {
    if (index === 0) return false;
    return entries[index].previous_sha256 !== entries[index - 1].sha256;
  }

  return (
    <section aria-label="evidence chain explorer">
      <h2>Evidence Chain for Run {runId}</h2>
      <ol>
        {entries.map((entry, index) => (
          <li key={entry.evidence_id} style={{ color: isChainBroken(index) ? "red" : undefined }}>
            {entry.kind}: {entry.sha256}
            {isChainBroken(index) && " (chain break detected)"}
          </li>
        ))}
      </ol>
    </section>
  );
}
