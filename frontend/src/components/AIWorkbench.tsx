import { FormEvent, useState } from "react";
import { aegisApi, AIAnswer } from "../api/client";

export function AIWorkbench() {
  const [tenantId, setTenantId] = useState("");
  const [document, setDocument] = useState("");
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState<AIAnswer | null>(null);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  async function ingest(event: FormEvent) {
    event.preventDefault();
    setError("");
    setStatus("");
    try {
      const result = await aegisApi.ingestKnowledge(tenantId, document);
      setStatus(
        `Indexed ${result.chunks_indexed} chunk(s) as ${result.document_id} using ${result.embedding_provider}.`,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function ask(event: FormEvent) {
    event.preventDefault();
    setError("");
    setAnswer(null);
    try {
      setAnswer(await aegisApi.answerAI(tenantId, query));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <section aria-label="AI workflow workbench">
      <h2>Knowledge and AI workflow</h2>
      <label>
        Tenant ID:
        <input
          value={tenantId}
          onChange={(event) => setTenantId(event.target.value)}
          required
        />
      </label>
      <form onSubmit={ingest}>
        <label>
          Knowledge document:
          <textarea
            value={document}
            onChange={(event) => setDocument(event.target.value)}
            maxLength={64000}
            required
          />
        </label>
        <button type="submit" disabled={!tenantId || !document}>
          Index document
        </button>
      </form>
      <form onSubmit={ask}>
        <label>
          Question:
          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            maxLength={8000}
            required
          />
        </label>
        <button type="submit" disabled={!tenantId || !query}>
          Ask with retrieved context
        </button>
      </form>
      {status && <p role="status">{status}</p>}
      {error && <p role="alert">{error}</p>}
      {answer && (
        <div>
          <h3>Answer</h3>
          <p>{answer.answer}</p>
          <p>
            {answer.provider} / {answer.model}; {answer.input_tokens} input tokens,{" "}
            {answer.output_tokens} output tokens; {answer.latency_ms.toFixed(1)} ms
          </p>
          <h4>Sources</h4>
          <ul>
            {answer.citations.map((citation) => (
              <li key={`${citation.document_id}:${citation.chunk_index}`}>
                {citation.document_id}, chunk {citation.chunk_index} (score{" "}
                {citation.score.toFixed(3)})
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
