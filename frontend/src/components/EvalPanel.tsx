import { useState } from "react";
import { generateEvalDataset, runRetrievalEval, runRagasEval, RetrievalMetrics, RagasMetrics } from "../lib/api";

const styles: Record<string, React.CSSProperties> = {
  panel: { padding: "1.5rem", borderTop: "1px solid #2a2a2a" },
  title: { fontSize: "0.8rem", fontWeight: 600, color: "#666", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "1rem" },
  row: { display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" },
  btn: {
    background: "#1e1e1e",
    border: "1px solid #333",
    color: "#ccc",
    borderRadius: "6px",
    padding: "0.4rem 0.85rem",
    cursor: "pointer",
    fontSize: "0.82rem",
  },
  metrics: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" },
  metric: {
    background: "#1a1a1a",
    border: "1px solid #2a2a2a",
    borderRadius: "8px",
    padding: "0.75rem 1rem",
  },
  metricLabel: { fontSize: "0.75rem", color: "#666", marginBottom: "0.25rem" },
  metricValue: { fontSize: "1.4rem", fontWeight: 700, color: "#4f8ef7" },
  error: { fontSize: "0.82rem", color: "#e06c6c", marginTop: "0.5rem" },
  status: { fontSize: "0.82rem", color: "#6dbf6d", marginTop: "0.5rem" },
};

export default function EvalPanel() {
  const [retrieval, setRetrieval] = useState<RetrievalMetrics | null>(null);
  const [ragas, setRagas] = useState<RagasMetrics | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async (label: string, fn: () => Promise<unknown>) => {
    setLoading(label);
    setError(null);
    setStatus(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed");
    } finally {
      setLoading(null);
    }
  };

  return (
    <div style={styles.panel}>
      <div style={styles.title}>Eval Harness</div>
      <div style={styles.row}>
        <button
          style={styles.btn}
          disabled={!!loading}
          onClick={() => run("generate", async () => {
            const res = await generateEvalDataset(20);
            setStatus(`Generated ${res.num_pairs} QA pairs`);
          })}
        >
          {loading === "generate" ? "Generating..." : "Generate QA Dataset"}
        </button>

        <button
          style={styles.btn}
          disabled={!!loading}
          onClick={() => run("retrieval", async () => {
            const m = await runRetrievalEval();
            setRetrieval(m);
          })}
        >
          {loading === "retrieval" ? "Running..." : "Run Retrieval Eval"}
        </button>

        <button
          style={styles.btn}
          disabled={!!loading}
          onClick={() => run("ragas", async () => {
            const m = await runRagasEval();
            setRagas(m);
          })}
        >
          {loading === "ragas" ? "Running..." : "Run RAGAS Eval"}
        </button>
      </div>

      {status && <div style={styles.status}>{status}</div>}
      {error && <div style={styles.error}>{error}</div>}

      {(retrieval || ragas) && (
        <div style={styles.metrics}>
          {retrieval && (
            <>
              <div style={styles.metric}>
                <div style={styles.metricLabel}>Recall@{retrieval.k}</div>
                <div style={styles.metricValue}>{retrieval.recall_at_k.toFixed(3)}</div>
              </div>
              <div style={styles.metric}>
                <div style={styles.metricLabel}>MRR@{retrieval.k}</div>
                <div style={styles.metricValue}>{retrieval.mrr_at_k.toFixed(3)}</div>
              </div>
            </>
          )}
          {ragas && (
            <>
              <div style={styles.metric}>
                <div style={styles.metricLabel}>Faithfulness</div>
                <div style={styles.metricValue}>{ragas.faithfulness.toFixed(3)}</div>
              </div>
              <div style={styles.metric}>
                <div style={styles.metricLabel}>Answer Relevancy</div>
                <div style={styles.metricValue}>{ragas.answer_relevancy.toFixed(3)}</div>
              </div>
              <div style={styles.metric}>
                <div style={styles.metricLabel}>Context Precision</div>
                <div style={styles.metricValue}>{ragas.context_precision.toFixed(3)}</div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
