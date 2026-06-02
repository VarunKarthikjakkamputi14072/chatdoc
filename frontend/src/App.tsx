import { useState } from "react";
import UploadPanel from "./components/UploadPanel";
import ChatPanel from "./components/ChatPanel";
import EvalPanel from "./components/EvalPanel";
import { IngestResponse } from "./lib/api";

const styles: Record<string, React.CSSProperties> = {
  app: { display: "flex", height: "100vh", overflow: "hidden" },
  sidebar: {
    width: "360px",
    minWidth: "300px",
    borderRight: "1px solid #2a2a2a",
    display: "flex",
    flexDirection: "column",
    background: "#111",
    overflow: "auto",
  },
  sidebarHeader: {
    padding: "1.25rem 1.5rem",
    borderBottom: "1px solid #2a2a2a",
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
  },
  logo: { fontSize: "1.1rem", fontWeight: 700, color: "#e8e8e8" },
  badge: {
    fontSize: "0.7rem",
    background: "#1e3a5f",
    color: "#4f8ef7",
    padding: "0.15rem 0.5rem",
    borderRadius: "4px",
    fontWeight: 600,
  },
  docList: { padding: "1rem 1.5rem", flex: 1 },
  docListTitle: { fontSize: "0.75rem", color: "#555", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "0.5rem" },
  docItem: { fontSize: "0.82rem", color: "#999", padding: "0.3rem 0", borderBottom: "1px solid #1e1e1e" },
  main: { flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" },
};

interface Doc {
  source: string;
  chunks: number;
}

export default function App() {
  const [docs, setDocs] = useState<Doc[]>([]);

  const handleIngested = (res: IngestResponse) => {
    setDocs((prev) => {
      const existing = prev.findIndex((d) => d.source === res.source);
      if (existing !== -1) {
        const updated = [...prev];
        updated[existing] = { source: res.source, chunks: res.chunks_indexed };
        return updated;
      }
      return [...prev, { source: res.source, chunks: res.chunks_indexed }];
    });
  };

  return (
    <div style={styles.app}>
      <aside style={styles.sidebar}>
        <div style={styles.sidebarHeader}>
          <span style={styles.logo}>ChatDoc</span>
          <span style={styles.badge}>Hybrid RAG</span>
        </div>
        <UploadPanel onIngested={handleIngested} />
        {docs.length > 0 && (
          <div style={styles.docList}>
            <div style={styles.docListTitle}>Indexed Documents</div>
            {docs.map((d) => (
              <div key={d.source} style={styles.docItem}>
                {d.source} <span style={{ color: "#555" }}>({d.chunks} chunks)</span>
              </div>
            ))}
          </div>
        )}
        <div style={{ flex: 1 }} />
        <EvalPanel />
      </aside>

      <main style={styles.main}>
        <ChatPanel hasDocuments={docs.length > 0} />
      </main>
    </div>
  );
}
