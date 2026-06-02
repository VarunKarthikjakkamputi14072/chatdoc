import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { chatStream, ChatResponse } from "../lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: ChatResponse["sources"];
}

const styles: Record<string, React.CSSProperties> = {
  container: { display: "flex", flexDirection: "column", height: "100%" },
  messages: { flex: 1, overflowY: "auto", padding: "1.5rem", display: "flex", flexDirection: "column", gap: "1.25rem" },
  bubble: (role: "user" | "assistant"): React.CSSProperties => ({
    maxWidth: "75%",
    alignSelf: role === "user" ? "flex-end" : "flex-start",
    background: role === "user" ? "#1e3a5f" : "#1e1e1e",
    borderRadius: "12px",
    padding: "0.75rem 1rem",
    lineHeight: 1.6,
    fontSize: "0.9rem",
  }),
  sources: {
    marginTop: "0.5rem",
    fontSize: "0.78rem",
    color: "#666",
    borderTop: "1px solid #2a2a2a",
    paddingTop: "0.4rem",
  },
  inputRow: {
    display: "flex",
    gap: "0.5rem",
    padding: "1rem 1.5rem",
    borderTop: "1px solid #2a2a2a",
  },
  input: {
    flex: 1,
    background: "#1a1a1a",
    border: "1px solid #333",
    borderRadius: "8px",
    padding: "0.65rem 1rem",
    color: "#e8e8e8",
    fontSize: "0.9rem",
    resize: "none",
  },
  sendBtn: {
    background: "#4f8ef7",
    color: "#fff",
    border: "none",
    borderRadius: "8px",
    padding: "0 1.25rem",
    cursor: "pointer",
    fontSize: "0.9rem",
    fontWeight: 500,
  },
  empty: { flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "#444", fontSize: "0.9rem" },
};

interface Props {
  hasDocuments: boolean;
}

export default function ChatPanel({ hasDocuments }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    if (!query.trim() || loading) return;
    const userMsg: Message = { role: "user", content: query.trim() };
    setMessages((prev) => [...prev, userMsg]);
    setQuery("");
    setLoading(true);

    const assistantMsg: Message = { role: "assistant", content: "" };
    setMessages((prev) => [...prev, assistantMsg]);

    try {
      for await (const token of chatStream(userMsg.content)) {
        assistantMsg.content += token;
        setMessages((prev) => [
          ...prev.slice(0, -1),
          { ...assistantMsg },
        ]);
      }
    } catch (e) {
      assistantMsg.content = e instanceof Error ? e.message : "Something went wrong.";
      setMessages((prev) => [...prev.slice(0, -1), { ...assistantMsg }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      {messages.length === 0 ? (
        <div style={styles.empty}>
          {hasDocuments ? "Ask a question about your documents." : "Upload a document to get started."}
        </div>
      ) : (
        <div style={styles.messages}>
          {messages.map((msg, i) => (
            <div key={i} style={styles.bubble(msg.role)}>
              <ReactMarkdown>{msg.content}</ReactMarkdown>
              {msg.sources && msg.sources.length > 0 && (
                <div style={styles.sources}>
                  Sources: {[...new Set(msg.sources.map((s) => s.source))].join(", ")}
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      )}

      <div style={styles.inputRow}>
        <textarea
          style={styles.input}
          rows={1}
          placeholder={hasDocuments ? "Ask anything about your documents..." : "Upload a document first"}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          disabled={!hasDocuments || loading}
        />
        <button style={styles.sendBtn} onClick={send} disabled={!hasDocuments || loading || !query.trim()}>
          {loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}
