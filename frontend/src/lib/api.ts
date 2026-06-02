const BASE = "/api/v1";

export interface ChatResponse {
  answer: string;
  sources: { source: string; chunk_index: number | null }[];
  chunks_used: number;
}

export interface IngestResponse {
  source: string;
  chunks_indexed: number;
  chunk_ids: string[];
}

export interface RetrievalMetrics {
  recall_at_k: number;
  mrr_at_k: number;
  k: number;
  num_queries: number;
}

export interface RagasMetrics {
  faithfulness: number;
  answer_relevancy: number;
  context_precision: number;
  num_samples: number;
}

export async function ingestFile(file: File): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/ingest/file`, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail ?? "Ingest failed");
  return res.json();
}

export async function ingestUrl(url: string): Promise<IngestResponse> {
  const res = await fetch(`${BASE}/ingest/url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!res.ok) throw new Error((await res.json()).detail ?? "Ingest failed");
  return res.json();
}

export async function chat(query: string): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/chat/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, stream: false }),
  });
  if (!res.ok) throw new Error((await res.json()).detail ?? "Chat failed");
  return res.json();
}

export type ChatStreamEvent =
  | { type: "token"; value: string }
  | { type: "sources"; value: ChatResponse["sources"] };

export async function* chatStream(query: string): AsyncGenerator<ChatStreamEvent> {
  const res = await fetch(`${BASE}/chat/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, stream: true }),
  });
  if (!res.ok || !res.body) throw new Error("Stream failed");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);

      const data = frame
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trim())
        .join("");
      if (!data || data === "[DONE]") continue;

      yield JSON.parse(data) as ChatStreamEvent;
    }
  }
}

export async function generateEvalDataset(numPairs = 20): Promise<{ num_pairs: number }> {
  const res = await fetch(`${BASE}/eval/generate?num_pairs=${numPairs}`, { method: "POST" });
  if (!res.ok) throw new Error((await res.json()).detail ?? "Generate failed");
  return res.json();
}

export async function runRetrievalEval(): Promise<RetrievalMetrics> {
  const res = await fetch(`${BASE}/eval/retrieval`);
  if (!res.ok) throw new Error((await res.json()).detail ?? "Eval failed");
  return res.json();
}

export async function runRagasEval(): Promise<RagasMetrics> {
  const res = await fetch(`${BASE}/eval/ragas`);
  if (!res.ok) throw new Error((await res.json()).detail ?? "RAGAS eval failed");
  return res.json();
}
