import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { ingestFile, ingestUrl, IngestResponse } from "../lib/api";

interface Props {
  onIngested: (res: IngestResponse) => void;
}

const styles: Record<string, React.CSSProperties> = {
  panel: { padding: "1.5rem", borderBottom: "1px solid #2a2a2a" },
  dropzone: {
    border: "2px dashed #444",
    borderRadius: "8px",
    padding: "2rem",
    textAlign: "center",
    cursor: "pointer",
    color: "#888",
    transition: "border-color 0.2s",
  },
  dropzoneActive: { borderColor: "#4f8ef7", color: "#4f8ef7" },
  urlRow: { display: "flex", gap: "0.5rem", marginTop: "1rem" },
  input: {
    flex: 1,
    background: "#1a1a1a",
    border: "1px solid #333",
    borderRadius: "6px",
    padding: "0.5rem 0.75rem",
    color: "#e8e8e8",
    fontSize: "0.9rem",
  },
  btn: {
    background: "#4f8ef7",
    color: "#fff",
    border: "none",
    borderRadius: "6px",
    padding: "0.5rem 1rem",
    cursor: "pointer",
    fontSize: "0.9rem",
  },
  status: { marginTop: "0.75rem", fontSize: "0.85rem", color: "#6dbf6d" },
  error: { marginTop: "0.75rem", fontSize: "0.85rem", color: "#e06c6c" },
};

export default function UploadPanel({ onIngested }: Props) {
  const [url, setUrl] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleFile = useCallback(async (file: File) => {
    setLoading(true);
    setStatus(null);
    setError(null);
    try {
      const res = await ingestFile(file);
      setStatus(`Indexed ${res.chunks_indexed} chunks from ${res.source}`);
      onIngested(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  }, [onIngested]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (files) => files[0] && handleFile(files[0]),
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
      "text/plain": [".txt"],
      "text/markdown": [".md"],
    },
    multiple: false,
    disabled: loading,
  });

  const handleUrlIngest = async () => {
    if (!url.trim()) return;
    setLoading(true);
    setStatus(null);
    setError(null);
    try {
      const res = await ingestUrl(url.trim());
      setStatus(`Indexed ${res.chunks_indexed} chunks from ${res.source}`);
      onIngested(res);
      setUrl("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "URL ingest failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.panel}>
      <div
        {...getRootProps()}
        style={{ ...styles.dropzone, ...(isDragActive ? styles.dropzoneActive : {}) }}
      >
        <input {...getInputProps()} />
        {loading
          ? "Processing..."
          : isDragActive
          ? "Drop it here"
          : "Drop a PDF, DOCX, TXT, or MD file — or click to browse"}
      </div>

      <div style={styles.urlRow}>
        <input
          style={styles.input}
          placeholder="Or paste a URL to ingest..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleUrlIngest()}
          disabled={loading}
        />
        <button style={styles.btn} onClick={handleUrlIngest} disabled={loading || !url.trim()}>
          Ingest
        </button>
      </div>

      {status && <div style={styles.status}>{status}</div>}
      {error && <div style={styles.error}>{error}</div>}
    </div>
  );
}
