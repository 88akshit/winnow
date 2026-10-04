import { useRef, useState } from "react";
import { chatWithWinnow, uploadDataset } from "./services/winnow.js";

const starterPrompts = [
  { icon: "↗", title: "Find the strongest relationships", text: "Which variables are most strongly related?" },
  { icon: "◷", title: "Check data quality", text: "Are there missing values or unusual outliers?" },
  { icon: "▥", title: "Compare groups", text: "Compare the key metrics across categories." },
];

function Icon({ children, className = "" }) {
  return <span aria-hidden="true" className={`icon ${className}`}>{children}</span>;
}

function App() {
  const fileInput = useRef(null);
  const [file, setFile] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [threadId, setThreadId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [busyLabel, setBusyLabel] = useState("");
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [plots, setPlots] = useState([]);

  const acceptFile = (incoming) => {
    const candidate = incoming?.[0] || incoming;
    if (!candidate) return;
    setError("");
    setFile(candidate);
  };

  async function uploadAndAnalyze(event) {
    event?.preventDefault();
    if (!file || busy) return;
    setBusy(true);
    setBusyLabel("Reading your file and starting the analysis…");
    setError("");
    setMessages([]);
    setPlots([]);
    try {
      const result = await uploadDataset(file);
      setDataset(result.dataset);
      setThreadId(result.thread_id);
      setMessages([]);
      setPlots([]);
      setFile(null);

      setBusyLabel("Winnow is inspecting your data…");
      const analysis = await chatWithWinnow(
        `I uploaded the dataset '${result.dataset.name}'. Begin the analysis workflow now. Inspect its shape, columns, missing values, and sample rows, then give me a concise initial exploratory analysis with useful findings and suitable visualizations.`,
        result.thread_id,
      );
      setMessages([{ role: "assistant", text: analysis.assistant }]);
      setPlots(analysis.plots || []);
    } catch (err) {
      setError(err.message || "Something went wrong while analyzing the file.");
    } finally {
      setBusy(false);
      setBusyLabel("");
    }
  }

  async function sendMessage(text = draft) {
    const message = text.trim();
    if (!message || !threadId || busy) return;
    setDraft("");
    setMessages((current) => [...current, { role: "user", text: message }]);
    setBusy(true);
    setBusyLabel("Winnow is working through your question…");
    setError("");
    try {
      const result = await chatWithWinnow(message, threadId);
      setMessages((current) => [...current, { role: "assistant", text: result.assistant }]);
      setPlots((current) => [...new Set([...current, ...(result.plots || [])])]);
    } catch (err) {
      setError(err.message || "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
      setBusyLabel("");
    }
  }

  const onDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files);
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#top" aria-label="Winnow home">
          <span className="brand-mark"><span /><span /><span /></span>
          <span>Winnow</span>
        </a>

        <div className="workspace-label">WORKSPACE</div>
        <button className="nav-item active" type="button"><Icon>◫</Icon> Data studio <span className="nav-dot" /></button>
        <button className="nav-item" type="button" onClick={() => { setDataset(null); setThreadId(null); setMessages([]); setPlots([]); setFile(null); setError(""); }}><Icon>＋</Icon> New analysis</button>

        <div className="sidebar-divider" />
        <div className="sidebar-heading">CURRENT DATASET</div>
        {dataset ? (
          <div className="dataset-mini">
            <span className="file-badge">{dataset.file_type === "xlsx" ? "XLS" : "CSV"}</span>
            <div className="dataset-mini-copy"><strong>{dataset.name}</strong><span>{dataset.rows.toLocaleString()} rows · {dataset.columns} columns</span></div>
            <span className="status-check">✓</span>
          </div>
        ) : (
          <div className="empty-dataset">Your uploaded file will appear here.</div>
        )}

        <div className="sidebar-bottom">
          <div className="privacy-card"><span className="privacy-icon">◈</span><div><strong>Your workspace, your files</strong><p>Files are stored here; analysis uses your configured model.</p></div></div>
          <div className="profile"><div className="avatar">W</div><div><strong>Winnow workspace</strong><span>Personal</span></div><span className="profile-menu">···</span></div>
        </div>
      </aside>

      <main className="main-panel" id="top">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><b>/</b><strong>Data studio</strong>{dataset && <><b>/</b><strong className="crumb-dataset">{dataset.name}</strong></>}</div>
          <div className="topbar-right"><span className="connection"><i /> Local workspace</span><button className="help-button" aria-label="Help">?</button></div>
        </header>

        {!dataset ? (
          <section className="welcome-view">
            <div className="welcome-copy">
              <div className="eyebrow"><span className="eyebrow-line" /> YOUR PERSONAL DATA ANALYST</div>
              <h1>Make sense of<br /><em>your data.</em></h1>
              <p className="welcome-subtitle">Drop in a dataset. Winnow will explore the details, find what matters, and help you ask better questions.</p>
            </div>
            <form className={`upload-card ${dragging ? "dragging" : ""}`} onSubmit={uploadAndAnalyze} onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
              <input ref={fileInput} className="visually-hidden" type="file" accept=".csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" onChange={(e) => acceptFile(e.target.files)} />
              <div className="upload-icon"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 15.5v3A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5v-3" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" /></svg></div>
              {file ? <><h2>Ready to explore</h2><p className="file-selected"><span className="file-badge">{file.name.toLowerCase().endsWith(".xlsx") ? "XLS" : "CSV"}</span>{file.name}<button type="button" className="remove-file" onClick={() => setFile(null)} aria-label="Remove file">×</button></p></> : <><h2>Start with a dataset</h2><p>Drag a file here, or <button className="text-button" type="button" onClick={() => fileInput.current?.click()}>browse files</button></p></>}
              <div className="file-rules"><span>CSV or Excel</span><i /> <span>Up to 25 MB</span></div>
              {error && <div className="error-banner upload-error"><strong>Upload didn’t finish</strong><span>{error}</span></div>}
              <button className="primary-button upload-submit" type="submit" disabled={!file || busy}>{busy ? <><span className="spinner" /> Starting analysis</> : <>Upload and analyze <span>→</span></>}</button>
            </form>
            <div className="workflow-note"><span className="sparkle">✳</span><span><strong>A little more clarity, right away.</strong> Get an overview, data quality notes, patterns, and charts automatically.</span></div>
          </section>
        ) : (
          <section className="analysis-layout">
            <div className="analysis-main">
              <div className="analysis-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> ANALYSIS WORKSPACE</div><h1>{dataset.name.replace(/_[a-f0-9]{8}$/, "")}</h1><p>{dataset.rows.toLocaleString()} rows <span>·</span> {dataset.columns} columns <span>·</span> Ready to explore</p></div><button className="outline-button" onClick={() => { setDataset(null); setThreadId(null); setMessages([]); setFile(null); }}><span>＋</span> New file</button></div>

              <div className="metric-row"><div className="metric-card"><span>OBSERVATIONS</span><strong>{dataset.rows.toLocaleString()}</strong><small>rows in your dataset</small></div><div className="metric-card"><span>VARIABLES</span><strong>{dataset.columns}</strong><small>columns to explore</small></div><div className="metric-card"><span>FILE STATUS</span><strong className="ready-text"><i /> Ready</strong><small>analysis in progress</small></div></div>

              {messages.length === 1 && !busy && <div className="suggestion-block"><div className="section-label">A GOOD PLACE TO START</div><div className="suggestion-grid">{starterPrompts.map((prompt) => <button className="suggestion-card" key={prompt.title} onClick={() => sendMessage(prompt.text)}><span className="suggestion-icon">{prompt.icon}</span><strong>{prompt.title}</strong><span>{prompt.text}</span><b>↗</b></button>)}</div></div>}

              <div className="conversation">
                <div className="section-label conversation-label">YOUR ANALYSIS <span className="conversation-line" /></div>
                {messages.map((message, index) => <article className={`message ${message.role}`} key={`${message.role}-${index}`}><div className={message.role === "assistant" ? "message-avatar" : "user-avatar"}>{message.role === "assistant" ? <span className="brand-mark small"><span /><span /><span /></span> : "A"}</div><div className="message-content"><div className="message-meta"><strong>{message.role === "assistant" ? "Winnow" : "You"}</strong><span>{message.role === "assistant" ? "DATA ANALYST" : "JUST NOW"}</span></div><div className="message-text">{message.text}</div></div></article>)}
                {busy && <div className="message assistant"><div className="message-avatar"><span className="brand-mark small"><span /><span /><span /></span></div><div className="message-content"><div className="message-meta"><strong>Winnow</strong><span>WORKING</span></div><div className="typing-row"><span className="typing-dots"><i /><i /><i /></span>{busyLabel}</div></div></div>}
                {plots.length > 0 && <div className="plot-grid">{plots.map((plot) => <figure className="plot-card" key={plot}><img src={plot} alt="Chart from your analysis" /><figcaption>Generated from your dataset</figcaption></figure>)}</div>}
                {error && <div className="error-banner"><strong>Something went wrong</strong><span>{error}</span></div>}
              </div>

              <form className="composer" onSubmit={(e) => { e.preventDefault(); sendMessage(); }}><textarea value={draft} onChange={(e) => setDraft(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }} placeholder="Ask anything about your data…" rows="1" disabled={busy} /><div className="composer-footer"><span>Winnow can make mistakes. Check important findings.</span><button className="send-button" type="submit" disabled={!draft.trim() || busy} aria-label="Send message"><svg viewBox="0 0 24 24" fill="none"><path d="M5 12h14m-6-6 6 6-6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg></button></div></form>
            </div>

            <aside className="data-panel"><div className="data-panel-head"><div><div className="section-label">DATA OVERVIEW</div><p>Quick look at your file</p></div><span className="more-button">···</span></div><div className="data-file-card"><span className="file-badge">{dataset.file_type === "xlsx" ? "XLS" : "CSV"}</span><div><strong>{dataset.name.replace(/_[a-f0-9]{8}$/, "")}</strong><span>{dataset.rows.toLocaleString()} × {dataset.columns}</span></div></div><div className="preview-title">COLUMN PREVIEW <span>{dataset.columns} total</span></div><div className="column-list">{dataset.column_names.slice(0, 12).map((column, index) => <div className="column-item" key={column}><span className="column-type">{typeof dataset.preview?.[0]?.[column] === "number" ? "#" : "T"}</span><span>{column}</span><span className="column-index">{String(index + 1).padStart(2, "0")}</span></div>)}{dataset.columns > 12 && <div className="more-columns">+ {dataset.columns - 12} more columns</div>}</div><div className="preview-title table-title">SAMPLE ROWS <span>First {Math.min(dataset.preview.length, 8)}</span></div><div className="table-wrap"><table><thead><tr>{dataset.column_names.slice(0, 3).map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{dataset.preview.slice(0, 5).map((row, index) => <tr key={index}>{dataset.column_names.slice(0, 3).map((column) => <td key={column}>{String(row[column] ?? "—")}</td>)}</tr>)}</tbody></table></div><div className="data-panel-foot"><span className="ready-pip" /> File loaded to this workspace</div></aside>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
