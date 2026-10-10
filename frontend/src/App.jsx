import { useEffect, useRef, useState } from "react";
import { chatWithWinnow, getDatasets, resumeChat, uploadDataset } from "./services/winnow.js";
import FormattedMessage from "./FormattedMessage.jsx";

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
  const [analysisQuestion, setAnalysisQuestion] = useState("");
  const [analysisDatasetChoice, setAnalysisDatasetChoice] = useState("__new__");
  const [chatQuestion, setChatQuestion] = useState("");
  const [workspaceMode, setWorkspaceMode] = useState("dataset");
  const [chatStarted, setChatStarted] = useState(false);
  const [datasetOptions, setDatasetOptions] = useState([]);
  const [selectedChatDataset, setSelectedChatDataset] = useState("");
  const [dataset, setDataset] = useState(null);
  const [threadId, setThreadId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [busyLabel, setBusyLabel] = useState("");
  const [error, setError] = useState("");
  const [errorTitle, setErrorTitle] = useState("Something went wrong");
  const [dragging, setDragging] = useState(false);
  const [plots, setPlots] = useState([]);
  const [clarification, setClarification] = useState("");
  const simpleChatSession = chatStarted && workspaceMode === "chat";

  useEffect(() => {
    getDatasets().then((result) => setDatasetOptions(result.datasets || [])).catch(() => {});
  }, []);

  const acceptFile = (incoming) => {
    const candidate = incoming?.[0] || incoming;
    if (!candidate) return;
    setError("");
    setErrorTitle("Something went wrong");
    setFile(candidate);
  };

  async function uploadAndAnalyze(event) {
    event?.preventDefault();
    const selectedName = analysisDatasetChoice === "__new__" ? null : analysisDatasetChoice;
    if ((!file && !selectedName) || !analysisQuestion.trim() || busy) return;
    setBusy(true);
    setBusyLabel("Reading your file and starting the analysis…");
    setError("");
    setMessages([]);
    setPlots([]);
    try {
      let activeDataset = datasetOptions.find((item) => item.name === selectedName) || null;
      let activeThreadId = crypto.randomUUID();
      if (selectedName) {
        if (!activeDataset) throw new Error("That dataset is no longer available. Refresh the page and select it again.");
      } else {
        const uploaded = await uploadDataset(file);
        activeDataset = uploaded.dataset;
        activeThreadId = uploaded.thread_id;
        setDatasetOptions((current) => [uploaded.dataset, ...current.filter((item) => item.name !== uploaded.dataset.name)]);
      }
      setDataset(activeDataset);
      setThreadId(activeThreadId);
      setMessages([{ role: "user", text: analysisQuestion.trim() }]);
      setPlots([]);
      setFile(null);

      setBusyLabel("Winnow is inspecting your data…");
      const analysis = await chatWithWinnow(
        `Use the selected dataset '${activeDataset.name}'. I want to dive deep into this question: ${analysisQuestion.trim()}\n\nAnalyze this question using the selected data. Inspect the relevant columns and data quality, run appropriate analysis, and provide an evidence-based answer with useful findings, visualizations where they help, and any important limitations.`,
        activeThreadId,
        "dataset",
        activeDataset.name,
      );
      if (!analysis.clarification) {
        setMessages((current) => [...current, { role: "assistant", text: analysis.assistant }]);
      }
      setClarification(analysis.clarification || "");
      setPlots(analysis.plots || []);
    } catch (err) {
      setErrorTitle(err.status === 429 ? "Provider is busy" : err.status === 413 ? "Request exceeds provider limit" : "Something went wrong");
      setError(err.message || "Something went wrong while analyzing the file.");
    } finally {
      setBusy(false);
      setBusyLabel("");
    }
  }

  async function startSimpleChat(event) {
    event?.preventDefault();
    const message = chatQuestion.trim();
    if (!message || busy) return;
    setBusy(true);
    setBusyLabel(file ? "Uploading your dataset…" : "Winnow is thinking…");
    setError("");
    setErrorTitle("Something went wrong");
    try {
      let selectedName = selectedChatDataset || null;
      let selected = datasetOptions.find((item) => item.name === selectedName) || null;
      let activeThreadId = crypto.randomUUID();
      if (file) {
        const uploaded = await uploadDataset(file);
        setDatasetOptions((current) => [uploaded.dataset, ...current.filter((item) => item.name !== uploaded.dataset.name)]);
        selectedName = uploaded.dataset.name;
        selected = uploaded.dataset;
        activeThreadId = uploaded.thread_id;
        setSelectedChatDataset(selectedName);
        setFile(null);
      }
      setDataset(selected);
      setThreadId(activeThreadId);
      setChatStarted(true);
      setMessages([{ role: "user", text: message }]);
      setPlots([]);
      setClarification("");
      setChatQuestion("");
      setBusyLabel("Winnow is thinking…");
      const result = await chatWithWinnow(message, activeThreadId, "simple", selectedName);
      setClarification(result.clarification || "");
      if (!result.clarification) {
        setMessages((current) => [...current, { role: "assistant", text: result.assistant }]);
      }
      setPlots(result.plots || []);
    } catch (err) {
      setErrorTitle(err.status === 429 ? "Provider is busy" : err.status === 413 ? "Request exceeds provider limit" : "Something went wrong");
      setError(err.message || "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
      setBusyLabel("");
    }
  }

  async function uploadDatasetForChat() {
    if (!file || busy) return;
    setBusy(true);
    setBusyLabel("Uploading your dataset…");
    setError("");
    setErrorTitle("Dataset upload failed");
    try {
      const result = await uploadDataset(file);
      setDatasetOptions((current) => [result.dataset, ...current.filter((item) => item.name !== result.dataset.name)]);
      setSelectedChatDataset(result.dataset.name);
      setFile(null);
    } catch (err) {
      setError(err.message || "Something went wrong while uploading the file.");
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
    setErrorTitle("Something went wrong");
    try {
      const result = clarification
        ? await resumeChat(message, threadId)
        : await chatWithWinnow(
          message,
          threadId,
          simpleChatSession ? "simple" : "dataset",
          dataset?.name || null,
        );
      setClarification(result.clarification || "");
      if (!result.clarification) {
        setMessages((current) => [...current, { role: "assistant", text: result.assistant }]);
      }
      setPlots((current) => [...new Set([...current, ...(result.plots || [])])]);
    } catch (err) {
      setErrorTitle(err.status === 429 ? "Provider is busy" : err.status === 413 ? "Request exceeds provider limit" : "Something went wrong");
      setError(err.message || "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
      setBusyLabel("");
    }
  }

  const onDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    if (workspaceMode === "dataset") setAnalysisDatasetChoice("__new__");
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
        <button className="nav-item" type="button" onClick={() => { setDataset(null); setThreadId(null); setMessages([]); setPlots([]); setFile(null); setAnalysisQuestion(""); setChatQuestion(""); setChatStarted(false); setWorkspaceMode("dataset"); setError(""); setClarification(""); }}><Icon>＋</Icon> New analysis</button>

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

        {!dataset && !chatStarted ? (
          <section className="welcome-view">
            <div className="welcome-mode-switch" role="group" aria-label="Choose how to start">
              <button type="button" aria-pressed={workspaceMode === "dataset"} className={workspaceMode === "dataset" ? "selected" : ""} onClick={() => setWorkspaceMode("dataset")}>Analyze a dataset</button>
              <button type="button" aria-pressed={workspaceMode === "chat"} className={workspaceMode === "chat" ? "selected" : ""} onClick={() => setWorkspaceMode("chat")}>Simple chat</button>
            </div>
            <div className="welcome-copy">
              <div className="eyebrow"><span className="eyebrow-line" /> {workspaceMode === "dataset" ? "YOUR PERSONAL DATA ANALYST" : "ASK WINNOW ANYTHING"}</div>
              <h1>{workspaceMode === "dataset" ? <>Make sense of<br /><em>your data.</em></> : <>A simple<br /><em>conversation.</em></>}</h1>
              <p className="welcome-subtitle">{workspaceMode === "dataset" ? "Choose a dataset already in your workspace or upload a new one, then ask Winnow to investigate it." : "Choose a loaded dataset for context, upload one, or start a general conversation."}</p>
            </div>
            {workspaceMode === "dataset" ? (
            <form className={`upload-card ${dragging ? "dragging" : ""}`} onSubmit={uploadAndAnalyze} onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
              <input ref={fileInput} className="visually-hidden" type="file" accept=".csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" onChange={(e) => { acceptFile(e.target.files); setAnalysisDatasetChoice("__new__"); e.target.value = ""; }} />
              <label className="chat-dataset-label" htmlFor="analysis-dataset">Dataset</label>
              <select id="analysis-dataset" className="chat-dataset-select-input" value={analysisDatasetChoice} onChange={(e) => setAnalysisDatasetChoice(e.target.value)} disabled={busy}>
                <option value="__new__">Upload a new dataset</option>
                {datasetOptions.map((option) => <option key={option.name} value={option.name}>{option.name} · {option.rows.toLocaleString()} rows</option>)}
              </select>
              {analysisDatasetChoice === "__new__" ? <>
                <div className="upload-icon"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 15.5v3A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5v-3" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" /></svg></div>
                {file ? <><h2>Ready to explore</h2><p className="file-selected"><span className="file-badge">{file.name.toLowerCase().endsWith(".xlsx") ? "XLS" : "CSV"}</span>{file.name}<button type="button" className="remove-file" onClick={() => setFile(null)} aria-label="Remove file">×</button></p></> : <><h2>Upload a dataset</h2><p>Drag a file here, or <button className="text-button" type="button" onClick={() => fileInput.current?.click()}>browse files</button></p></>}
                <div className="file-rules"><span>CSV or Excel</span><i /> <span>Up to 25 MB</span></div>
              </> : <p className="selected-dataset-note">Using {datasetOptions.find((option) => option.name === analysisDatasetChoice)?.rows?.toLocaleString() || ""} rows from the selected dataset.</p>}
              <div className="deep-question">
                <label htmlFor="analysis-question">What would you like to dive deep into?</label>
                <textarea id="analysis-question" value={analysisQuestion} onChange={(e) => setAnalysisQuestion(e.target.value)} placeholder="e.g., Which factors are most associated with customer churn?" rows={3} maxLength={8000} required disabled={busy} />
              </div>
              {error && <div className="error-banner upload-error"><strong>Upload didn’t finish</strong><span>{error}</span></div>}
              <button className="primary-button upload-submit" type="submit" disabled={(analysisDatasetChoice === "__new__" && !file) || !analysisQuestion.trim() || busy}>{busy ? <><span className="spinner" /> Starting analysis</> : <>Start analysis <span>→</span></>}</button>
            </form>
            ) : (
              <form className="upload-card simple-chat-start" onSubmit={startSimpleChat}>
                <input ref={fileInput} className="visually-hidden" type="file" accept=".csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" onChange={(e) => acceptFile(e.target.files)} />
                <div className="upload-icon" aria-hidden="true">✳</div>
                <h2>What’s on your mind?</h2>
                <p>Choose a dataset for context, or start without one.</p>
                <label className="chat-dataset-label" htmlFor="chat-dataset">Dataset</label>
                <select id="chat-dataset" className="chat-dataset-select-input" value={selectedChatDataset} onChange={(e) => setSelectedChatDataset(e.target.value)} disabled={busy}>
                  <option value="">No dataset</option>
                  {datasetOptions.map((option) => <option key={option.name} value={option.name}>{option.name} · {option.rows.toLocaleString()} rows</option>)}
                </select>
                <p className="chat-context-note">Simple chat uses the dataset summary and sample rows. Use Analyze a dataset for exact calculations across the full file.</p>
                <div className="chat-dataset-upload">
                  <button className="text-button" type="button" onClick={() => fileInput.current?.click()} disabled={busy}>＋ Upload a CSV or Excel file</button>
                  {file && <><span className="chat-file-name">{file.name}</span><button className="text-button" type="button" onClick={uploadDatasetForChat} disabled={busy}>{busy ? "Uploading…" : "Upload dataset"}</button><button type="button" className="remove-file" onClick={() => setFile(null)} aria-label="Remove file">×</button></>}
                </div>
                <textarea className="chat-start-question" value={chatQuestion} onChange={(e) => setChatQuestion(e.target.value)} placeholder="Ask Winnow anything…" rows={4} maxLength={8000} required disabled={busy} />
                {error && <div className="error-banner upload-error"><strong>{errorTitle}</strong><span>{error}</span></div>}
                <button className="primary-button upload-submit" type="submit" disabled={!chatQuestion.trim() || busy}>{busy ? <><span className="spinner" /> Starting chat</> : <>Start chatting <span>→</span></>}</button>
              </form>
            )}
            {workspaceMode === "dataset" && <div className="workflow-note"><span className="sparkle">✳</span><span><strong>A little more clarity, right away.</strong> Get an overview, data quality notes, patterns, and charts automatically.</span></div>}
          </section>
        ) : (
          <section className={`analysis-layout ${dataset && !simpleChatSession ? "" : "chat-only-layout"}`}>
            <div className="analysis-main">
              <div className="analysis-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> {simpleChatSession ? "SIMPLE CHAT" : "ANALYSIS WORKSPACE"}</div><h1>{simpleChatSession ? "Chat with Winnow" : dataset ? dataset.name.replace(/_[a-f0-9]{8}$/, "") : "Chat with Winnow"}</h1><p>{dataset && !simpleChatSession ? <>{dataset.rows.toLocaleString()} rows <span>·</span> {dataset.columns} columns <span>·</span> Ready to explore</> : dataset ? <>Chatting with {dataset.name} <span>·</span> {dataset.rows.toLocaleString()} rows</> : "No dataset attached"}</p></div><button className="outline-button" onClick={() => { setDataset(null); setThreadId(null); setMessages([]); setFile(null); setAnalysisQuestion(""); setChatQuestion(""); setChatStarted(false); setWorkspaceMode(simpleChatSession ? "chat" : "dataset"); setClarification(""); }}><span>＋</span> New {simpleChatSession || !dataset ? "chat" : "file"}</button></div>

              {dataset && !simpleChatSession && <div className="metric-row"><div className="metric-card"><span>OBSERVATIONS</span><strong>{dataset.rows.toLocaleString()}</strong><small>rows in your dataset</small></div><div className="metric-card"><span>VARIABLES</span><strong>{dataset.columns}</strong><small>columns to explore</small></div><div className="metric-card"><span>FILE STATUS</span><strong className="ready-text"><i /> Ready</strong><small>analysis in progress</small></div></div>}

              {dataset && !simpleChatSession && messages.length === 1 && !busy && <div className="suggestion-block"><div className="section-label">A GOOD PLACE TO START</div><div className="suggestion-grid">{starterPrompts.map((prompt) => <button className="suggestion-card" key={prompt.title} onClick={() => sendMessage(prompt.text)}><span className="suggestion-icon">{prompt.icon}</span><strong>{prompt.title}</strong><span>{prompt.text}</span><b>↗</b></button>)}</div></div>}

              <div className="conversation">
                <div className="section-label conversation-label">{simpleChatSession ? "CONVERSATION" : "YOUR ANALYSIS"} <span className="conversation-line" /></div>
                {clarification && <div className="clarification-banner"><strong>Winnow needs one detail</strong><span>{clarification}</span></div>}
                {messages.map((message, index) => <article className={`message ${message.role}`} key={`${message.role}-${index}`}><div className={message.role === "assistant" ? "message-avatar" : "user-avatar"}>{message.role === "assistant" ? <span className="brand-mark small"><span /><span /><span /></span> : "A"}</div><div className="message-content"><div className="message-meta"><strong>{message.role === "assistant" ? "Winnow" : "You"}</strong><span>{message.role === "assistant" ? simpleChatSession ? "CHAT ASSISTANT" : "DATA ANALYST" : "JUST NOW"}</span></div>{message.role === "assistant" ? <FormattedMessage content={message.text} /> : <div className="message-text">{message.text}</div>}</div></article>)}
                {busy && <div className="message assistant"><div className="message-avatar"><span className="brand-mark small"><span /><span /><span /></span></div><div className="message-content"><div className="message-meta"><strong>Winnow</strong><span>WORKING</span></div><div className="typing-row"><span className="typing-dots"><i /><i /><i /></span>{busyLabel}</div></div></div>}
                {plots.length > 0 && <div className="plot-grid">{plots.map((plot) => <figure className="plot-card" key={plot}><img src={plot} alt="Chart from your analysis" /><figcaption>Generated from your dataset</figcaption></figure>)}</div>}
                {error && <div className="error-banner"><strong>{errorTitle}</strong><span>{error}</span></div>}
              </div>

              <form className="composer" onSubmit={(e) => { e.preventDefault(); sendMessage(); }}><textarea value={draft} onChange={(e) => setDraft(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }} placeholder={clarification ? "Answer Winnow’s question…" : dataset && !simpleChatSession ? "Ask anything about your data…" : "Message Winnow…"} rows="1" disabled={busy} /><div className="composer-footer"><span>Winnow can make mistakes. Check important findings.</span><button className="send-button" type="submit" disabled={!draft.trim() || busy} aria-label={clarification ? "Answer clarification" : "Send message"}><svg viewBox="0 0 24 24" fill="none"><path d="M5 12h14m-6-6 6 6-6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg></button></div></form>
            </div>

            {dataset && !simpleChatSession && <aside className="data-panel"><div className="data-panel-head"><div><div className="section-label">DATA OVERVIEW</div><p>Quick look at your file</p></div><span className="more-button">···</span></div><div className="data-file-card"><span className="file-badge">{dataset.file_type === "xlsx" ? "XLS" : "CSV"}</span><div><strong>{dataset.name.replace(/_[a-f0-9]{8}$/, "")}</strong><span>{dataset.rows.toLocaleString()} × {dataset.columns}</span></div></div><div className="preview-title">COLUMN PREVIEW <span>{dataset.columns} total</span></div><div className="column-list">{dataset.column_names.slice(0, 12).map((column, index) => <div className="column-item" key={column}><span className="column-type">{typeof dataset.preview?.[0]?.[column] === "number" ? "#" : "T"}</span><span>{column}</span><span className="column-index">{String(index + 1).padStart(2, "0")}</span></div>)}{dataset.columns > 12 && <div className="more-columns">+ {dataset.columns - 12} more columns</div>}</div><div className="preview-title table-title">SAMPLE ROWS <span>First {Math.min(dataset.preview.length, 8)}</span></div><div className="table-wrap"><table><thead><tr>{dataset.column_names.slice(0, 3).map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{dataset.preview.slice(0, 5).map((row, index) => <tr key={index}>{dataset.column_names.slice(0, 3).map((column) => <td key={column}>{String(row[column] ?? "—")}</td>)}</tr>)}</tbody></table></div><div className="data-panel-foot"><span className="ready-pip" /> File loaded to this workspace</div></aside>}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
