import { useState } from "react";
import ActionGraph from "./ActionGraph.jsx";
import { sendAssistantMessage } from "./assistantApi.js";
import JsonView from "./JsonView.jsx";

function ArrowIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <path d="M5 12h14M13 6l6 6-6 6" />
    </svg>
  );
}

export default function App() {
  const [input, setInput] = useState("");
  const [response, setResponse] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [activeTab, setActiveTab] = useState("graph");

  async function handleSubmit(event) {
    event.preventDefault();
    const message = input.trim();

    if (!message || isSubmitting) return;

    setIsSubmitting(true);
    setActiveTab("graph");
    try {
      setResponse(await sendAssistantMessage(message));
      setInput("");
    } catch (error) {
      setResponse({
        error: error instanceof Error ? error.message : "Something went wrong.",
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="assistant-card" aria-label="Action Graph Retriever">
        <div className="eyebrow">
          <span className="status-dot" aria-hidden="true" />
          Action Graph Retriever
        </div>

        <div className="workspace-grid">
          <aside className="request-panel">
            <form className="message-form" onSubmit={handleSubmit}>
              <label htmlFor="assistant-input">Your request</label>
              <div className="input-row">
                <input
                  id="assistant-input"
                  type="text"
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder="Schedule a call with Alex next Tuesday"
                  autoComplete="off"
                  autoFocus
                />
                <button type="submit" disabled={!input.trim() || isSubmitting}>
                  <span>{isSubmitting ? "Sending" : "Send"}</span>
                  <ArrowIcon />
                </button>
              </div>
            </form>
          </aside>

          <div className={`output-panel ${response ? "has-response" : ""}`} aria-live="polite">
          {response ? (
            <>
              {response.error ? (
                <p className="error">{response.error}</p>
              ) : (
                <>
                  <div className="result-tabs" role="tablist" aria-label="Result view">
                    <button
                      className="tab-button"
                      id="graph-tab"
                      type="button"
                      role="tab"
                      aria-controls="graph-panel"
                      aria-selected={activeTab === "graph"}
                      onClick={() => setActiveTab("graph")}
                    >
                      Graph
                    </button>
                    <button
                      className="tab-button"
                      id="json-tab"
                      type="button"
                      role="tab"
                      aria-controls="json-panel"
                      aria-selected={activeTab === "json"}
                      onClick={() => setActiveTab("json")}
                    >
                      JSON
                    </button>
                  </div>

                  <div
                    id="graph-panel"
                    role="tabpanel"
                    aria-labelledby="graph-tab"
                    hidden={activeTab !== "graph"}
                  >
                    <ActionGraph graph={response} />
                  </div>
                  <div
                    id="json-panel"
                    role="tabpanel"
                    aria-labelledby="json-tab"
                    hidden={activeTab !== "json"}
                  >
                    <JsonView data={response} />
                  </div>
                </>
              )}
            </>
          ) : (
            <p className="empty-state">The action graph will appear here.</p>
          )}
          </div>
        </div>
      </section>

      <p className="privacy-note">Connected to the local intent-recognition service.</p>
    </main>
  );
}
