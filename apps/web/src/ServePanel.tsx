import { useEffect, useMemo, useRef, useState } from "react";
import {
  ApiClientError,
  getHostingStatus,
  startHosting,
  stopHosting,
  testHosting,
  type HostingStatus,
} from "./api";

function statusTone(status?: string) {
  if (status === "live") return "good";
  if (status === "loading") return "info";
  if (status === "failed" || status === "not_ready") return "danger";
  return "muted";
}

function joinEndpoint(endpoint: string) {
  const clean = endpoint.replace(/\/$/, "");
  return clean.endsWith("/v1") ? `${clean}/chat/completions` : `${clean}/v1/chat/completions`;
}

function adapterLabel(adapter?: string | null) {
  if (adapter === "grpo") return "Practice adapter";
  if (adapter === "sft") return "Teaching adapter";
  if (adapter === "base") return "Base model (no adapter)";
  return adapter || "Resolved when the host starts";
}

export default function ServePanel() {
  const adapterChoiceTouched = useRef(false);
  const [hosting, setHosting] = useState<HostingStatus | null>(null);
  const [adapter, setAdapter] = useState("auto");
  const [prompt, setPrompt] = useState("Explain the smallest safe change you would make to this project.");
  const [response, setResponse] = useState<string | null>(null);
  const [action, setAction] = useState<"start" | "stop" | "test" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  useEffect(() => {
    let stopped = false;
    const controller = new AbortController();
    let timer = 0;
    const poll = async () => {
      try {
        const next = await getHostingStatus(controller.signal);
        if (stopped) return;
        setHosting(next);
        // Live/loading state is authoritative, including hosts started by the
        // CLI. Before start, preserve a deliberate GUI choice after the first
        // backend hydration instead of resetting it on every poll.
        if (next.adapter && (!adapterChoiceTouched.current || ["loading", "live"].includes(next.status))) {
          setAdapter(next.adapter);
        }
        setError(null);
      } catch (reason) {
        if (!stopped && !controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Local serving status could not be refreshed.");
      } finally {
        if (!stopped) timer = window.setTimeout(() => void poll(), 2_000);
      }
    };
    void poll();
    return () => {
      stopped = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, []);

  const endpoint = hosting?.endpoint || "";
  const chatEndpoint = endpoint ? joinEndpoint(endpoint) : "";
  const curl = useMemo(() => chatEndpoint ? `curl ${chatEndpoint} -H "Content-Type: application/json" -d '{"model":"${hosting?.model || "autotrainer-local"}","messages":[{"role":"user","content":"Hello"}]}'` : "", [chatEndpoint, hosting?.model]);
  const live = hosting?.status === "live";
  const loading = hosting?.status === "loading";

  const start = async () => {
    setAction("start");
    setError(null);
    try {
      const next = await startHosting(adapter.trim());
      setHosting(next);
      if (next.adapter) setAdapter(next.adapter);
      adapterChoiceTouched.current = false;
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "The local model host could not start.");
    } finally {
      setAction(null);
    }
  };

  const stop = async () => {
    setAction("stop");
    setError(null);
    try {
      const next = await stopHosting();
      setHosting(next);
      if (next.adapter) setAdapter(next.adapter);
      adapterChoiceTouched.current = false;
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "The local model host could not stop.");
    } finally {
      setAction(null);
    }
  };

  const test = async () => {
    setAction("test");
    setError(null);
    setResponse(null);
    try {
      const result = await testHosting(prompt.trim());
      const text = result.response ?? result.content ?? result.text;
      setResponse(typeof text === "string" ? text : JSON.stringify(result, null, 2));
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "The local test request failed.");
    } finally {
      setAction(null);
    }
  };

  const copy = async (id: string, value: string) => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(id);
      window.setTimeout(() => setCopied(null), 1_500);
    } catch {
      setError("Copy was blocked. Select the text manually.");
    }
  };

  return (
    <section className="serve-workspace" aria-labelledby="serve-heading" data-tour="serve">
      {error && <div className="source-error serve-error" role="alert">{error}</div>}
      <article className="panel serve-command-panel">
        <header className="panel-header"><div><p className="panel-kicker">Post-training</p><h2 id="serve-heading">Serve the specialist locally</h2></div><span className={`status-chip ${statusTone(hosting?.status)}`}>{hosting?.status?.replaceAll("_", " ") || "connecting"}</span></header>
        <p className="serve-dream">Your specialist stays on this machine. Start one OpenAI-compatible endpoint and call it from an app, an agent, or a terminal.</p>
        <div className="serve-controls">
          <label htmlFor="adapter-choice"><span>Model output</span><select id="adapter-choice" value={adapter} onChange={(event) => { adapterChoiceTouched.current = true; setAdapter(event.target.value); }} disabled={live || loading || action !== null}><option value="auto">Best completed adapter</option><option value="grpo">Practice adapter</option><option value="sft">Teaching adapter</option><option value="base">Base model preview</option></select></label>
          {!live ? <button className="primary-button" type="button" onClick={() => void start()} disabled={loading || action !== null}>{action === "start" || loading ? "Loading model..." : "Start local endpoint"}</button> : <button className="secondary-button danger-button" type="button" onClick={() => void stop()} disabled={action !== null}>{action === "stop" ? "Stopping..." : "Stop endpoint"}</button>}
        </div>
        {hosting?.adapter && <p className="field-note">Backend resolved output: <strong>{adapterLabel(hosting.adapter)}</strong>.</p>}
        {hosting?.message && <p className="hosting-message" role="status">{hosting.message}</p>}
      </article>

      <div className="serve-grid">
        <article className="panel endpoint-panel">
          <header className="panel-header"><div><p className="panel-kicker">Callable interface</p><h2>Endpoint</h2></div>{live && <span className="status-chip good">live</span>}</header>
          {!live || !endpoint ? <div className="evidence-empty"><strong>No endpoint running</strong><p>Choose the best completed output and start the local host. AutoTrainer will show only an endpoint it can reach.</p></div> : (
            <div className="endpoint-details">
              <dl><div><dt>Chat completions</dt><dd>{chatEndpoint}</dd></div><div><dt>Model</dt><dd>{hosting.model || hosting.base_model || "autotrainer-local"}</dd></div><div><dt>Loaded output</dt><dd>{adapterLabel(hosting.adapter)}</dd></div><div><dt>Base revision</dt><dd>{hosting.revision || "Recorded by the host"}</dd></div></dl>
              <div className="copy-block"><code>{chatEndpoint}</code><button type="button" onClick={() => void copy("endpoint", chatEndpoint)}>{copied === "endpoint" ? "Copied" : "Copy endpoint"}</button></div>
              <div className="copy-block curl-block"><code>{curl}</code><button type="button" onClick={() => void copy("curl", curl)}>{copied === "curl" ? "Copied" : "Copy curl"}</button></div>
              <p className="field-note">Text only. One bounded request at a time. Training and evaluation must release the GPU first.</p>
            </div>
          )}
        </article>

        <article className="panel host-test-panel">
          <header className="panel-header"><div><p className="panel-kicker">Live request</p><h2>Test the endpoint</h2></div></header>
          <label htmlFor="host-test-prompt"><span>Prompt</span><textarea id="host-test-prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} disabled={!live || action !== null} /></label>
          <button className="primary-button" type="button" onClick={() => void test()} disabled={!live || !prompt.trim() || action !== null}>{action === "test" ? "Calling local model..." : "Send test request"}</button>
          {response ? <div className="host-response" aria-live="polite"><strong>Local response</strong><pre>{response}</pre></div> : <div className="evidence-empty compact"><strong>No response yet</strong><p>A real response appears here after the local endpoint answers.</p></div>}
        </article>
      </div>
    </section>
  );
}
