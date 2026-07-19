import { useEffect, useState } from "react";
import {
  ApiClientError,
  getFableWorkspace,
  pinFableRunner,
  runFableAction,
  type FableAction,
  type FableWorkspace,
} from "./api";

const liveStatuses = new Set(["queued", "running"]);

export default function FableWorkflowPanel() {
  const [workspace, setWorkspace] = useState<FableWorkspace | null>(null);
  const [version, setVersion] = useState("");
  const [runtimePath, setRuntimePath] = useState("");
  const [resultPath, setResultPath] = useState("");
  const [reviewPath, setReviewPath] = useState("");
  const [pinBusy, setPinBusy] = useState(false);
  const [editingPin, setEditingPin] = useState(false);
  const [actionBusy, setActionBusy] = useState<FableAction["id"] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = async (signal?: AbortSignal) => {
    const next = await getFableWorkspace(signal);
    setWorkspace(next);
    setError(null);
    return next;
  };

  useEffect(() => {
    const controller = new AbortController();
    refresh(controller.signal).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Fable workflow could not be inspected.");
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!workspace || !liveStatuses.has(workspace.job.status)) return;
    let stopped = false;
    const controller = new AbortController();
    let timer = 0;
    const poll = async () => {
      try {
        const next = await refresh(controller.signal);
        if (stopped) return;
        if (liveStatuses.has(next.job.status)) timer = window.setTimeout(() => void poll(), 2_000);
        else setActionBusy(null);
      } catch (reason) {
        if (!stopped && !controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Fable workflow status could not be refreshed.");
      }
    };
    timer = window.setTimeout(() => void poll(), 2_000);
    return () => {
      stopped = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [workspace?.job.id, workspace?.job.status]);

  const pin = async () => {
    if (!version.trim() || !runtimePath.trim()) return;
    setPinBusy(true);
    setError(null);
    try {
      setWorkspace(await pinFableRunner(version.trim(), runtimePath.trim()));
      setVersion("");
      setRuntimePath("");
      setEditingPin(false);
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "The Fable runtime could not be pinned.");
    } finally {
      setPinBusy(false);
    }
  };

  const run = async (action: FableAction) => {
    const inputPath = action.id === "ingest" ? resultPath.trim() : action.id === "review_import" ? reviewPath.trim() : undefined;
    if (action.input_required && !inputPath) return;
    setActionBusy(action.id);
    setError(null);
    try {
      const job = await runFableAction(action.id, inputPath);
      setWorkspace((current) => current ? { ...current, job } : current);
    } catch (reason) {
      setActionBusy(null);
      setError(reason instanceof ApiClientError ? reason.message : "The Fable workflow action could not start.");
    }
  };

  const jobLive = Boolean(workspace && liveStatuses.has(workspace.job.status));
  const exchange = workspace?.exchange;
  const runner = workspace?.runner;
  const job = workspace?.job;

  return (
    <article className="panel fable-workflow-panel" aria-labelledby="fable-workflow-heading">
      <header className="panel-header">
        <div><p className="panel-kicker">External comparison</p><h2 id="fable-workflow-heading">Fable A/B exchange</h2><p>AutoTrainer pins the supplied Fable bytes, exports verifier-free work, and trusts only patches re-scored by the trusted local verifier.</p></div>
        <span className={`status-chip ${workspace?.status === "report_ready" ? "good" : runner?.pinned ? "info" : "warning"}`}>{workspace?.status === "report_ready" ? "Report ready" : runner?.pinned ? "Runner pinned" : "Pin required"}</span>
      </header>

      {error && <div className="source-error" role="alert">{error}</div>}

      {runner?.pinned && runner.receipt_matches && !editingPin ? (
        <div className="fable-pin-summary">
          <div><span>Version</span><strong>{runner.version}</strong></div>
          <div><span>Runtime bytes</span><code>{runner.orchestration_sha256}</code></div>
          <div><span>Local pin</span><code>{runner.runtime_path}</code></div>
          <button className="secondary-button" type="button" onClick={() => { setVersion(runner.version || ""); setRuntimePath(runner.runtime_path || ""); setEditingPin(true); }} disabled={jobLive}>Change pin</button>
        </div>
      ) : (
        <form className="fable-pin-form" onSubmit={(event) => { event.preventDefault(); void pin(); }}>
          <label htmlFor="fable-version"><span>Fable version or immutable revision</span><input id="fable-version" value={version} onChange={(event) => setVersion(event.target.value)} placeholder="1.0.0 or commit identifier" disabled={pinBusy || jobLive} /></label>
          <label htmlFor="fable-runtime"><span>Runtime or orchestration bundle</span><input id="fable-runtime" value={runtimePath} onChange={(event) => setRuntimePath(event.target.value)} placeholder="C:\path\to\pinned-fable-bundle" disabled={pinBusy || jobLive} /></label>
          <button className="primary-button" type="submit" disabled={!version.trim() || !runtimePath.trim() || pinBusy || jobLive}>{pinBusy ? "Hashing..." : "Hash and pin"}</button>
        </form>
      )}

      {job && job.status !== "idle" && (
        <div className={`fable-job ${job.status}`} role={job.status === "failed" ? "alert" : "status"}><strong>{job.status === "failed" ? "Fable action stopped" : job.status === "completed" ? "Fable action completed" : "Fable action in progress"}</strong><p>{job.message}</p></div>
      )}

      {workspace && (
        <ol className="fable-steps">
          {workspace.actions.map((action, index) => {
            const inputValue = action.id === "ingest" ? resultPath : reviewPath;
            const setInput = action.id === "ingest" ? setResultPath : setReviewPath;
            return (
              <li key={action.id} className={action.status}>
                <span className="fable-step-number" aria-hidden="true">{index + 1}</span>
                <div><strong>{action.title}</strong><p>{action.detail}</p>{action.input_required && <input value={inputValue} onChange={(event) => setInput(event.target.value)} placeholder={action.id === "ingest" ? "Folder containing returned result envelopes" : "Reviewer choices JSONL file"} disabled={jobLive} />}</div>
                <button className={action.status === "available" ? "primary-button" : "secondary-button"} type="button" onClick={() => void run(action)} disabled={action.status !== "available" || jobLive || actionBusy !== null || (action.input_required && !inputValue.trim())}>{actionBusy === action.id ? "Starting..." : action.status === "complete" ? "Done" : action.status === "available" ? "Run" : "Waiting"}</button>
              </li>
            );
          })}
        </ol>
      )}

      {exchange && (
        <div className="fable-exchange-summary">
          <div><span>Frozen plan</span><code>{exchange.plan_id.slice(0, 16)}</code></div>
          <div><span>Locally scored</span><strong>{exchange.scored_count} / {exchange.trial_count}</strong></div>
          <div><span>Requests</span><code>{exchange.request_path || "Not exported"}</code></div>
          <div><span>Blind packet</span><code>{exchange.blind_review_path || "Not exported"}</code></div>
          <div><span>Report</span><code>{exchange.report_path || "Not built"}</code></div>
        </div>
      )}
    </article>
  );
}
