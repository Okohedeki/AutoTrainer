import { useEffect, useState } from "react";
import {
  ApiClientError,
  getLanguageEvaluation,
  setEvaluationLanguage,
  type LanguageEvaluationWorkspace,
} from "./api";

export default function LanguageEvaluationPanel() {
  const [workspace, setWorkspace] = useState<LanguageEvaluationWorkspace | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getLanguageEvaluation(controller.signal)
      .then(setWorkspace)
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Language evaluation could not be loaded.");
      });
    return () => controller.abort();
  }, []);

  const select = async (language: LanguageEvaluationWorkspace["configured"]) => {
    setBusy(true);
    setError(null);
    try {
      setWorkspace(await setEvaluationLanguage(language));
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "The evaluation language could not be saved.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <article className="panel language-evaluation-panel">
      <header className="panel-header">
        <div><p className="panel-kicker">Shipped evaluator</p><h2>Language-matched code proof</h2><p>The held-out evaluator must match the language that taught the adapter.</p></div>
        <span className={`status-chip ${workspace?.status === "ready" ? "good" : "warning"}`}>{workspace?.status || "loading"}</span>
      </header>

      {workspace && (
        <>
          <label className="evaluation-language-select"><span>Evaluation language</span><select value={workspace.configured} onChange={(event) => void select(event.target.value as LanguageEvaluationWorkspace["configured"])} disabled={busy}><option value="auto">Auto-detect from frozen dataset</option>{workspace.available.map((suite) => <option key={suite.id} value={suite.id}>{suite.label}</option>)}</select></label>
          {workspace.selected_suite && (
            <div className="language-suite-grid">
              <div><span>Selected suite</span><strong>{workspace.selected_suite.label}</strong><small>Training primary: {workspace.inferred_training_language || "not detected"}</small></div>
              <div><span>Trusted checks</span><strong>{workspace.selected_suite.checks.join(" / ")}</strong></div>
              <div><span>Metrics</span><strong>{workspace.selected_suite.metrics.join(" / ")}</strong></div>
              <div><span>Open benchmark inspiration</span><strong>{workspace.selected_suite.benchmark_inspirations.join(" / ")}</strong><small>Suite logic is shipped by AutoTrainer; benchmark names are design references.</small></div>
            </div>
          )}
          {workspace.blockers.length > 0 && <div className="evaluation-blocked"><ul>{workspace.blockers.map((blocker) => <li key={blocker}>{blocker}</li>)}</ul></div>}
        </>
      )}
      {error && <div className="source-error" role="alert">{error}</div>}
    </article>
  );
}
