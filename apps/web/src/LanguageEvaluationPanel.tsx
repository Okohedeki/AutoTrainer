import { useEffect, useState } from "react";
import {
  ApiClientError,
  getEvaluationPacks,
  getLanguageEvaluation,
  installEvaluationPack,
  setEvaluationLanguage,
  type EvaluationPackWorkspace,
  type LanguageEvaluationWorkspace,
} from "./api";

export default function LanguageEvaluationPanel() {
  const [workspace, setWorkspace] = useState<LanguageEvaluationWorkspace | null>(null);
  const [packWorkspace, setPackWorkspace] = useState<EvaluationPackWorkspace | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      getLanguageEvaluation(controller.signal),
      getEvaluationPacks(controller.signal),
    ])
      .then(([language, packs]) => {
        setWorkspace(language);
        setPackWorkspace(packs);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Language evaluation could not be loaded.");
      });
    return () => controller.abort();
  }, []);

  const select = async (language: LanguageEvaluationWorkspace["configured"]) => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      setWorkspace(await setEvaluationLanguage(language));
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "The evaluation language could not be saved.");
    } finally {
      setBusy(false);
    }
  };

  const installPack = async (packId: string) => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await installEvaluationPack(packId);
      const [language, packs] = await Promise.all([
        getLanguageEvaluation(),
        getEvaluationPacks(),
      ]);
      setWorkspace(language);
      setPackWorkspace(packs);
      setNotice("Installed locally. Next: go to Data and choose Lock new version so this held-out pack is included in the immutable evaluation inputs.");
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "The local benchmark pack could not be installed.");
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
      {packWorkspace && (
        <section className="evaluation-pack-section" aria-labelledby="evaluation-pack-heading">
          <header>
            <div><p className="panel-kicker">Ready-made held-out tasks</p><h3 id="evaluation-pack-heading">Local benchmark packs</h3></div>
            <p>Install a shipped, reproducible pack when you do not already have private evaluation tasks. Manual task authoring stays in the advanced section of Data.</p>
          </header>
          <div className="evaluation-pack-list">
            {packWorkspace.packs.map((pack) => (
              <article className={`evaluation-pack-card ${pack.selected ? "selected" : ""}`} key={pack.id}>
                <header>
                  <div><h4>{pack.label}</h4><p>{pack.description}</p></div>
                  <span className={`status-chip ${pack.installed ? "good" : "muted"}`}>{pack.status}</span>
                </header>
                <dl>
                  <div><dt>Language</dt><dd>{pack.language === "python" ? "Python" : pack.language}</dd></div>
                  <div><dt>Held-out coverage</dt><dd>{pack.task_count} tasks · {pack.independent_group_count} independent groups</dd></div>
                  <div><dt>License</dt><dd>{pack.license}</dd></div>
                  <div><dt>Checks</dt><dd>{pack.checks.join(" / ")}</dd></div>
                  <div className="evaluation-pack-runtime"><dt>Pinned runtime</dt><dd><code>{pack.runtime_image}</code></dd></div>
                </dl>
                <button className={pack.selected ? "secondary-button" : "primary-button"} type="button" disabled={busy || pack.selected} onClick={() => void installPack(pack.id)}>
                  {pack.selected ? "Installed and in use" : pack.installed ? "Use pack" : "Install and use"}
                </button>
              </article>
            ))}
          </div>
          {packWorkspace.packs.length === 0 && <p className="evaluation-pack-empty">No local benchmark packs are shipped for this build.</p>}
          {notice && <div className="evaluation-pack-notice" role="status">{notice}</div>}
        </section>
      )}
      {error && <div className="source-error" role="alert">{error}</div>}
    </article>
  );
}
