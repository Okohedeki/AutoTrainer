import { useEffect, useState } from "react";
import {
  ApiClientError,
  designDatasetCandidate,
  freezeDataset,
  getDatasetWorkspace,
  retireStaleHistoryReviews,
  reviewHistoryCandidate,
  syncDatasetSources,
  type DatasetWorkspace,
} from "./api";

const languageLabels: Record<string, string> = {
  python: "Python",
  typescript_react: "TypeScript / React",
  csharp: "C#",
  cpp: "C++",
};

export default function DatasetWorkspacePanel({
  refreshKey = 0,
  onDatasetChanged,
  disabled = false,
}: {
  refreshKey?: number;
  onDatasetChanged?: () => void;
  disabled?: boolean;
}) {
  const [workspace, setWorkspace] = useState<DatasetWorkspace | null>(null);
  const [instruction, setInstruction] = useState("");
  const [rightsConfirmed, setRightsConfirmed] = useState(false);
  const [provider, setProvider] = useState<"local" | "anthropic">("local");
  const [model, setModel] = useState("qwen-local");
  const [endpoint, setEndpoint] = useState("http://127.0.0.1:8000/v1/chat/completions");
  const [busy, setBusy] = useState<"sync" | "design" | "review" | "freeze" | "retire" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const candidate = workspace?.candidates.find((item) => item.decision === "pending") || null;

  useEffect(() => {
    const controller = new AbortController();
    getDatasetWorkspace(controller.signal)
      .then((next) => {
        setWorkspace(next);
        setEndpoint(next.designers.local.default_endpoint);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : "The dataset workspace could not be loaded.");
        }
      });
    return () => controller.abort();
  }, [refreshKey]);

  useEffect(() => {
    setInstruction(candidate?.design?.instruction || candidate?.proposed_instruction || "");
    setRightsConfirmed(false);
  }, [candidate?.candidate_id, candidate?.design?.instruction, candidate?.proposed_instruction]);

  const run = async (
    action: NonNullable<typeof busy>,
    operation: () => Promise<DatasetWorkspace>,
  ) => {
    setBusy(action);
    setError(null);
    try {
      const next = await operation();
      setWorkspace(next);
      onDatasetChanged?.();
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "The dataset operation failed.");
    } finally {
      setBusy(null);
    }
  };

  const design = () => {
    if (!candidate || !model.trim()) return;
    void run("design", () => designDatasetCandidate({
      candidate_id: candidate.candidate_id,
      provider,
      model: model.trim(),
      ...(provider === "local" ? { endpoint: endpoint.trim() } : {}),
    }));
  };

  const review = (decision: "approved" | "rejected") => {
    if (!candidate) return;
    setBusy("review");
    setError(null);
    reviewHistoryCandidate(
      decision === "approved"
        ? {
            candidate_id: candidate.candidate_id,
            decision,
            instruction: instruction.trim(),
            rights_confirmed: rightsConfirmed,
          }
        : { candidate_id: candidate.candidate_id, decision },
    )
      .then(() => getDatasetWorkspace())
      .then((next) => {
        setWorkspace(next);
        onDatasetChanged?.();
      })
      .catch((reason: unknown) => {
        setError(reason instanceof ApiClientError ? reason.message : "That review could not be saved.");
      })
      .finally(() => setBusy(null));
  };

  const retire = () => {
    setBusy("retire");
    setError(null);
    retireStaleHistoryReviews()
      .then(() => getDatasetWorkspace())
      .then((next) => {
        setWorkspace(next);
        onDatasetChanged?.();
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "Old approvals could not be retired."))
      .finally(() => setBusy(null));
  };

  const frozen = workspace?.freeze.status === "ready";
  const canFreeze = Boolean(
    workspace
    && workspace.catalog.status !== "needs_sync"
    && workspace.summary.pending_count === 0
    && workspace.summary.stale_review_count === 0,
  );

  return (
    <section className="panel dataset-workspace" aria-labelledby="dataset-heading">
      <header className="dataset-heading">
        <div>
          <p className="panel-kicker">Review before training</p>
          <h2 id="dataset-heading">Build the local training dataset</h2>
          <p>AutoTrainer imports licensed PRs merged into main or master, asks your chosen LLM to structure each one, and waits for your approval before locking anything for training.</p>
        </div>
        <span className={`status-chip ${frozen ? "ready" : "muted"}`}>
          {frozen ? "Frozen" : workspace?.freeze.status === "stale" ? "Changed" : "Draft"}
        </span>
      </header>

      <ol className="dataset-flow" aria-label="Dataset workflow">
        <li><span>1</span><div><strong>Import</strong><small>Find merged PRs</small></div></li>
        <li><span>2</span><div><strong>Analyze</strong><small>Use local or Claude</small></div></li>
        <li><span>3</span><div><strong>Review</strong><small>Approve each record</small></div></li>
        <li><span>4</span><div><strong>Lock</strong><small>Make it trainable</small></div></li>
      </ol>

      <div className="dataset-policy" aria-label="Dataset policy">
        <div><span>Quality rule</span><strong>Merged PR</strong></div>
        <div><span>Branches</span><strong>main / master</strong></div>
        <div><span>License</span><strong>Required</strong></div>
        <div><span>Storage</span><strong>Local only</strong></div>
      </div>

      {!workspace && !error && <p className="panel-empty">Loading the local dataset catalog…</p>}

      {workspace && (
        <>
          <div className="dataset-catalog-row">
            <div>
              <strong>
                {workspace.catalog.source_count > 0
                  ? `${workspace.catalog.merged_pull_request_count} merged PR${workspace.catalog.merged_pull_request_count === 1 ? "" : "s"}`
                  : "No GitHub accepted-change source"}
              </strong>
              <p>
                {workspace.catalog.status === "needs_sync"
                  ? "Import merged pull requests to create the review queue."
                  : workspace.catalog.source_count > 0
                    ? `${workspace.catalog.source_count} licensed allowlisted ${workspace.catalog.source_count === 1 ? "repository" : "repositories"}.`
                    : "Local examples and executable tasks can still be inspected and frozen."}
              </p>
            </div>
            {workspace.catalog.source_count > 0 && (
              <button className="secondary-button" type="button" onClick={() => void run("sync", syncDatasetSources)} disabled={Boolean(busy) || disabled}>
                {busy === "sync" ? "Importing…" : workspace.catalog.status === "needs_sync" ? "Import merged PRs" : "Refresh merged PRs"}
              </button>
            )}
          </div>

          {Object.keys(workspace.summary.language_counts).length > 0 && (
            <div className="dataset-languages" aria-label="Approved dataset languages">
              {Object.entries(workspace.summary.language_counts).map(([language, count]) => (
                <span key={language}>{languageLabels[language] || language}: {count}</span>
              ))}
            </div>
          )}

          {workspace.summary.stale_review_count > 0 && (
            <div className="history-stale" role="alert">
              <div><strong>Approved data changed</strong><p>Retire stale authority before freezing a new dataset.</p></div>
              <button className="secondary-button" type="button" onClick={retire} disabled={Boolean(busy) || disabled}>{busy === "retire" ? "Retiring…" : "Retire old approval"}</button>
            </div>
          )}

          {candidate && (
            <div className="history-candidate dataset-candidate">
              <div className="dataset-pr-title">
                <div>
                  <span>{candidate.pull_request ? `PR #${candidate.pull_request.number} · merged to ${candidate.pull_request.base_branch}` : "Reviewed local commit"}</span>
                  <strong>{candidate.pull_request?.title || "Accepted repository change"}</strong>
                </div>
                <span>{workspace.summary.pending_count} remaining</span>
              </div>

              <div className="dataset-designer">
                <label><span>Analyze with</span><select value={provider} onChange={(event) => {
                  const next = event.target.value as "local" | "anthropic";
                  setProvider(next);
                  setModel(next === "anthropic" ? "claude-haiku-4-5" : "qwen-local");
                }} disabled={Boolean(busy) || disabled}><option value="local">Local model</option><option value="anthropic">Anthropic Claude</option></select></label>
                <label><span>LLM</span><input value={model} onChange={(event) => setModel(event.target.value)} disabled={Boolean(busy) || disabled} /></label>
                {provider === "local" && <label className="dataset-endpoint"><span>Loopback endpoint</span><input value={endpoint} onChange={(event) => setEndpoint(event.target.value)} disabled={Boolean(busy) || disabled} /></label>}
                <button className="secondary-button" type="button" onClick={design} disabled={Boolean(busy) || disabled || !model.trim()}>{busy === "design" ? "Analyzing…" : "Analyze this PR"}</button>
              </div>

              {candidate.design && (
                <div className="dataset-recommendation">
                  <strong>{candidate.design.recommended_method === "qlora" ? "QLoRA example" : "GRPO task"} · {languageLabels[candidate.design.language]}</strong>
                  <p>{candidate.design.reason}</p>
                  {candidate.design.grpo_task && <p><b>Verifier focus:</b> {candidate.design.grpo_task.verifier_focus}</p>}
                  {candidate.design.recommended_method === "grpo" && <p><b>Recommendation:</b> this change is better suited to verified practice than imitation. Reject it as a QLoRA example unless you intentionally want the accepted patch in supervised training. Expert users can create the suggested task under manual data tools above.</p>}
                </div>
              )}

              <label className="history-instruction" htmlFor="dataset-instruction">
                <span>Training instruction</span>
                <textarea id="dataset-instruction" value={instruction} onChange={(event) => setInstruction(event.target.value)} disabled={Boolean(busy) || disabled} rows={3} />
              </label>

              <div className="history-files" aria-label="Changed files">
                {candidate.files.map((file) => <div key={file.path}><code>{file.path}</code><span>{file.status}</span><span className="history-lines"><b>+{file.additions}</b> <i>−{file.deletions}</i></span></div>)}
              </div>
              {candidate.flags.length > 0 && <ul className="history-flags" aria-label="Review notes">{candidate.flags.map((flag) => <li key={flag}>{flag}</li>)}</ul>}
              <pre className="history-diff" aria-label="Accepted patch"><code>{candidate.patch}</code></pre>
              <label className="rights-check"><input type="checkbox" checked={rightsConfirmed} onChange={(event) => setRightsConfirmed(event.target.checked)} disabled={Boolean(busy) || disabled} /><span>I have the right to use this change for local refinement.</span></label>
              <div className="history-actions">
                <button className="text-button" type="button" onClick={() => review("rejected")} disabled={Boolean(busy) || disabled}>Reject</button>
                <button className="primary-button" type="button" onClick={() => review("approved")} disabled={Boolean(busy) || disabled || !rightsConfirmed || !instruction.trim()}>{busy === "review" ? "Saving…" : candidate.design?.recommended_method === "grpo" ? "Add to QLoRA anyway" : "Add to QLoRA dataset"}</button>
              </div>
            </div>
          )}

          {!candidate && workspace.summary.reviewable_count > 0 && <p className="history-complete">All {workspace.summary.reviewable_count} candidates have a decision.</p>}

          <div className="dataset-freeze">
            <div>
              <strong>{frozen ? "Dataset is locked for training" : workspace.freeze.status === "stale" ? "Review changes before locking again" : "Lock the reviewed dataset"}</strong>
              <p>{frozen ? `${workspace.freeze.receipt?.counts.sft_train || 0} QLoRA examples and ${workspace.freeze.receipt?.counts.rl_train || 0} GRPO tasks are locked to this version.` : "Locking writes the local training files and a reproducible fingerprint. Changing a source or decision unlocks it again."}</p>
              {workspace.freeze.receipt?.compiler_fingerprint && <code>{workspace.freeze.receipt.compiler_fingerprint}</code>}
            </div>
            <button className="primary-button" type="button" onClick={() => void run("freeze", freezeDataset)} disabled={Boolean(busy) || disabled || !canFreeze}>{busy === "freeze" ? "Locking…" : frozen ? "Lock new version" : "Lock dataset for training"}</button>
          </div>
        </>
      )}

      {(error || (workspace?.errors.length ?? 0) > 0) && <div className="source-error history-error" role="alert">{error || workspace?.errors[0]}</div>}
    </section>
  );
}
