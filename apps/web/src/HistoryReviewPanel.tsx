import { useEffect, useState } from "react";
import {
  ApiClientError,
  getReviewHistory,
  reviewHistoryCandidate,
  type HistoryWorkspace,
} from "./api";

// History is optional input, so an empty or unavailable history endpoint does
// not add another setup card. The panel appears only when there is real work to
// review or an approved example worth acknowledging.
export default function HistoryReviewPanel({ refreshKey = 0 }: { refreshKey?: number }) {
  const [workspace, setWorkspace] = useState<HistoryWorkspace | null>(null);
  const [instruction, setInstruction] = useState("");
  const [rightsConfirmed, setRightsConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const candidate = workspace?.candidates[0] || null;

  useEffect(() => {
    const controller = new AbortController();
    getReviewHistory(controller.signal)
      .then((next) => {
        setWorkspace(next);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        // Preparation reports required blockers. Optional history stays absent
        // instead of duplicating the page's connection errors.
        setError(reason instanceof Error ? reason.message : "History could not be loaded.");
      });
    return () => controller.abort();
  }, [refreshKey]);

  useEffect(() => {
    setInstruction(candidate?.proposed_instruction || "");
    setRightsConfirmed(false);
  }, [candidate?.candidate_id, candidate?.proposed_instruction]);

  if (!workspace || (!candidate && workspace.summary.approved_count === 0)) return null;

  const review = async (decision: "approved" | "rejected") => {
    if (!candidate) return;
    setBusy(true);
    setError(null);
    try {
      // Skip is persisted as a rejection rather than moving a local cursor, so
      // the same candidate does not return on the next page load.
      const next = await reviewHistoryCandidate(
        decision === "approved"
          ? {
              candidate_id: candidate.candidate_id,
              decision,
              instruction: instruction.trim(),
              rights_confirmed: rightsConfirmed,
            }
          : { candidate_id: candidate.candidate_id, decision },
      );
      setWorkspace(next);
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "That review could not be saved.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel history-review" aria-labelledby="history-heading">
      <header className="history-heading">
        <div>
          <p className="panel-kicker">From your repositories</p>
          <h2 id="history-heading">Review accepted work</h2>
          <p>Turn a real change into one clear example for the model.</p>
        </div>
        <span className="status-chip muted">
          {candidate ? `${workspace.candidates.length} to review` : `${workspace.summary.approved_count} approved`}
        </span>
      </header>

      {!candidate ? (
        <p className="history-complete">
          {workspace.summary.approved_count} approved {workspace.summary.approved_count === 1 ? "change is" : "changes are"} ready as training examples.
        </p>
      ) : (
        <div className="history-candidate">
          <label className="history-instruction" htmlFor="history-instruction">
            <span>What was this change meant to do?</span>
            <textarea
              id="history-instruction"
              value={instruction}
              onChange={(event) => setInstruction(event.target.value)}
              disabled={busy}
              rows={3}
            />
          </label>

          <div className="history-files" aria-label="Changed files">
            {candidate.files.map((file) => (
              <div key={file.path}>
                <code>{file.path}</code>
                <span>{file.status}</span>
                <span className="history-lines"><b>+{file.additions}</b> <i>−{file.deletions}</i></span>
              </div>
            ))}
          </div>

          {candidate.flags.length > 0 && (
            <ul className="history-flags" aria-label="Review notes">
              {candidate.flags.map((flag) => <li key={flag}>{flag}</li>)}
            </ul>
          )}

          <pre className="history-diff" aria-label="Change diff"><code>{candidate.patch}</code></pre>

          <label className="rights-check">
            <input
              type="checkbox"
              checked={rightsConfirmed}
              onChange={(event) => setRightsConfirmed(event.target.checked)}
              disabled={busy}
            />
            <span>I have the right to use this change for training.</span>
          </label>

          {error && <div className="source-error history-error" role="alert">{error}</div>}

          <div className="history-actions">
            <button className="text-button" type="button" onClick={() => void review("rejected")} disabled={busy}>
              Skip
            </button>
            <button
              className="primary-button"
              type="button"
              onClick={() => void review("approved")}
              disabled={busy || !rightsConfirmed || !instruction.trim()}
            >
              {busy ? "Saving…" : "Approve example"}
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
