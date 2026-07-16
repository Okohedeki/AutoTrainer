import { useEffect, useMemo, useRef, useState } from "react";
import {
  ApiClientError,
  getEvaluationWorkspace,
  planEvaluation,
  startEvaluation,
  type EvaluationResult,
  type EvaluationSuite,
  type EvaluationWorkspace,
} from "./api";

const liveStatuses = new Set(["queued", "running"]);

function phaseLabel(value: string) {
  return value.replaceAll("_", " ");
}

function suiteTone(suite: EvaluationSuite | undefined) {
  if (!suite) return "muted";
  if (["reported", "ready_to_report", "completed"].includes(suite.phase)) return "good";
  if (["paused", "awaiting_external_results", "awaiting_blind_reviews", "ready_for_blind_review"].includes(suite.phase)) return "warning";
  if (["generating", "verifying", "queued", "running"].includes(suite.phase)) return "info";
  return "muted";
}

function ResultRows({ results }: { results: EvaluationResult[] }) {
  if (results.length === 0) {
    return <div className="evaluation-empty"><strong>No scored trials yet</strong><p>Rows appear only after a patch passes through the trusted local verifier.</p></div>;
  }
  return (
    <div className="evaluation-table-scroll">
      <table className="evaluation-table">
        <thead><tr><th>Task</th><th>Arm</th><th>Repeat</th><th>Gate</th><th>Verified reward</th></tr></thead>
        <tbody>
          {results.slice(-12).reverse().map((result) => (
            <tr key={result.trial_id}>
              <td><strong>{result.task_id}</strong></td>
              <td><code>{result.arm_id}</code></td>
              <td>{result.repetition + 1}</td>
              <td><span className={`status-chip ${result.hard_gate_passed ? "good" : "danger"}`}>{result.hard_gate_passed ? "pass" : "fail"}</span></td>
              <td>
                <div className="reward-cell">
                  <span className="reward-track" aria-hidden="true"><i style={{ width: `${Math.max(0, Math.min(1, result.reward)) * 100}%` }} /></span>
                  <code>{result.reward.toFixed(3)}</code>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Evaluation is observable at real boundaries: frozen plan, current immutable
// trial, trusted verification, scored result, and external handoff. Opaque
// model generation never becomes a made-up token counter or ETA.
export default function EvaluationMonitorPanel({ onOpenSetup }: { onOpenSetup: () => void }) {
  const requestVersion = useRef(0);
  const [workspace, setWorkspace] = useState<EvaluationWorkspace | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [action, setAction] = useState<"plan" | "start" | null>(null);

  const refresh = async (signal?: AbortSignal) => {
    const version = ++requestVersion.current;
    try {
      const next = await getEvaluationWorkspace(signal);
      if (version !== requestVersion.current) return;
      setWorkspace(next);
      setError(null);
    } catch (reason) {
      if (signal?.aborted) return;
      setError(reason instanceof Error ? reason.message : "Evaluation status could not be refreshed.");
    }
  };

  useEffect(() => {
    const controller = new AbortController();
    let stopped = false;
    let timer = 0;
    const poll = async () => {
      await refresh(controller.signal);
      // Schedule after completion so older workspace responses cannot overtake
      // a newer one while a slow local verifier has the filesystem busy.
      if (!stopped) timer = window.setTimeout(() => void poll(), 2_000);
    };
    void poll();
    return () => {
      stopped = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, []);

  const benchmark = useMemo(
    () => workspace?.suites.find((suite) => suite.id === "model_benchmark"),
    [workspace],
  );
  const fable = useMemo(
    () => workspace?.suites.find((suite) => suite.id === "fable_ab"),
    [workspace],
  );
  const jobLive = liveStatuses.has(workspace?.job.status ?? "idle");
  const benchmarkProgress = benchmark?.total ? benchmark.completed / benchmark.total : 0;
  const canPlan = workspace?.readiness.status === "ready" && !jobLive;
  const canStart = Boolean(
    workspace?.plan
    && benchmark
    && benchmark.runner_type === "command"
    && benchmark.completed < benchmark.total
    && !jobLive,
  );

  const freezePlan = async () => {
    setAction("plan");
    setError(null);
    requestVersion.current += 1;
    try {
      const next = await planEvaluation();
      requestVersion.current += 1;
      setWorkspace(next);
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "The evaluation plan could not be frozen.");
    } finally {
      setAction(null);
    }
  };

  const startBenchmark = async () => {
    setAction("start");
    setError(null);
    requestVersion.current += 1;
    try {
      await startEvaluation("model_benchmark");
      await refresh();
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "The local benchmark could not start.");
    } finally {
      setAction(null);
    }
  };

  return (
    <section className="evaluation-workspace" aria-label="Evaluation workspace">
      {error && <div className="source-error evaluation-error" role="alert">{error}</div>}

      <div className="evaluation-summary">
        <article className="panel"><span>Held-out tasks</span><strong>{workspace?.readiness.ready_task_count ?? "—"}</strong><small>From the prepared project</small></article>
        <article className="panel"><span>Frozen plan</span><strong>{workspace?.plan ? workspace.plan.plan_id.slice(7, 19) : "Not frozen"}</strong><small>{workspace?.plan ? `${workspace.plan.repetitions} repetitions` : "Prepare, then freeze"}</small></article>
        <article className="panel"><span>Local benchmark</span><strong>{benchmark ? `${benchmark.completed} / ${benchmark.total}` : "—"}</strong><small>Trusted scored trials</small></article>
        <article className="panel"><span>External comparison</span><strong>{fable ? `${fable.completed} / ${fable.total}` : "—"}</strong><small>Verified results ingested</small></article>
      </div>

      {!workspace?.plan && (
        <article className="panel evaluation-plan-panel">
          <header className="panel-header">
            <div><p className="panel-kicker">Step 1</p><h2 id="evaluation-workspace-heading">Freeze the proof</h2></div>
            <span className={`status-chip ${canPlan ? "good" : "warning"}`}>{workspace?.readiness.status ?? "connecting"}</span>
          </header>
          <p className="evaluation-lead">Lock the held-out tasks, model arms, repetitions, seeds, and runner identities before any result exists.</p>
          {(workspace?.readiness.blockers.length ?? 0) > 0 && (
            <ul className="blocker-list">
              {workspace?.readiness.blockers.map((blocker) => <li key={blocker}>{blocker}</li>)}
            </ul>
          )}
          <div className="evaluation-actions">
            {!canPlan && <button className="secondary-button" type="button" onClick={onOpenSetup}>Open setup</button>}
            <button className="primary-button" type="button" onClick={() => void freezePlan()} disabled={!canPlan || action !== null}>{action === "plan" ? "Freezing..." : "Freeze evaluation plan"}</button>
          </div>
        </article>
      )}

      {workspace?.plan && benchmark && (
        <div className="evaluation-grid">
          <article className="panel evaluation-run-panel">
            <header className="panel-header">
              <div><p className="panel-kicker">Local proof</p><h2 id="evaluation-workspace-heading">Model benchmark</h2></div>
              <span className={`status-chip ${suiteTone(benchmark)}`}>{phaseLabel(benchmark.phase)}</span>
            </header>
            <p className="evaluation-lead">{benchmark.message}</p>

            <div className="observed-progress" aria-label={`${benchmark.completed} of ${benchmark.total} trusted trials complete`}>
              <div><strong>{benchmark.completed} / {benchmark.total}</strong><span>trusted trials complete</span></div>
              <span aria-hidden="true"><i style={{ width: `${benchmarkProgress * 100}%` }} /></span>
            </div>

            {jobLive && workspace.job.suite === benchmark.id && workspace.job.current_trial && (
              <div className="current-trial" role="status">
                <span className="pulse-marker" aria-hidden="true" />
                <div><strong>{phaseLabel(workspace.job.phase)}</strong><p>{workspace.job.message}</p></div>
                <dl>
                  <div><dt>Task</dt><dd>{workspace.job.current_trial.task_id}</dd></div>
                  <div><dt>Arm</dt><dd>{workspace.job.current_trial.arm_id}</dd></div>
                  <div><dt>Seed</dt><dd>{workspace.job.current_trial.seed}</dd></div>
                </dl>
              </div>
            )}

            <div className="evaluation-run-actions">
              <span>{jobLive ? "The project is locked while this trial runs." : "Completed immutable trials are kept when you resume."}</span>
              {canStart && <button className="primary-button" type="button" onClick={() => void startBenchmark()} disabled={action !== null}>{action === "start" ? "Starting..." : benchmark.completed ? "Resume benchmark" : "Start benchmark"}</button>}
            </div>

            <ResultRows results={benchmark.results} />
          </article>

          <aside className="panel external-suite-panel">
            <header className="panel-header">
              <div><p className="panel-kicker">External proof</p><h2>Fable A/B</h2></div>
              <span className={`status-chip ${suiteTone(fable)}`}>{phaseLabel(fable?.phase ?? "not ready")}</span>
            </header>
            <p>{fable?.message ?? "The frozen external suite is not available."}</p>
            <dl className="external-status-list">
              <div><dt>Verified results</dt><dd>{fable ? `${fable.completed} / ${fable.total}` : "—"}</dd></div>
              <div><dt>Blind pairs</dt><dd>{fable?.review?.pairs_exported ? "Exported" : "Waiting"}</dd></div>
              <div><dt>Reviews</dt><dd>{fable?.review ? `${fable.review.review_count} / ${fable.review.required_reviews}` : "—"}</dd></div>
            </dl>
            <div className="external-truth"><strong>No simulated run</strong><p>Fable runs outside AutoTrainer. This view advances only when returned work is ingested and verified locally.</p></div>
          </aside>
        </div>
      )}
    </section>
  );
}
