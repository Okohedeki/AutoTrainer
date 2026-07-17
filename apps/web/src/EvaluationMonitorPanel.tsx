import { useEffect, useMemo, useRef, useState } from "react";
import {
  ApiClientError,
  getEvaluationEvents,
  getEvaluationWorkspace,
  startEvaluation,
  type EvaluationEvent,
  type EvaluationResult,
  type EvaluationSuite,
  type EvaluationTrial,
  type EvaluationWorkspace,
} from "./api";
import TelemetryChart, { type ChartPoint, type ChartSeries } from "./TelemetryChart";

const liveStatuses = new Set(["queued", "running"]);
const rubricKeys = ["design_rules", "patch_quality", "regression_safety", "responsive_rules", "task_tests"] as const;
const rubricLabels: Record<(typeof rubricKeys)[number], string> = {
  design_rules: "Design rules",
  patch_quality: "Patch quality",
  regression_safety: "Regression safety",
  responsive_rules: "Responsive rules",
  task_tests: "Task tests",
};
const rubricColors = ["#7c82ff", "#2fb785", "#f0a34a", "#e06c75", "#57a9d9"];

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

function cumulative(results: EvaluationResult[], read: (result: EvaluationResult) => number | null): ChartPoint[] {
  let total = 0;
  let count = 0;
  const points: ChartPoint[] = [];
  results.forEach((result, index) => {
    const value = read(result);
    if (value === null) return;
    total += value;
    count += 1;
    points.push({ x: index + 1, y: total / count });
  });
  return points;
}

function eventText(event: EvaluationEvent) {
  if (event.message) return event.message;
  const task = event.task_id || event.trial?.task_id || event.result?.task_id;
  const kind = event.type || event.phase || "evaluation_event";
  if (["trial_started", "generating"].includes(kind)) return task ? `Generating a patch for ${task}` : "Generating a model patch";
  if (["verification_started", "verifying"].includes(kind)) return task ? `Running trusted checks for ${task}` : "Running trusted checks";
  if (kind === "trial_completed") return task ? `Scored ${task}` : "Trial scored";
  if (["job_completed", "completed"].includes(kind)) return "Held-out evaluation completed";
  if (["job_failed", "failed", "interrupted"].includes(kind)) return "Evaluation stopped";
  return kind.replaceAll("_", " ");
}

function eventMarkerTone(kind: string, hardGatePassed: boolean | undefined) {
  // A completed verifier event can still be a failed model attempt. The hard
  // gate therefore takes precedence over the event lifecycle label.
  if (hardGatePassed === false || kind.includes("failed") || kind === "interrupted") return "danger";
  if (hardGatePassed === true || kind.includes("completed")) return "good";
  return "info";
}

function ResultRows({ results }: { results: EvaluationResult[] }) {
  if (results.length === 0) return <div className="evaluation-empty"><strong>No scored trials yet</strong><p>Results appear only after generated work passes through the trusted local verifier.</p></div>;
  return (
    <div className="evaluation-table-scroll">
      <table className="evaluation-table">
        <thead><tr><th>Task</th><th>Arm</th><th>Repeat</th><th>Hard gate</th><th>Verified reward</th></tr></thead>
        <tbody>
          {results.slice(-16).reverse().map((result) => (
            <tr key={result.trial_id}>
              <td><strong>{result.task_id}</strong></td>
              <td><code>{result.arm_id}</code></td>
              <td>{result.repetition + 1}</td>
              <td><span className={`status-chip ${result.hard_gate_passed ? "good" : "danger"}`}>{result.hard_gate_passed ? "pass" : "fail"}</span></td>
              <td><div className="reward-cell"><span className="reward-track" aria-hidden="true"><i style={{ width: `${Math.max(0, Math.min(1, result.reward)) * 100}%` }} /></span><code>{result.reward.toFixed(3)}</code></div></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TrialMatrix({ trials, current, results }: { trials: EvaluationTrial[]; current: EvaluationTrial | null; results: EvaluationResult[] }) {
  const resultIds = new Set(results.map((result) => result.trial_id));
  return (
    <div className="evaluation-table-scroll trial-matrix-scroll">
      <table className="evaluation-table trial-matrix">
        <thead><tr><th>Status</th><th>Task</th><th>Model arm</th><th>Repeat</th><th>Seed</th></tr></thead>
        <tbody>
          {trials.map((trial) => {
            const published = "status" in trial && typeof trial.status === "string" ? trial.status : null;
            const status = resultIds.has(trial.trial_id) ? "scored" : current?.trial_id === trial.trial_id ? "running" : published || "queued";
            return (
              <tr key={trial.trial_id}>
                <td><span className={`status-chip ${status === "scored" || status === "completed" ? "good" : status === "running" || status === "generating" || status === "verifying" ? "info" : "muted"}`}>{phaseLabel(status)}</span></td>
                <td><strong>{trial.task_id}</strong></td><td><code>{trial.arm_id}</code></td><td>{trial.repetition + 1}</td><td>{trial.seed}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// Evaluation never changes weights. The screen follows frozen trial identities
// through generation, verification, and scoring without exposing hidden prompts
// or inventing token-level activity.
export default function EvaluationMonitorPanel({ onOpenData }: { onOpenData: () => void }) {
  const requestVersion = useRef(0);
  const cursorRef = useRef(0);
  const planIdRef = useRef<string | null>(null);
  const [workspace, setWorkspace] = useState<EvaluationWorkspace | null>(null);
  const [events, setEvents] = useState<EvaluationEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const refresh = async (signal?: AbortSignal) => {
    const version = ++requestVersion.current;
    try {
      const [next, initialPage] = await Promise.all([
        getEvaluationWorkspace(signal),
        getEvaluationEvents(cursorRef.current, signal),
      ]);
      if (version !== requestVersion.current) return;
      setWorkspace(next);
      const currentPlanId = next.plan?.plan_id || next.job.plan_id;
      const planRolledOver = planIdRef.current !== null && planIdRef.current !== currentPlanId;
      let page = initialPage;
      if (planRolledOver) {
        // Event sequence numbers may restart with a replacement manager/job.
        // Fetch from zero before advancing the cursor or early evidence from a
        // CLI-started plan could be skipped permanently.
        page = await getEvaluationEvents(0, signal);
        if (version !== requestVersion.current) return;
      }
      const currentEvents = page.events.filter((event) => !currentPlanId || event.plan_id === currentPlanId);
      // A CLI can freeze a replacement plan while this screen is open. Even
      // when that plan has no events yet, replace the old evidence instead of
      // merging two immutable plans into one rail.
      if (page.cursor_reset || planRolledOver) setEvents(currentEvents);
      else if (currentEvents.length) {
        setEvents((current) => {
          const merged = new Map(current.map((event) => [event.sequence, event]));
          currentEvents.forEach((event) => merged.set(event.sequence, event));
          return [...merged.values()].sort((a, b) => a.sequence - b.sequence).slice(-500);
        });
      }
      planIdRef.current = currentPlanId;
      cursorRef.current = page.latest_sequence ?? cursorRef.current;
      setError(null);
    } catch (reason) {
      if (!signal?.aborted) setError(reason instanceof Error ? reason.message : "Evaluation status could not be refreshed.");
    }
  };

  useEffect(() => {
    const controller = new AbortController();
    let stopped = false;
    let timer = 0;
    const poll = async () => {
      await refresh(controller.signal);
      if (!stopped) timer = window.setTimeout(() => void poll(), 2_000);
    };
    void poll();
    return () => {
      stopped = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, []);

  const benchmark = useMemo(() => workspace?.suites.find((suite) => suite.id === "model_benchmark"), [workspace]);
  const fable = useMemo(() => workspace?.suites.find((suite) => suite.id === "fable_ab"), [workspace]);
  const results = benchmark?.results ?? [];
  const trials = benchmark?.trials ?? workspace?.plan?.trials ?? workspace?.job.planned_trials ?? [];
  const resultsTruncated = Boolean(benchmark?.results_truncated || workspace?.job.results_truncated);
  const trialsTruncated = Boolean(benchmark?.trials_truncated);
  const jobLive = liveStatuses.has(workspace?.job.status ?? "idle");
  const canRun = workspace?.readiness.status === "ready" && !jobLive;
  const benchmarkProgress = benchmark?.total ? benchmark.completed / benchmark.total : 0;

  const proofSeries = useMemo<ChartSeries[]>(() => [
    { id: "reward", label: resultsTruncated ? "Visible-window reward" : "Mean reward", color: "#d8ff58", points: cumulative(results, (result) => result.reward) },
    { id: "success", label: resultsTruncated ? "Visible-window success" : "Verified success", color: "#2fb785", points: cumulative(results, (result) => result.hard_gate_passed ? 1 : 0) },
  ], [results, resultsTruncated]);
  const rubricSeries = useMemo<ChartSeries[]>(() => rubricKeys.map((key, index) => ({
    id: key,
    label: rubricLabels[key],
    color: rubricColors[index],
    points: cumulative(results, (result) => typeof result.components[key] === "number" ? result.components[key] : null),
  })), [results]);
  const proofDescription = resultsTruncated
    ? `Cumulative means over the latest ${results.length} returned trials, not all ${benchmark?.completed ?? workspace?.job.completed ?? results.length} scored trials.`
    : `Cumulative means from ${results.length} scored trial${results.length === 1 ? "" : "s"}.`;
  const resultRowsShown = Math.min(results.length, 16);

  const run = async () => {
    setStarting(true);
    setError(null);
    requestVersion.current += 1;
    try {
      cursorRef.current = 0;
      setEvents([]);
      await startEvaluation();
      await refresh();
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "The held-out evaluation could not start.");
    } finally {
      setStarting(false);
    }
  };

  return (
    <section className="evaluation-workspace" aria-label="Evaluation workspace" data-tour="evaluate">
      {error && <div className="source-error evaluation-error" role="alert">{error}</div>}

      <article className="panel evaluation-command-panel">
        <header className="panel-header"><div><p className="panel-kicker">Frozen held-out proof</p><h2>Run evaluation</h2></div><span className={`status-chip ${jobLive ? "info" : canRun ? "good" : "warning"}`}>{jobLive ? phaseLabel(workspace?.job.phase || "running") : workspace?.readiness.status || "connecting"}</span></header>
        <div className="evaluation-command-copy">
          <div><strong>Weights are frozen. Nothing learns here.</strong><p>One action freezes the local trial plan, generates model work, runs trusted checks, and scores every completed trial.</p></div>
          <button className="primary-button" type="button" onClick={() => void run()} disabled={!canRun || starting}>{starting ? "Starting..." : benchmark && benchmark.completed > 0 && benchmark.completed < benchmark.total ? "Resume evaluation" : "Run held-out evaluation"}</button>
        </div>
        {(workspace?.readiness.blockers.length ?? 0) > 0 && <div className="evaluation-blocked"><ul>{workspace?.readiness.blockers.map((blocker) => <li key={blocker}>{blocker}</li>)}</ul><button className="secondary-button" type="button" onClick={onOpenData}>Review Data</button></div>}
      </article>

      <div className="evaluation-summary">
        <article className="panel"><span>Held-out tasks</span><strong>{workspace?.readiness.ready_task_count ?? "-"}</strong><small>Isolated from training</small></article>
        <article className="panel"><span>Frozen plan</span><strong>{workspace?.plan ? workspace.plan.plan_id.slice(0, 14) : "Not frozen"}</strong><small>{workspace?.plan ? `${workspace.plan.repetitions} repetitions` : "Created when evaluation starts"}</small></article>
        <article className="panel"><span>Verified trials</span><strong>{benchmark ? `${benchmark.completed} / ${benchmark.total}` : "-"}</strong><small>Trusted local scores</small></article>
        <article className="panel"><span>Current phase</span><strong>{phaseLabel(workspace?.job.phase || benchmark?.phase || "idle")}</strong><small>{workspace?.job.current_trial?.task_id || "No active trial"}</small></article>
      </div>

      {benchmark && <div className="observed-progress" aria-label={`${benchmark.completed} of ${benchmark.total} trusted trials complete`}><div><strong>{benchmark.completed} / {benchmark.total}</strong><span>trusted trials complete</span></div><span aria-hidden="true"><i style={{ width: `${benchmarkProgress * 100}%` }} /></span></div>}

      <div className="evaluation-observation-grid">
        <article className="panel evaluation-graph-panel">
          <header className="panel-header"><div><p className="panel-kicker">Observed proof</p><h2>Live evaluation rubric</h2></div><span className={`status-chip ${suiteTone(benchmark)}`}>{phaseLabel(benchmark?.phase || "idle")}</span></header>
          {jobLive && workspace?.job.current_trial && <div className="current-trial" role="status"><span className="pulse-marker" aria-hidden="true" /><div><strong>{phaseLabel(workspace.job.phase)}</strong><p>{workspace.job.message}</p></div><dl><div><dt>Task</dt><dd>{workspace.job.current_trial.task_id}</dd></div><div><dt>Arm</dt><dd>{workspace.job.current_trial.arm_id}</dd></div><div><dt>Seed</dt><dd>{workspace.job.current_trial.seed}</dd></div></dl></div>}
          {resultsTruncated && <div className="evidence-disclosure" role="status"><strong>Graph window is truncated.</strong><span>The backend returned only its latest result window, so these lines are visible-window means rather than whole-run means.</span></div>}
          <TelemetryChart title="Reward and verified success" description={proofDescription} series={proofSeries} fixedY={{ min: 0, max: 1 }} emptyMessage="The graph begins after the trusted verifier scores the first generated result." />
          <TelemetryChart title="Rubric components" description={resultsTruncated ? `Five verifier components over the latest ${results.length} returned trials only.` : `Five verifier components from ${results.length} scored trial${results.length === 1 ? "" : "s"}.`} series={rubricSeries} fixedY={{ min: 0, max: 1 }} emptyMessage="Component lines appear as trusted trial results arrive." />
        </article>

        <aside className="panel event-rail-panel evaluation-event-rail">
          <header className="panel-header"><div><p className="panel-kicker">Live verification</p><h2>What is happening</h2></div><span className="status-chip muted">{events.length} events</span></header>
          {events.length === 0 ? <div className="evidence-empty"><strong>No verification events yet</strong><p>Run evaluation to watch each frozen trial move through generation, trusted verification, and scoring.</p></div> : (
            <ol className="event-rail">
              {events.slice(-30).reverse().map((event) => {
                const kind = event.type || event.phase || "event";
                const passed = event.hard_gate_passed ?? event.result?.hard_gate_passed ?? event.rubric?.hard_gate_passed;
                return <li key={event.sequence}><span className={`event-marker ${eventMarkerTone(kind, passed)}`} aria-hidden="true" /><div><strong>{eventText(event)}</strong><p>{phaseLabel(kind)}{event.gate_reason ? ` / ${event.gate_reason}` : ""}</p></div><code>{event.sequence}</code></li>;
              })}
            </ol>
          )}
        </aside>
      </div>

      <article className="panel trial-matrix-panel">
        <header className="panel-header"><div><p className="panel-kicker">Frozen work order</p><h2>Planned trial matrix</h2></div><span className="status-chip muted">{trialsTruncated ? `${trials.length} of ${benchmark?.total ?? trials.length} shown` : `${trials.length || benchmark?.total || 0} trials`}</span></header>
        {trialsTruncated && <div className="evidence-disclosure" role="status"><strong>Trial list is truncated.</strong><span>Showing the first {trials.length} of {benchmark?.total ?? "the full"} frozen trials; the completed count still covers the whole plan.</span></div>}
        {trials.length > 0 ? <TrialMatrix trials={trials} current={workspace?.job.current_trial ?? null} results={results} /> : <div className="evaluation-empty"><strong>No frozen trial identities yet</strong><p>The complete task, model arm, repetition, and seed matrix appears here when the local evaluation starts.</p></div>}
      </article>

      <article className="panel scored-results-panel">
        <header className="panel-header"><div><p className="panel-kicker">Immutable evidence</p><h2>Scored trials</h2></div><span className="status-chip muted">{resultsTruncated || results.length > resultRowsShown ? `${resultRowsShown} shown` : `${results.length} recorded`}</span></header>
        {resultsTruncated && <div className="evidence-disclosure" role="status"><strong>Result history is truncated.</strong><span>The table shows the latest {resultRowsShown} rows from the latest {results.length} results returned for {benchmark?.completed ?? workspace?.job.completed ?? results.length} completed trials.</span></div>}
        {!resultsTruncated && results.length > resultRowsShown && <div className="evidence-disclosure"><strong>Showing the latest {resultRowsShown} rows.</strong><span>The graphs above still include all {results.length} returned results.</span></div>}
        <ResultRows results={results} />
      </article>

      <article className="panel external-proof-strip">
        <div><p className="panel-kicker">External comparison</p><h2>Fable A/B remains separate</h2><p>AutoTrainer never simulates Fable. This state advances only after externally generated work returns and passes local verification.</p></div>
        <dl><div><dt>Status</dt><dd>{phaseLabel(fable?.phase || "not ready")}</dd></div><div><dt>Verified</dt><dd>{fable ? `${fable.completed} / ${fable.total}` : "-"}</dd></div><div><dt>Blind reviews</dt><dd>{fable?.review ? `${fable.review.review_count} / ${fable.review.required_reviews}` : "-"}</dd></div></dl>
      </article>
    </section>
  );
}
