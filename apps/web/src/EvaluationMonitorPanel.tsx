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
import LanguageEvaluationPanel from "./LanguageEvaluationPanel";
import TelemetryChart, { type ChartPoint, type ChartSeries } from "./TelemetryChart";

const liveStatuses = new Set(["queued", "running"]);
const runnableReadinessStatuses = new Set(["ready", "inputs_ready"]);
const rubricKeys = ["design_rules", "patch_quality", "regression_safety", "responsive_rules", "task_tests"] as const;
const rubricLabels: Record<(typeof rubricKeys)[number], string> = {
  design_rules: "Design rules",
  patch_quality: "Patch quality",
  regression_safety: "Regression safety",
  responsive_rules: "Responsive rules",
  task_tests: "Task tests",
};
const rubricColors = ["#7c82ff", "#2fb785", "#f0a34a", "#e06c75", "#57a9d9"];
const armRewardColors = ["#7c82ff", "#2fb785", "#f0a34a", "#57a9d9"];
const armSuccessColors = ["#a6aaff", "#79cfa9", "#f5c27f", "#93cae8"];

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

function percent(value: number | undefined, signed = false) {
  if (typeof value !== "number") return "-";
  const prefix = signed && value > 0 ? "+" : "";
  return `${prefix}${(value * 100).toFixed(1)} pp`;
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
  const [rubricArmId, setRubricArmId] = useState<string | null>(null);

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
  const results = benchmark?.results ?? [];
  const arms = benchmark?.arms ?? [];
  const trials = benchmark?.trials ?? workspace?.plan?.trials ?? workspace?.job.planned_trials ?? [];
  const resultsTruncated = Boolean(benchmark?.results_truncated || workspace?.job.results_truncated);
  const trialsTruncated = Boolean(benchmark?.trials_truncated);
  const jobLive = liveStatuses.has(workspace?.job.status ?? "idle");
  const canRun = runnableReadinessStatuses.has(workspace?.readiness.status ?? "") && !jobLive;
  const benchmarkProgress = benchmark?.total ? benchmark.completed / benchmark.total : 0;

  // Each line advances on that arm's own scored trials. Pooling reference and
  // candidate values would make a healthy average while hiding who improved.
  const proofSeries = useMemo<ChartSeries[]>(() => arms.flatMap((arm, index) => {
    const armResults = results.filter((result) => result.arm_id === arm.id);
    const windowLabel = resultsTruncated ? "visible window" : "cumulative";
    return [
      { id: `${arm.id}-reward`, label: `${arm.label} reward (${windowLabel})`, color: armRewardColors[index % armRewardColors.length], points: cumulative(armResults, (result) => result.reward) },
      { id: `${arm.id}-success`, label: `${arm.label} verified success`, color: armSuccessColors[index % armSuccessColors.length], points: cumulative(armResults, (result) => result.hard_gate_passed ? 1 : 0) },
    ];
  }), [arms, results, resultsTruncated]);
  const selectedRubricArm = arms.some((arm) => arm.id === rubricArmId)
    ? rubricArmId
    : arms.find((arm) => arm.role === "candidate")?.id ?? arms[0]?.id ?? null;
  const selectedRubricResults = useMemo(
    () => results.filter((result) => result.arm_id === selectedRubricArm),
    [results, selectedRubricArm],
  );
  const rubricSeries = useMemo<ChartSeries[]>(() => rubricKeys.map((key, index) => ({
    id: key,
    label: rubricLabels[key],
    color: rubricColors[index],
    points: cumulative(selectedRubricResults, (result) => typeof result.components[key] === "number" ? result.components[key] : null),
  })), [selectedRubricResults]);
  const proofDescription = resultsTruncated
    ? `Separate per-arm means over the latest ${results.length} returned trials, not all ${benchmark?.completed ?? workspace?.job.completed ?? results.length} scored trials.`
    : `Separate per-arm means from ${results.length} scored trial${results.length === 1 ? "" : "s"}; no values are pooled across models.`;
  const armObservationSummary = arms.map((arm) => {
    const count = results.filter((result) => result.arm_id === arm.id).length;
    return `${arm.label}: ${count > 0 ? `${count} scored` : "no scored trials"}`;
  }).join(" · ");
  const report = benchmark?.report ?? null;
  const decision = report?.decision;
  const interval = decision?.confidence_interval;
  const verdict = decision?.verified_better
    ? "Verified improvement"
    : decision?.observed_better
      ? "Observed lead, not verified"
      : report
        ? "No verified improvement"
        : "Decision pending";
  const verdictTone = decision?.verified_better ? "good" : report ? "warning" : "muted";
  const resultRowsShown = Math.min(results.length, 16);
  const hasEvaluationEvidence = Boolean(
    workspace?.plan
    || workspace?.job.id
    || trials.length
    || results.length
    || events.length
    || report,
  );

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

      {!hasEvaluationEvidence ? (
        <article className="panel evaluation-before-run">
          <div><p className="panel-kicker">What you will get</p><h2>No evaluation evidence yet</h2><p>Nothing is missing or broken. Detailed plots and trial tables appear only after the first held-out evaluation starts.</p></div>
          <ol>
            <li><span>1</span><div><strong>Freeze the comparison</strong><small>The same tasks, repeats, and seeds are assigned to both model outputs.</small></div></li>
            <li><span>2</span><div><strong>Run trusted checks</strong><small>AutoTrainer uses the evaluator that matches the project’s primary language.</small></div></li>
            <li><span>3</span><div><strong>Show the proof</strong><small>Plots, scored trials, and the improvement decision appear as real results arrive.</small></div></li>
          </ol>
        </article>
      ) : (
        <>
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
          {arms.length > 0 && <div className="evidence-disclosure" role="status"><strong>Observed model coverage</strong><span>{armObservationSummary}</span><span>An arm with no scored trials is waiting for evidence; AutoTrainer does not treat it as zero.</span></div>}
          <TelemetryChart title="Reward and verified success by model arm" description={proofDescription} series={proofSeries} fixedY={{ min: 0, max: 1 }} emptyMessage="Separate model lines begin after the trusted verifier scores the first generated result." />
          <div className="rubric-arm-switch" aria-label="Rubric model arm">
            <span>Rubric detail</span>
            {arms.map((arm) => <button key={arm.id} type="button" className={selectedRubricArm === arm.id ? "active" : ""} aria-pressed={selectedRubricArm === arm.id} onClick={() => setRubricArmId(arm.id)}>{arm.label}</button>)}
          </div>
          <TelemetryChart title="Rubric components for one model" description={selectedRubricArm ? `Five trusted verifier components for ${arms.find((arm) => arm.id === selectedRubricArm)?.label ?? selectedRubricArm}. Switch arms above to compare without mixing their scores.` : "Select a model arm to inspect its verifier components."} series={rubricSeries} fixedY={{ min: 0, max: 1 }} emptyMessage="Component lines appear after this model arm receives a trusted score." />
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

      <article className="panel evaluation-decision-panel" aria-label="Paired comparison decision">
        <header className="panel-header"><div><p className="panel-kicker">Paired comparison</p><h2>Did training improve the model?</h2></div><span className={`status-chip ${verdictTone}`}>{verdict}</span></header>
        {!report ? <div className="evaluation-empty"><strong>The decision waits for complete paired evidence</strong><p>After both model arms finish every held-out task, the backend calculates the task-clustered confidence interval and publishes the final decision here.</p></div> : (
          <>
            <div className="decision-stat-grid">
              <div><span>Candidate minus reference</span><strong>{percent(decision?.delta, true)}</strong><small>Verified task success</small></div>
              <div><span>{interval ? `${(interval.confidence * 100).toFixed(0)}% confidence interval` : "Confidence interval"}</span><strong>{interval ? `${percent(interval.low, true)} to ${percent(interval.high, true)}` : "Not available"}</strong><small>{interval?.method ?? "At least two independent tasks required"}</small></div>
              <div><span>Independent tasks</span><strong>{decision?.task_count ?? 0}</strong><small>Minimum {decision?.minimum_tasks ?? "-"}</small></div>
              <div><span>Evidence integrity</span><strong>{report.completeness.fairness_passed ? "Passed" : "Not passed"}</strong><small>{report.completeness.completed_trials} / {report.completeness.expected_trials} paired trials</small></div>
            </div>
            <div className="arm-result-grid">
              {report.comparison.candidates.map((candidate) => <section key={candidate.candidate_id}><div><strong>{candidate.label}</strong><span>Rank {candidate.rank}</span></div><dl><div><dt>Verified success</dt><dd>{(candidate.hard_gate_pass_rate * 100).toFixed(1)}%</dd></div><div><dt>Mean reward</dt><dd>{candidate.reward_mean.toFixed(3)}</dd></div></dl></section>)}
            </div>
            <p className="decision-explanation">{decision?.verified_better ? "The candidate cleared the configured minimum delta and the lower confidence bound stayed above it." : decision?.observed_better ? "The point estimate is ahead, but the confidence gate or evidence requirements were not met." : "The trained candidate did not beat the configured reference threshold on this frozen plan."}</p>
          </>
        )}
      </article>

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
        </>
      )}

      <LanguageEvaluationPanel />
    </section>
  );
}
