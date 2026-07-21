import { useEffect, useMemo, useState } from "react";
import {
  getCurriculumWorkspace,
  type CurriculumRollout,
  type CurriculumTask,
  type CurriculumWorkspace,
} from "./api";
import TelemetryChart, { type ChartPoint } from "./TelemetryChart";

type Granularity = "overview" | "tasks" | "rollouts";
type PanelContext = "data" | "training";

export type TrainingSystemTelemetry = {
  throughput: ChartPoint[];
  vramAllocated: ChartPoint[];
  vramReserved: ChartPoint[];
  vramLimit: ChartPoint[];
};

const granularities: Array<{ id: Granularity; label: string }> = [
  { id: "overview", label: "Overview" },
  { id: "tasks", label: "Tasks" },
  { id: "rollouts", label: "Rollouts" },
];

const rubricLabels: Record<string, string> = {
  design_rules: "Design rules",
  patch_quality: "Patch quality",
  regression_safety: "Regression safety",
  responsive_rules: "Responsive rules",
  task_tests: "Task tests",
};

const checkLabels: Record<string, string> = {
  browser_tests: "Browser tests",
  build: "Build check",
  hidden_verifier: "Hidden verifier",
  tests: "Regression tests",
};

const outcomeLabels: Record<string, string> = {
  flat: "Flat outcomes",
  uncalibrated: "Needs more evidence",
  unobserved: "Not observed",
  varied: "Varied outcomes",
};

function humanize(value: string) {
  return value.replaceAll("_", " ");
}

function labelFor(value: string, labels: Record<string, string>) {
  return labels[value] ?? humanize(value).replace(/^./, (letter) => letter.toUpperCase());
}

function score(value: number | null) {
  return value !== null && Number.isFinite(value) ? value.toFixed(3) : "-";
}

function percent(value: number | null) {
  return value !== null && Number.isFinite(value) ? `${Math.round(value * 100)}%` : "-";
}

function barWidth(value: number | null) {
  return value === null || !Number.isFinite(value) ? 0 : Math.min(100, Math.max(0, value * 100));
}

function duration(value: number | null) {
  if (value === null || !Number.isFinite(value)) return "-";
  if (value < 60) return `${value.toFixed(1)}s`;
  return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
}

function observedAt(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value || "Time unavailable" : date.toLocaleString();
}

function tone(value: string) {
  const normalized = value.toLowerCase();
  if (["ready", "compiled", "completed", "aligned", "matched", "varied"].includes(normalized)) return "good";
  if (["blocked", "failed", "mismatch", "mismatched"].includes(normalized)) return "danger";
  if (["running", "queued", "uncalibrated"].includes(normalized)) return "info";
  return "muted";
}

function GranularitySwitch({ value, onChange, label }: { value: Granularity; onChange: (value: Granularity) => void; label: string }) {
  return (
    <div className="granularity-switch" role="group" aria-label={label}>
      {granularities.map((item) => (
        <button key={item.id} type="button" aria-pressed={value === item.id} onClick={() => onChange(item.id)}>
          {item.label}
        </button>
      ))}
    </div>
  );
}

function TruthDisclosure({ workspace }: { workspace: CurriculumWorkspace }) {
  if (workspace.summary.rollout_count === 0 && workspace.unmatched_observations.length === 0) return null;
  const aligned = ["aligned", "matched"].includes(workspace.run.catalog_alignment.toLowerCase());
  const window = workspace.run.window;
  return (
    <div className="evidence-disclosure" role="status">
      <strong>{window.truncated ? "Truncated retained rollout window." : "Latest retained rollout window."}</strong>
      <span>
        Showing {workspace.summary.rollout_count} sanitized scored rollout{workspace.summary.rollout_count === 1 ? "" : "s"}
        {workspace.run.job_id ? ` from job ${workspace.run.job_id.slice(0, 12)}` : ""}. This is not reconstructed or estimated history.
      </span>
      <span>The backend retained {window.retained_event_count} of {window.observed_event_count} observed event{window.observed_event_count === 1 ? "" : "s"} in this {humanize(window.scope)}.</span>
      {!aligned && <span>Catalog alignment is {humanize(workspace.run.catalog_alignment)}; these observations may not describe the current compiled tasks.</span>}
      {workspace.unmatched_observations.length > 0 && <span>{workspace.unmatched_observations.length} retained observation{workspace.unmatched_observations.length === 1 ? " does" : "s do"} not match this catalog and remain excluded below.</span>}
    </div>
  );
}

function ActiveEpisodes({ workspace }: { workspace: CurriculumWorkspace }) {
  if (
    workspace.run.catalog_alignment !== "matched"
    || !["queued", "running"].includes(workspace.run.status)
    || workspace.run.active_episodes.length === 0
  ) return null;
  return (
    <div className="current-trial grpo-active-episodes" role="status" aria-live="polite">
      <span className="pulse-marker" aria-hidden="true" />
      <div><strong>{workspace.run.active_episodes.length} rollout{workspace.run.active_episodes.length === 1 ? " is" : "s are"} active</strong><p>Started by the local trainer and not scored yet.</p></div>
      <dl>
        {workspace.run.active_episodes.slice(-3).map((episode) => (
          <div key={episode.episode_id}><dt>Task / family</dt><dd>{episode.task_id}{episode.task_family_id ? ` / ${episode.task_family_id}` : ""}</dd></div>
        ))}
      </dl>
    </div>
  );
}

function Overview({ workspace, tasks, teachingLoss, systemTelemetry, training }: {
  workspace: CurriculumWorkspace;
  tasks: CurriculumTask[];
  teachingLoss: ChartPoint[];
  systemTelemetry: TrainingSystemTelemetry;
  training: boolean;
}) {
  const outcomes = Object.entries(workspace.summary.outcome_states).filter(([, count]) => count > 0);
  const outcomeDenominator = Math.max(tasks.length, 1);
  const taskIds = new Set(tasks.map((task) => task.id));
  const matchedRollouts = workspace.rollouts.filter((rollout) => taskIds.has(rollout.task_id));
  const rewardPoints = matchedRollouts.map((rollout) => ({ x: rollout.sequence, y: rollout.reward }));
  const rubricSeries = Object.entries(rubricLabels).map(([name, label], index) => ({
    id: name,
    label,
    color: ["#169b66", "#e09232", "#4c81d9", "#9b63d8", "#d04d73"][index],
    points: matchedRollouts.flatMap((rollout) => {
      const value = rollout.rubric[name];
      return typeof value === "number" && Number.isFinite(value) ? [{ x: rollout.sequence, y: value }] : [];
    }),
  }));
  return (
    <div className="grpo-view" data-granularity="overview">
      <div className="run-message grpo-run-summary">
        <span className={`health-dot ${tone(workspace.run.status || workspace.catalog.status)}`} aria-hidden="true" />
        <div>
          <strong>{tasks.length} train task{tasks.length === 1 ? "" : "s"}</strong>
          <p>{workspace.run.job_id ? `${humanize(workspace.run.status)}${workspace.run.stage ? ` / ${humanize(workspace.run.stage)}` : ""}. ` : ""}{workspace.summary.protected_holdout_count} held-out task{workspace.summary.protected_holdout_count === 1 ? " stays" : "s stay"} protected and out of this training view.</p>
        </div>
        <span className={`status-chip ${tone(workspace.catalog.status)}`}>{humanize(workspace.catalog.status)}</span>
      </div>
      <ActiveEpisodes workspace={workspace} />

      {tasks.length === 0 ? (
        <div className="evidence-empty"><strong>No compiled GRPO tasks</strong><p>Add and prepare an executable task pack. AutoTrainer will not infer a curriculum from repository files.</p></div>
      ) : (
        <div className="grpo-overview-grid">
          <dl className="metric-list grpo-overview-facts">
            <div><dt>Observed tasks</dt><dd>{workspace.summary.observed_task_count} / {tasks.length}</dd></div>
            <div><dt>Retained rollouts</dt><dd>{workspace.summary.rollout_count}</dd></div>
            <div><dt>Verified gate pass</dt><dd>{percent(workspace.summary.hard_gate_pass_rate)}</dd></div>
            <div><dt>Mean reward</dt><dd>{score(workspace.summary.reward_mean)}</dd></div>
          </dl>
          <section className="grpo-outcome-distribution" aria-labelledby="grpo-outcome-heading">
            <h3 id="grpo-outcome-heading">Observed outcome mix</h3>
            {outcomes.length > 0 ? (
              <ul>
                {outcomes.map(([state, count]) => (
                  <li key={state}>
                    <span><strong>{labelFor(state, outcomeLabels)}</strong><code>{count}</code></span>
                    <span className="reward-track" aria-label={`${labelFor(state, outcomeLabels)}: ${count} of ${tasks.length} training tasks`}><i style={{ width: `${Math.min(100, (count / outcomeDenominator) * 100)}%` }} /></span>
                  </li>
                ))}
              </ul>
            ) : <div className="evidence-empty compact"><strong>No task outcomes yet</strong><p>Signal states appear only after trusted rollouts are scored.</p></div>}
          </section>
        </div>
      )}

      {training && (
        <div className="training-observation-charts">
          <TelemetryChart title="Teaching loss" description="Observed supervised loss from the SFT trainer." series={[{ id: "loss", label: "Loss", color: "#7c82ff", points: teachingLoss }]} emptyMessage="This remains empty for practice-only runs or until SFT emits its first trainer log." />
          <TelemetryChart title="GPU memory" description="Allocator observations in GiB, shown against the configured process limit." series={[
            { id: "allocated", label: "Allocated", color: "#4c81d9", points: systemTelemetry.vramAllocated },
            { id: "reserved", label: "Reserved", color: "#9b63d8", points: systemTelemetry.vramReserved },
            { id: "limit", label: "Configured limit", color: "#d04d73", points: systemTelemetry.vramLimit },
          ]} emptyMessage="Memory appears only after a trainer log reads CUDA allocator counters." />
          <TelemetryChart title="Optimization throughput" description="Observed optimizer-step rate between real trainer log windows." series={[{ id: "throughput", label: "Steps / second", color: "#169b66", points: systemTelemetry.throughput }]} emptyMessage="Throughput needs two observed optimizer boundaries; no completion time is inferred from it." />
          <TelemetryChart title="Verified GRPO reward and rubric" description="Trusted task reward and rubric components for scored practice rollouts." series={[
            { id: "reward", label: "Reward", color: "#171e25", points: rewardPoints },
            ...rubricSeries,
          ]} fixedY={{ min: 0, max: 1 }} emptyMessage="This remains empty for teaching-only runs or until a trusted GRPO verifier scores a rollout." />
        </div>
      )}

      {workspace.summary.rollout_count > 0 && (
        <details className="advanced-options grpo-rubric-means">
          <summary>Observed rubric means</summary>
          <dl className="metric-list">
            {Object.entries(workspace.summary.rubric_means).map(([name, value]) => <div key={name}><dt>{labelFor(name, rubricLabels)}</dt><dd>{score(value)}</dd></div>)}
          </dl>
        </details>
      )}
      <TruthDisclosure workspace={workspace} />
    </div>
  );
}

function TaskDetail({ task, onViewRollouts }: { task: CurriculumTask; onViewRollouts: (task: CurriculumTask) => void }) {
  return (
    <section className="grpo-task-detail" aria-labelledby="selected-task-heading">
      <header>
        <div><p className="panel-kicker">Selected training task</p><h3 id="selected-task-heading">{task.id}</h3></div>
        <span className={`status-chip ${tone(task.status)}`}>{humanize(task.status)}</span>
      </header>
      <p className="grpo-task-instruction">{task.instruction}</p>
      <div className="grpo-aspects" aria-label="Task contract completeness">
        {Object.entries(task.aspects).map(([name, ready]) => (
          <span key={name} className={`status-chip ${ready ? "good" : "danger"}`}>{labelFor(name, rubricLabels)}</span>
        ))}
      </div>
      <div className="grpo-detail-grid">
        <section>
          <h3>Environment</h3>
          <dl className="metric-list">
            <div><dt>Locked snapshot</dt><dd><code>{task.source_id || "Unknown source"}@{task.source_revision || "unknown revision"}</code></dd></div>
            <div><dt>Task family</dt><dd><code>{task.task_family_id || "unassigned"}</code></dd></div>
            <div><dt>Bounded tools</dt><dd>{task.tools.length > 0 ? task.tools.map(humanize).join(", ") : "-"}</dd></div>
            <div><dt>Tool-call limit</dt><dd>{String(task.limits.tool_calls ?? "-")}</dd></div>
            <div><dt>Episode timeout</dt><dd>{task.limits.episode_timeout_seconds ? `${task.limits.episode_timeout_seconds}s` : "-"}</dd></div>
            <div><dt>Network</dt><dd>{task.limits.network_access === false ? "disabled" : "not verified"}</dd></div>
          </dl>
        </section>
        <section>
          <h3>Verifier and reward</h3>
          <dl className="metric-list">
            {Object.entries(task.checks).map(([name, configured]) => (
              <div key={name}><dt>{labelFor(name, checkLabels)}</dt><dd>{configured ? "configured" : "not declared"}</dd></div>
            ))}
            {Object.entries(task.reward_weights).map(([name, weight]) => (
              <div key={name}><dt>{labelFor(name, rubricLabels)}</dt><dd>{percent(weight)}</dd></div>
            ))}
          </dl>
        </section>
      </div>
      <button className="secondary-button" type="button" onClick={() => onViewRollouts(task)}>View this task's rollouts</button>
    </section>
  );
}

function Tasks({ tasks, onViewRollouts }: { tasks: CurriculumTask[]; onViewRollouts: (task: CurriculumTask) => void }) {
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const selectedTask = tasks.find((task) => task.id === selectedTaskId) ?? null;
  if (tasks.length === 0) return <div className="evidence-empty"><strong>No training tasks to inspect</strong><p>Held-out tasks stay protected; only train-split tasks appear here.</p></div>;
  return (
    <div className="grpo-view" data-granularity="tasks">
      <div className="evaluation-table-scroll">
        <table className="evaluation-table grpo-task-table">
          <thead><tr><th>Task</th><th>Family</th><th>Declaration</th><th>Rollouts</th><th>Reward</th><th>Outcome</th><th /></tr></thead>
          <tbody>
            {tasks.map((task) => (
              <tr key={task.id}>
                <td className="grpo-task-name"><strong>{task.id}</strong><small>{task.instruction}</small></td>
                <td><code>{task.task_family_id || "Unassigned"}</code><small>{task.source_id || "Unknown source"}</small></td>
                <td><span className={`status-chip ${tone(task.status)}`}>{humanize(task.declaration_state)}</span></td>
                <td>{task.observed.rollout_count}</td>
                <td><div className="reward-cell"><span className="reward-track"><i style={{ width: `${barWidth(task.observed.reward_mean)}%` }} /></span><code>{score(task.observed.reward_mean)}</code></div></td>
                <td><span className={`status-chip ${tone(task.observed.outcome_mix || task.observed.gate_pattern)}`}>{humanize(task.observed.outcome_mix || task.observed.gate_pattern || "unobserved")}</span></td>
                <td><button className="text-button" type="button" onClick={() => setSelectedTaskId(task.id)}>Inspect</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {selectedTask && <TaskDetail task={selectedTask} onViewRollouts={onViewRollouts} />}
    </div>
  );
}

function RolloutDetail({ rollout, task }: { rollout: CurriculumRollout; task: CurriculumTask | null }) {
  // Privacy boundary: render only sanitized counts and verifier scores, never
  // prompts, reasoning, patches, tool arguments, or command output.
  const tools = Object.entries(rollout.tool_calls_by_name).filter(([, count]) => count > 0);
  return (
    <section className="grpo-rollout-detail" aria-labelledby="selected-rollout-heading">
      <header>
        <div><p className="panel-kicker">Selected observed rollout</p><h3 id="selected-rollout-heading"><code>{rollout.episode_id || `episode-${rollout.sequence}`}</code> on {rollout.task_id}</h3></div>
        <span className={`status-chip ${rollout.hard_gate_passed ? "good" : "danger"}`}>{rollout.hard_gate_passed ? "verified" : "gate failed"}</span>
      </header>
      {task && (
        <div className="grpo-task-context">
          <p>{task.instruction}</p>
          <p><code>{task.source_id || "Unknown source"}@{task.source_revision || "unknown revision"}</code> / family <code>{task.task_family_id || "unassigned"}</code></p>
          <div className="grpo-aspects" aria-label="Declared task aspects">
            {Object.entries(task.aspects).map(([name, ready]) => <span key={name} className={`status-chip ${ready ? "good" : "danger"}`}>{labelFor(name, rubricLabels)}</span>)}
          </div>
        </div>
      )}
      {rollout.gate_reason && <div className="evidence-disclosure"><strong>Gate reason</strong><span>{rollout.gate_reason}</span></div>}
      <div className="grpo-detail-grid">
        <section>
          <h3>Observed work</h3>
          <dl className="metric-list">
            <div><dt>Tool calls</dt><dd>{rollout.tool_call_count ?? "-"}</dd></div>
            <div><dt>Changed files</dt><dd>{rollout.changed_file_count ?? "-"}</dd></div>
            <div><dt>Elapsed</dt><dd>{duration(rollout.elapsed_seconds)}</dd></div>
            <div><dt>Recorded</dt><dd>{observedAt(rollout.observed_at)}</dd></div>
            {tools.map(([name, count]) => <div key={name}><dt>{humanize(name)}</dt><dd>x {count}</dd></div>)}
          </dl>
        </section>
        <section>
          <h3>Verified score</h3>
          <dl className="metric-list">
            <div><dt>Total reward</dt><dd>{score(rollout.reward)}</dd></div>
            {Object.entries(rollout.rubric).map(([name, value]) => (
              <div key={name}><dt>{labelFor(name, rubricLabels)}</dt><dd>{score(value)}{task?.reward_weights[name] !== undefined ? <small> / {percent(task.reward_weights[name])} weight</small> : null}</dd></div>
            ))}
          </dl>
        </section>
      </div>
    </section>
  );
}

function Rollouts({ tasks, rollouts, selectedTaskId, selectedSequence, onSelect }: {
  tasks: CurriculumTask[];
  rollouts: CurriculumRollout[];
  selectedTaskId: string | null;
  selectedSequence: number | null;
  onSelect: (rollout: CurriculumRollout) => void;
}) {
  const taskIds = new Set(tasks.map((task) => task.id));
  // Holdout or unmatched observations never enter the visible rollout surface.
  const matched = rollouts.filter((rollout) => taskIds.has(rollout.task_id));
  const filtered = selectedTaskId ? matched.filter((rollout) => rollout.task_id === selectedTaskId) : matched;
  const visible = [...filtered].sort((left, right) => right.sequence - left.sequence).slice(0, 50);
  const selected = selectedSequence === null ? null : matched.find((rollout) => rollout.sequence === selectedSequence) ?? null;
  const selectedTask = selected ? tasks.find((task) => task.id === selected.task_id) ?? null : null;

  if (matched.length === 0) return <div className="evidence-empty"><strong>No retained rollout observations</strong><p>Rollouts appear only after the trusted environment scores real model work. No activity is simulated here.</p></div>;
  return (
    <div className="grpo-view" data-granularity="rollouts">
      {selectedTaskId && filtered.length === 0 ? <div className="evidence-empty compact"><strong>No retained rollout for {selectedTaskId}</strong><p>This task has no scored episode in the current retained window.</p></div> : (
        <div className="evaluation-table-scroll">
          <table className="evaluation-table grpo-rollout-table">
            <thead><tr><th>Episode</th><th>Task</th><th>Observed work</th><th>Files</th><th>Reward</th><th>Gate</th><th /></tr></thead>
            <tbody>
              {visible.map((rollout) => (
                <tr key={`${rollout.episode_id || "episode"}-${rollout.sequence}`}>
                  <td><code>{rollout.episode_id || "Unavailable"}</code><small>#{rollout.sequence}</small></td>
                  <td><strong>{rollout.task_id}</strong></td>
                  <td>{rollout.tool_call_count ?? "-"} tools / {duration(rollout.elapsed_seconds)}</td>
                  <td>{rollout.changed_file_count ?? "-"}</td>
                  <td><div className="reward-cell"><span className="reward-track"><i style={{ width: `${Math.min(100, Math.max(0, rollout.reward * 100))}%` }} /></span><code>{score(rollout.reward)}</code></div></td>
                  <td><span className={`status-chip ${rollout.hard_gate_passed ? "good" : "danger"}`}>{rollout.hard_gate_passed ? "pass" : "fail"}</span></td>
                  <td><button className="text-button" type="button" onClick={() => onSelect(rollout)}>Inspect</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {filtered.length > 50 && <p className="grpo-window-note">Showing the latest 50 of {filtered.length} matched rollouts in the retained backend window.</p>}
      {selected ? <RolloutDetail rollout={selected} task={selectedTask} /> : selectedSequence !== null ? (
        <div className="evidence-disclosure"><strong>The selected rollout is no longer retained.</strong><span>Choose another observed rollout; AutoTrainer will not silently replace your selection.</span></div>
      ) : null}
    </div>
  );
}

export default function GrpoEvidencePanel({ context, refreshKey = 0, live = false, teachingLoss = [], systemTelemetry = { throughput: [], vramAllocated: [], vramReserved: [], vramLimit: [] } }: {
  context: PanelContext;
  refreshKey?: string | number;
  live?: boolean;
  teachingLoss?: ChartPoint[];
  systemTelemetry?: TrainingSystemTelemetry;
}) {
  const [granularity, setGranularity] = useState<Granularity>("overview");
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [selectedSequence, setSelectedSequence] = useState<number | null>(null);
  const [workspace, setWorkspace] = useState<CurriculumWorkspace | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stopped = false;
    const controller = new AbortController();
    let timer = 0;
    setWorkspace(null);
    setLoading(true);
    setError(null);

    const load = async () => {
      try {
        const next = await getCurriculumWorkspace(controller.signal);
        if (stopped) return;
        setWorkspace(next);
        setError(null);
      } catch (reason) {
        if (!stopped && !controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Curriculum evidence could not be refreshed.");
      } finally {
        if (!stopped) {
          setLoading(false);
          if (live) timer = window.setTimeout(() => void load(), 2_000);
        }
      }
    };

    // Only the sanitized localhost curriculum contract feeds this panel; it
    // never fabricates progress from timers, task declarations, or UI state.
    void load();
    return () => {
      stopped = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [live, refreshKey]);

  // Live refreshes intentionally never change granularity or the user's
  // selected task/rollout, so watching training cannot steal their context.
  const tasks = useMemo(() => workspace?.tasks.filter((task) => task.split === "train") ?? [], [workspace]);

  const viewRollouts = (task: CurriculumTask) => {
    setSelectedTaskId(task.id);
    setSelectedSequence(null);
    setGranularity("rollouts");
  };
  const selectRollout = (rollout: CurriculumRollout) => {
    setSelectedTaskId(rollout.task_id);
    setSelectedSequence(rollout.sequence);
  };

  const headingId = `${context}-grpo-evidence-heading`;
  const isTraining = context === "training";
  return (
    <article className="panel grpo-evidence-panel" aria-labelledby={headingId}>
      <header className="panel-header">
        <div className="grpo-evidence-heading">
          <p className="panel-kicker">{isTraining ? "Observed model work" : "Executable practice"}</p>
          <h2 id={headingId}>{isTraining ? "GRPO observations" : "GRPO curriculum"}</h2>
          <p>{isTraining ? "Watch sanitized, verifier-backed outcomes from this local run." : "Inspect train tasks and the verified outcomes retained from local runs."}</p>
        </div>
        <GranularitySwitch value={granularity} onChange={setGranularity} label={`${isTraining ? "Training" : "Curriculum"} detail`} />
      </header>

      {error && <div className="source-error grpo-evidence-error" role="alert">{error}</div>}
      {!workspace ? (
        <div className="evidence-empty"><strong>{loading ? "Loading observed evidence" : "Curriculum evidence unavailable"}</strong><p>{loading ? "Reading the compiled catalog and its retained rollout window from the local backend." : "Reconnect the local backend to inspect real GRPO tasks and outcomes."}</p></div>
      ) : (
        <>
          {granularity === "overview" && <Overview workspace={workspace} tasks={tasks} teachingLoss={isTraining ? teachingLoss : []} systemTelemetry={systemTelemetry} training={isTraining} />}
          {granularity === "tasks" && <Tasks tasks={tasks} onViewRollouts={viewRollouts} />}
          {granularity === "rollouts" && <Rollouts tasks={tasks} rollouts={workspace.rollouts} selectedTaskId={selectedTaskId} selectedSequence={selectedSequence} onSelect={selectRollout} />}
          {workspace.next_action && <div className="training-next-action grpo-next-action" role="note"><span>Suggested next step</span><strong>{workspace.next_action.title}</strong><p>{workspace.next_action.detail}</p></div>}
          <details className="advanced-options grpo-provenance">
            <summary>Catalog provenance</summary>
            <dl className="metric-list">
              <div><dt>Fingerprint</dt><dd><code>{workspace.catalog.fingerprint || "Unavailable"}</code></dd></div>
              <div><dt>Dataset SHA-256</dt><dd><code>{workspace.catalog.dataset_sha256 || "Unavailable"}</code></dd></div>
              <div><dt>Run</dt><dd><code>{workspace.run.job_id || "No observed run"}</code></dd></div>
              <div><dt>Catalog alignment</dt><dd>{humanize(workspace.run.catalog_alignment)}</dd></div>
            </dl>
          </details>
        </>
      )}
    </article>
  );
}
