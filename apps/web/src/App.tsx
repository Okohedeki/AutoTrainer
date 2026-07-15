import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import {
  commands,
  environment,
  evaluationSuites,
  navigation,
  pipelineStages,
  preparationEvents,
  projectSnapshot,
  rewardSignals,
  runtimeSnapshot,
  sources,
  type CommandDefinition,
  type StatusTone,
  type ViewId,
} from "./data";

const viewTitles: Record<ViewId, { eyebrow: string; title: string; description: string }> = {
  overview: {
    eyebrow: "Project control plane",
    title: "Training overview",
    description: "Resolve local blockers, then move one adapter through QLoRA, GRPO, and proof.",
  },
  runs: {
    eyebrow: "Single-GPU queue",
    title: "Training runs",
    description: "Prepare and inspect SFT and GRPO jobs without hiding their local runtime dependencies.",
  },
  data: {
    eyebrow: "Locked provenance",
    title: "Data sources",
    description: "See exactly which repositories, demonstrations, and task packs feed each split.",
  },
  environments: {
    eyebrow: "Executable reinforcement",
    title: "Environments",
    description: "Inspect task tools, sandbox limits, hard gates, and the reward contract used by GRPO.",
  },
  evaluations: {
    eyebrow: "Release proof",
    title: "Evaluations",
    description: "Keep the model benchmark and Fable A/B comparison separate, paired, and held out.",
  },
  artifacts: {
    eyebrow: "Reproducible outputs",
    title: "Artifacts",
    description: "Track locks, datasets, adapters, reports, and packages produced by each stage.",
  },
  runtime: {
    eyebrow: "Local machine",
    title: "Runtime readiness",
    description: "Recorded checks for the one GPU, Python stack, model cache, and isolated task runner.",
  },
};

// Status is always passed explicitly so color never becomes the only signal
// and a static snapshot cannot silently imply progress it did not observe.
function StatusChip({ tone, children }: { tone: StatusTone; children: ReactNode }) {
  return <span className={`status-chip ${tone}`}>{children}</span>;
}

function PageHeading({ view, onPrepare }: { view: ViewId; onPrepare: () => void }) {
  const content = viewTitles[view];
  return (
    <div className="page-heading">
      <div>
        <p className="eyebrow">{content.eyebrow}</p>
        <div className="title-row">
          <h1>{content.title}</h1>
          {view === "overview" && <StatusChip tone="danger">Setup blocked</StatusChip>}
        </div>
        <p className="page-description">{content.description}</p>
      </div>
      <div className="heading-actions">
        <button className="secondary-button" type="button" onClick={onPrepare}>
          CLI actions
        </button>
        <button className="primary-button" type="button" onClick={onPrepare}>
          Prepare run
        </button>
      </div>
    </div>
  );
}

// The pipeline combines preparation and training in execution order. Baseline
// evaluation belongs after the candidate adapter has been frozen.
function PipelinePanel() {
  const completeCount = pipelineStages.filter((stage) => stage.status === "complete").length;
  return (
    <section className="panel pipeline-panel" aria-labelledby="pipeline-heading">
      <div className="panel-header">
        <div>
          <p className="panel-kicker">Execution path</p>
          <h2 id="pipeline-heading">QLoRA → GRPO → proof</h2>
        </div>
        <span className="panel-meta">{completeCount} of {pipelineStages.length} complete</span>
      </div>
      <ol className="stage-list">
        {pipelineStages.map((stage, index) => (
          <li className={`stage-row ${stage.status}`} key={stage.id}>
            <span className="stage-marker" aria-hidden="true">
              {stage.status === "complete" ? "✓" : String(index + 1).padStart(2, "0")}
            </span>
            <div className="stage-copy">
              <div>
                <strong>{stage.label}</strong>
                <span>{stage.meta}</span>
              </div>
              <p>{stage.detail}</p>
            </div>
            <StatusChip
              tone={stage.status === "complete" ? "good" : stage.status === "blocked" ? "danger" : "muted"}
            >
              {stage.status}
            </StatusChip>
          </li>
        ))}
      </ol>
    </section>
  );
}

// Runtime readiness is separate from data readiness: compiled inputs do not
// mean the machine can load a model or execute a sandboxed RL episode.
function RuntimePanel({ onOpenCommands }: { onOpenCommands: () => void }) {
  return (
    <section className="panel runtime-panel" aria-labelledby="runtime-heading">
      <div className="panel-header">
        <div>
          <p className="panel-kicker">Recorded readiness snapshot · not live</p>
          <h2 id="runtime-heading">Local runtime</h2>
        </div>
        <StatusChip tone="danger">Not ready</StatusChip>
      </div>
      <div className="runtime-list">
        {runtimeSnapshot.map((item) => (
          <div className="runtime-row" key={item.label}>
            <span className={`health-dot ${item.tone}`} aria-hidden="true" />
            <div><span>{item.label}</span><small>{item.detail}</small></div>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>
      <button className="panel-action" type="button" onClick={onOpenCommands}>
        Review runtime commands <span aria-hidden="true">→</span>
      </button>
    </section>
  );
}

// Empty telemetry is intentional. Real charts should only appear after the
// shared backend persists timestamped trainer events.
function TelemetryPanel() {
  return (
    <section className="panel telemetry-panel" aria-labelledby="telemetry-heading">
      <div className="panel-header telemetry-header">
        <div>
          <p className="panel-kicker">Run monitor</p>
          <h2 id="telemetry-heading">Training telemetry</h2>
        </div>
        <StatusChip tone="muted">No active run</StatusChip>
      </div>
      <div className="metric-strip" aria-label="Training metrics">
        {[
          ["Step", "—", "of —"],
          ["Loss", "—", "SFT"],
          ["Reward", "—", "GRPO"],
          ["GPU memory", "—", "24 GB max"],
          ["Throughput", "—", "tokens / sec"],
        ].map(([label, value, detail]) => (
          <div className="metric-card" key={label}>
            <span>{label}</span><strong>{value}</strong><small>{detail}</small>
          </div>
        ))}
      </div>
      <div className="empty-chart">
        <div className="chart-grid" aria-hidden="true" />
        <div className="empty-chart-copy">
          <span className="empty-icon" aria-hidden="true">↗</span>
          <strong>Metrics begin when a local training process starts</strong>
          <p>Loss, reward, learning rate, throughput, and GPU usage will share the run timeline.</p>
        </div>
      </div>
    </section>
  );
}

function ActivityPanel() {
  return (
    <section className="panel activity-panel" aria-labelledby="activity-heading">
      <div className="panel-header">
        <div>
          <p className="panel-kicker">Preparation log</p>
          <h2 id="activity-heading">Latest checks</h2>
        </div>
        <span className="panel-meta">Static validation</span>
      </div>
      <ul className="activity-list">
        {preparationEvents.map((event) => (
          <li key={event.label}>
            <span className={`health-dot ${event.tone}`} aria-hidden="true" />
            <div><strong>{event.label}</strong><small>{event.detail}</small></div>
          </li>
        ))}
      </ul>
    </section>
  );
}

// The immutable model contract is kept above run controls because model ID,
// revision, loader, and adapter recipe define what a future run actually means.
function ModelContractPanel() {
  return (
    <section className="panel model-contract" aria-labelledby="model-contract-heading">
      <div className="panel-header">
        <div><p className="panel-kicker">Declared configuration</p><h2 id="model-contract-heading">Model contract</h2></div>
        <StatusChip tone="warning">{projectSnapshot.model.cache}</StatusChip>
      </div>
      <div className="model-contract-grid">
        <article className="wide"><span>Trainable model</span><strong>{projectSnapshot.model.id}</strong><code>{projectSnapshot.model.revision}</code></article>
        <article><span>Loader</span><strong>{projectSnapshot.model.loader}</strong><small>{projectSnapshot.model.state}</small></article>
        <article><span>Adapter recipe</span><strong>{projectSnapshot.recipe.method}</strong><small>{projectSnapshot.recipe.quantization}</small></article>
        <article><span>LoRA rank</span><strong>{projectSnapshot.recipe.rank}</strong><small>Single adapter</small></article>
        <article><span>Context</span><strong>{projectSnapshot.recipe.context}</strong><small>V1 text only</small></article>
        <article className="wide planned-reference"><span>Deferred benchmark reference</span><strong>{projectSnapshot.referenceModel.id}</strong><code>{projectSnapshot.referenceModel.revision}</code></article>
      </div>
    </section>
  );
}

function OverviewView({ onOpenCommands }: { onOpenCommands: () => void }) {
  return (
    <>
      <section className="summary-grid" aria-label="Project status">
        <article><span>Run state</span><strong>Not started</strong><small>No model weights loaded</small></article>
        <article><span>Compiled inputs</span><strong>3</strong><small>1 SFT · 1 RL · 1 evaluation</small></article>
        <article><span>Compute</span><strong>1 × RTX 4090</strong><small>24,564 MiB detected</small></article>
        <article><span>Release proof</span><strong>Blocked</strong><small>Holdout + runner pins</small></article>
      </section>
      <ModelContractPanel />
      <div className="overview-grid">
        <PipelinePanel />
        <RuntimePanel onOpenCommands={onOpenCommands} />
      </div>
      <div className="monitor-grid">
        <TelemetryPanel />
        <ActivityPanel />
      </div>
    </>
  );
}

function RunsView({ onPrepare }: { onPrepare: () => void }) {
  return (
    <div className="runs-layout">
      <section className="panel table-panel" aria-labelledby="runs-table-heading">
        <div className="panel-header table-heading">
          <div><p className="panel-kicker">Job registry</p><h2 id="runs-table-heading">All runs</h2></div>
          <div className="table-tools">
            <label>
              <span className="sr-only">Filter runs</span>
              <input placeholder="Filter runs · backend required" disabled />
            </label>
            <button className="primary-button compact" type="button" onClick={onPrepare}>Prepare via CLI</button>
          </div>
        </div>
        <div className="table-scroll">
          <table>
            <thead><tr><th>Name</th><th>Stage</th><th>Model</th><th>Status</th><th>Progress</th><th>Updated</th></tr></thead>
            <tbody>
              <tr className="empty-table-row">
                <td colSpan={6}>
                  <div className="table-empty-state">
                    <span aria-hidden="true">00</span>
                    <strong>No training runs yet</strong>
                    <p>The recipe is configured, but local runtime blockers must be resolved before launch.</p>
                    <button className="secondary-button" type="button" onClick={onPrepare}>Review launch checklist</button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
      <aside className="panel queue-panel">
        <div className="panel-header"><div><p className="panel-kicker">Intended local policy</p><h2>One visible GPU</h2></div></div>
        <dl className="definition-list">
          <div><dt>Intended capacity</dt><dd>1 training process</dd></div>
          <div><dt>Queue</dt><dd>Unavailable</dd></div>
          <div><dt>Training order</dt><dd>SFT → GRPO</dd></div>
          <div><dt>Scheduler / mutex</dt><dd>Not implemented</dd></div>
        </dl>
        <div className="info-callout warning"><strong>Job control is not connected</strong><p>The CLI does not prevent concurrent processes, and the web layer cannot start, cancel, or resume one. Commands are handed off without simulating success.</p></div>
      </aside>
      <section className="panel run-detail-empty">
        <div className="detail-tabs" aria-label="Run detail sections">
          {['Metrics', 'Rollouts', 'Logs', 'Configuration', 'Checkpoints', 'Evaluation'].map((tab) => <span key={tab}>{tab}</span>)}
        </div>
        <div><strong>Select a run to inspect its execution record</strong><p>Live metrics and rollouts will appear only after the shared backend persists structured run events.</p></div>
      </section>
    </div>
  );
}

// Source partitions stay visible together because training/evaluation identity
// overlap is a release blocker, not a warning to hide in a generated report.
function DataView() {
  return (
    <>
      <section className="summary-grid compact-summary" aria-label="Source summary">
        <article><span>Declared sources</span><strong>5</strong><small>All statically readable</small></article>
        <article><span>Repository files</span><strong>21</strong><small>Content-hashed</small></article>
        <article><span>SFT records</span><strong>1</strong><small>Messages format</small></article>
        <article><span>Executable tasks</span><strong>2</strong><small>Train + evaluation</small></article>
      </section>
      <section className="panel table-panel" aria-labelledby="sources-heading">
        <div className="panel-header"><div><p className="panel-kicker">Source inventory</p><h2 id="sources-heading">Declared inputs</h2></div><StatusChip tone="warning">Compiled with warnings</StatusChip></div>
        <div className="table-scroll">
          <table>
            <thead><tr><th>Source</th><th>Kind</th><th>Partition</th><th>Role</th><th>Location</th><th>Locked identity</th><th>Contents</th><th>State</th></tr></thead>
            <tbody>{sources.map((source) => <tr key={source.id}><td><strong>{source.id}</strong></td><td>{source.kind}</td><td>{source.partition}</td><td>{source.role}</td><td><code>{source.location}</code></td><td><code>{source.identity}</code></td><td>{source.records}</td><td><StatusChip tone={source.tone}>{source.state}</StatusChip></td></tr>)}</tbody>
          </table>
        </div>
      </section>
      <div className="info-callout danger wide-callout"><strong>Repository holdout is not established</strong><p>The training and evaluation fixtures are different folders in the same Git repository. They are useful for authoring, but final evaluation correctly refuses them.</p></div>
    </>
  );
}

function EnvironmentsView() {
  return (
    <div className="environment-layout">
      <section className="panel environment-card">
        <div className="panel-header"><div><p className="panel-kicker">Task environment</p><h2>{environment.id}</h2></div><StatusChip tone="warning">Docker missing</StatusChip></div>
        <dl className="definition-list environment-definition">
          <div><dt>Factory</dt><dd>{environment.factory}</dd></div>
          <div><dt>Image</dt><dd>{environment.image}</dd></div>
          <div><dt>Isolation</dt><dd>{environment.backend}</dd></div>
          <div><dt>Task</dt><dd>{environment.task}</dd></div>
        </dl>
        <div className="tool-section"><span>Policy tools</span><div>{environment.tools.map((tool) => <code key={tool}>{tool}</code>)}</div></div>
      </section>
      <section className="panel limit-card">
        <div className="panel-header"><div><p className="panel-kicker">Execution contract</p><h2>Limits & gates</h2></div></div>
        <dl className="definition-list">{environment.limits.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
        <div className="gate-stack"><span>Build gate</span><span>Regression gate</span><span>Hidden verifier</span></div>
      </section>
      <section className="panel reward-panel">
        <div className="panel-header"><div><p className="panel-kicker">Rubric</p><h2>Reward components</h2></div><span className="panel-meta">Total 100%</span></div>
        <div className="reward-list">{rewardSignals.map((signal) => <div className="reward-row" key={signal.id}><div><strong>{signal.label}</strong><span>{Math.round(signal.weight * 100)}%</span></div><p>{signal.description}</p><i><span style={{ width: `${signal.weight * 100}%` }} /></i></div>)}</div>
      </section>
    </div>
  );
}

function EvaluationsView() {
  return (
    <>
      <div className="evaluation-grid">
        {evaluationSuites.map((suite, index) => (
          <section className="panel evaluation-card" key={suite.label}>
            <div className="panel-header"><div><p className="panel-kicker">Suite {String(index + 1).padStart(2, "0")}</p><h2>{suite.label}</h2></div><StatusChip tone="danger">{suite.status}</StatusChip></div>
            <p className="comparison-copy">{suite.comparison}</p>
            <dl className="definition-list"><div><dt>Planned reference</dt><dd>{suite.plannedReference}</dd></div><div><dt>Decision metric</dt><dd>{suite.metric}</dd></div><div><dt>Task repetitions</dt><dd>3 paired seeds</dd></div><div><dt>Proof completion</dt><dd>{suite.proofRequirement}</dd></div></dl>
            <div className="info-callout warning"><strong>Plan not frozen</strong><p>{suite.planBlocker}</p></div>
          </section>
        ))}
      </div>
      <section className="panel blocker-panel">
        <div className="panel-header"><div><p className="panel-kicker">Fail-closed readiness</p><h2>Evaluation blockers</h2></div><span className="panel-meta">3 plan blockers · 1 proof gap</span></div>
        <ul className="blocker-list">
          <li><span>01</span><div><strong>Candidate adapter does not exist</strong><p>SFT and GRPO have not run.</p></div></li>
          <li><span>02</span><div><strong>Repository holdout overlaps training</strong><p>Replace the authoring fixture with independent held-out project families.</p></div></li>
          <li><span>03</span><div><strong>Runner identities are placeholders</strong><p>Pin model-agent and Fable versions plus orchestration digests.</p></div></li>
        </ul>
        <div className="info-callout warning proof-gap"><strong>Post-plan proof gap</strong><p>One held-out task exists; the winner decision requires at least two unique tasks. This does not prevent writing the immutable plan.</p></div>
      </section>
    </>
  );
}

function ArtifactsView() {
  const artifacts = [
    ["Source lock", ".autotrainer/sources.lock.json", "Generated", "good"],
    ["Experiment lock", ".autotrainer/autotrainer.lock.json", "Stale · relock required", "warning"],
    ["Plan snapshot", ".autotrainer/plan.json", "Blocked", "warning"],
    ["Compile report", ".autotrainer/compiled/compile-report.json", "Generated", "good"],
    ["SFT dataset", ".autotrainer/compiled/sft/train.jsonl", "1 record", "good"],
    ["RL dataset", ".autotrainer/compiled/rl/train.jsonl", "1 task", "good"],
    ["SFT adapter", ".autotrainer/runs/sft", "Missing", "muted"],
    ["GRPO adapter", ".autotrainer/runs/grpo", "Missing", "muted"],
    ["Evaluation report", ".autotrainer/evaluation/<plan-id>/summary.json", "Blocked", "danger"],
    ["Release package", ".autotrainer/packages/polished-frontend-9b", "Unavailable", "muted"],
  ] as const;
  return (
    <section className="panel table-panel" aria-labelledby="artifacts-heading">
      <div className="panel-header"><div><p className="panel-kicker">Project outputs</p><h2 id="artifacts-heading">Artifact registry</h2></div><span className="panel-meta">Local · content-addressed</span></div>
      <div className="table-scroll"><table><thead><tr><th>Artifact</th><th>Project-relative path</th><th>State</th></tr></thead><tbody>{artifacts.map(([label, path, state, tone]) => <tr key={label}><td><strong>{label}</strong></td><td><code>{path}</code></td><td><StatusChip tone={tone}>{state}</StatusChip></td></tr>)}</tbody></table></div>
    </section>
  );
}

function RuntimeView({ onOpenCommands }: { onOpenCommands: () => void }) {
  const packages = [
    ["Python", "3.11.4", "3.11.x", "Ready", "good"],
    ["torch", "2.10.0+cu128", "2.13.0", "Mismatch", "warning"],
    ["transformers", "4.57.6", "5.13.1", "Mismatch", "warning"],
    ["trl", "—", "1.8.0", "Missing", "danger"],
    ["peft", "0.18.1", "0.19.1", "Mismatch", "warning"],
    ["accelerate", "1.10.0", "1.14.0", "Mismatch", "warning"],
    ["datasets", "4.0.0", "5.0.0", "Mismatch", "warning"],
    ["jmespath", "0.10.0", "1.1.0", "Mismatch", "warning"],
    ["bitsandbytes", "—", "0.49.2", "Missing", "danger"],
    ["Docker", "—", "Available", "Missing", "danger"],
  ] as const;
  return (
    <div className="runtime-view-layout">
      <RuntimePanel onOpenCommands={onOpenCommands} />
      <section className="panel table-panel runtime-table">
        <div className="panel-header"><div><p className="panel-kicker">Recorded readiness snapshot · not live</p><h2>Dependency readiness</h2></div></div>
        <div className="table-scroll"><table><thead><tr><th>Component</th><th>Installed</th><th>Expected</th><th>State</th></tr></thead><tbody>{packages.map(([name, installed, expected, state, tone]) => <tr key={name}><td><strong>{name}</strong></td><td>{installed}</td><td>{expected}</td><td><StatusChip tone={tone}>{state}</StatusChip></td></tr>)}</tbody></table></div>
      </section>
    </div>
  );
}

function CommandDrawer({
  open,
  copiedId,
  onClose,
  onCopy,
}: {
  open: boolean;
  copiedId: string | null;
  onClose: () => void;
  onCopy: (command: CommandDefinition) => void;
}) {
  const drawerRef = useRef<HTMLElement>(null);

  // Treat the drawer as a real modal: move focus inside, keep Tab navigation
  // within its controls, close on Escape, and restore the invoking control.
  useEffect(() => {
    if (!open) return;
    const previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const drawer = drawerRef.current;
    const focusable = drawer
      ? Array.from(drawer.querySelectorAll<HTMLElement>("button:not([disabled]), a[href], input:not([disabled])"))
      : [];
    const first = focusable[0];
    const last = focusable.at(-1);
    first?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab" || !first || !last) return;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      previouslyFocused?.focus();
    };
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="drawer-layer">
      <button className="drawer-backdrop" type="button" tabIndex={-1} aria-label="Close launch checklist" onClick={onClose} />
      <aside ref={drawerRef} className="command-drawer" role="dialog" aria-modal="true" aria-labelledby="drawer-heading">
        <div className="drawer-header"><div><p className="panel-kicker">Local handoff</p><h2 id="drawer-heading">Prepare the first run</h2></div><button className="icon-button" type="button" onClick={onClose} aria-label="Close">×</button></div>
        <div className="info-callout danger"><strong>Backend not connected</strong><p>The dashboard will not pretend a job started. Resolve Doctor blockers, then run the same commands the future local backend will call.</p></div>
        <div className="launch-summary">
          <div><span>Project</span><strong>{projectSnapshot.slug}</strong></div>
          <div><span>Recipe</span><strong>{projectSnapshot.recipe.method}</strong></div>
          <div><span>Device</span><strong>1 × RTX 4090</strong></div>
          <div><span>Weights</span><strong>Not downloaded</strong></div>
        </div>
        <div className="command-list">
          {commands.map((command) => (
            <article key={command.id}>
              <div><strong>{command.label}</strong><p>{command.description}</p></div>
              <code>{command.command}</code>
              <button type="button" onClick={() => onCopy(command)}>{copiedId === command.id ? "Copied" : "Copy"}</button>
            </article>
          ))}
        </div>
        <button className="primary-button disabled-action" type="button" disabled>Start SFT · resolve runtime first</button>
      </aside>
    </div>
  );
}

export default function App() {
  // Navigation and the CLI drawer are local presentation state. Project,
  // runtime, and run state remain read-only until the backend contract exists.
  const [activeView, setActiveView] = useState<ViewId>("overview");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const openDrawer = useCallback(() => setDrawerOpen(true), []);
  const closeDrawer = useCallback(() => setDrawerOpen(false), []);

  // Until the shared local backend exists, browser actions hand operators the
  // exact reproducible command instead of fabricating a queued/running state.
  const copyCommand = async (command: CommandDefinition) => {
    try {
      await navigator.clipboard.writeText(command.command);
      setCopiedId(command.id);
      window.setTimeout(() => setCopiedId(null), 1600);
    } catch {
      setCopiedId(null);
    }
  };

  const renderView = () => {
    switch (activeView) {
      case "runs": return <RunsView onPrepare={openDrawer} />;
      case "data": return <DataView />;
      case "environments": return <EnvironmentsView />;
      case "evaluations": return <EvaluationsView />;
      case "artifacts": return <ArtifactsView />;
      case "runtime": return <RuntimeView onOpenCommands={openDrawer} />;
      default: return <OverviewView onOpenCommands={openDrawer} />;
    }
  };

  return (
    <>
    <div className="console-shell" inert={drawerOpen ? true : undefined}>
      <aside className="sidebar">
        <div className="brand-row"><span className="brand-mark" aria-hidden="true">A</span><div><strong>AutoTrainer</strong><small>Local control plane</small></div></div>
        <div className="project-switcher"><span className="project-avatar" aria-hidden="true">PF</span><div><strong>{projectSnapshot.name}</strong><small>{projectSnapshot.slug}</small></div><span className="project-state">Snapshot</span></div>
        <nav className="primary-nav" aria-label="Project navigation">
          {navigation.map((item) => (
            <button className={activeView === item.id ? "active" : ""} type="button" key={item.id} onClick={() => setActiveView(item.id)} aria-label={item.label} aria-current={activeView === item.id ? "page" : undefined}>
              <span className="nav-icon" aria-hidden="true">{item.short}</span><span className="nav-label">{item.label}</span>{item.count && <small>{item.count}</small>}
            </button>
          ))}
        </nav>
        <div className="sidebar-compute">
          <div><span className="health-dot good" aria-hidden="true" /><strong>RTX 4090</strong></div>
          <p>24 GB · 1 local device</p>
          <div className="capacity-track"><span /></div>
          <small>Snapshot · no model loaded</small>
        </div>
      </aside>

      <main className="console-main">
        <header className="topbar">
          <div className="breadcrumbs"><span>Projects</span><b>/</b><strong>{projectSnapshot.slug}</strong><StatusChip tone="info">{projectSnapshot.mode}</StatusChip></div>
          <div className="topbar-actions"><span className="config-source">{projectSnapshot.configPath}</span><button className="icon-button" type="button" aria-label="Open command checklist" onClick={openDrawer}>⌘</button></div>
        </header>
        <div className="page-content">
          <PageHeading view={activeView} onPrepare={openDrawer} />
          {renderView()}
        </div>
        <footer className="console-footer"><span>autotrainer.yaml is the source of truth</span><span>Configured ≠ downloaded ≠ trained ≠ verified</span></footer>
      </main>

    </div>
    <CommandDrawer open={drawerOpen} copiedId={copiedId} onClose={closeDrawer} onCopy={copyCommand} />
    </>
  );
}
