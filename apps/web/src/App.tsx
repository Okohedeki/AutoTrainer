import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getBackendHealth, getTrainingJob, type BackendHealth } from "./api";
import HistoryReviewPanel from "./HistoryReviewPanel";
import ModelSetupPanel from "./ModelSetupPanel";
import PreparePanel from "./PreparePanel";
import SourceSetupPanel from "./SourceSetupPanel";
import TrainingMonitorPanel from "./TrainingMonitorPanel";

const WALKTHROUGH_STORAGE_KEY = "autotrainer.walkthrough.v2";

type ViewId = "setup" | "training";
type WalkthroughStep = { target: string; label: string; title: string; body: string };

// The walkthrough points at real controls in the operating console. It does
// not create a separate demo path or describe features that are not connected.
const walkthroughSteps: WalkthroughStep[] = [
  {
    target: '[data-tour="model"]',
    label: "1 of 3",
    title: "Choose the model",
    body: "Pick a small model that fits your machine. AutoTrainer downloads the exact version it will use.",
  },
  {
    target: '[data-tour="sources"]',
    label: "2 of 3",
    title: "Add the work",
    body: "Paste a GitHub repository or a local path. Add accepted examples or practice tasks when you have them.",
  },
  {
    target: '[data-tour="prepare"]',
    label: "3 of 3",
    title: "Prepare training",
    body: "One check finds what is ready, what is missing, and the useful next step. Training never starts by accident.",
  },
];

function Walkthrough({
  stepIndex,
  onBack,
  onNext,
  onClose,
}: {
  stepIndex: number;
  onBack: () => void;
  onNext: () => void;
  onClose: () => void;
}) {
  const dialogRef = useRef<HTMLElement>(null);
  const step = walkthroughSteps[stepIndex];
  const finalStep = stepIndex === walkthroughSteps.length - 1;

  useEffect(() => {
    document.querySelector<HTMLElement>(step.target)?.scrollIntoView({ block: "center" });
    dialogRef.current?.querySelector<HTMLElement>("button")?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [step, onClose]);

  return (
    <div className="walkthrough-layer">
      <div className="walkthrough-shade" aria-hidden="true" />
      <aside ref={dialogRef} className="walkthrough-card" role="dialog" aria-modal="true" aria-labelledby="walkthrough-title" aria-describedby="walkthrough-body">
        <span className="walkthrough-progress">{step.label}</span>
        <h2 id="walkthrough-title">{step.title}</h2>
        <p id="walkthrough-body">{step.body}</p>
        <div className="walkthrough-actions">
          <button className="text-button" type="button" onClick={onClose}>Skip</button>
          <div>
            {stepIndex > 0 && <button className="secondary-button" type="button" onClick={onBack}>Back</button>}
            <button className="primary-button" type="button" onClick={onNext}>{finalStep ? "Start setup" : "Next"}</button>
          </div>
        </div>
      </aside>
    </div>
  );
}

export default function App() {
  const restartButtonRef = useRef<HTMLButtonElement>(null);
  const [activeView, setActiveView] = useState<ViewId>("setup");
  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [backendConnected, setBackendConnected] = useState(false);
  const [sourceRevision, setSourceRevision] = useState(0);
  const [projectRevision, setProjectRevision] = useState(0);
  const [trainingActive, setTrainingActive] = useState(false);
  const [walkthroughStep, setWalkthroughStep] = useState<number | null>(() => {
    try {
      return window.localStorage.getItem(WALKTHROUGH_STORAGE_KEY) ? null : 0;
    } catch {
      return 0;
    }
  });
  const walkthroughOpen = walkthroughStep !== null;

  useEffect(() => {
    let stopped = false;
    const refresh = async () => {
      try {
        const [nextHealth, job] = await Promise.all([getBackendHealth(), getTrainingJob()]);
        if (stopped) return;
        setHealth(nextHealth);
        setBackendConnected(true);
        setTrainingActive(job.status === "queued" || job.status === "running");
      } catch {
        if (!stopped) setBackendConnected(false);
      }
    };
    void refresh();
    const interval = window.setInterval(refresh, 2_000);
    return () => {
      stopped = true;
      window.clearInterval(interval);
    };
  }, []);

  const projectName = useMemo(() => {
    const parts = health?.config.replaceAll("\\", "/").split("/") ?? [];
    return parts.at(-2) || "Local project";
  }, [health]);

  const rememberWalkthrough = useCallback(() => {
    try {
      window.localStorage.setItem(WALKTHROUGH_STORAGE_KEY, "complete");
    } catch {
      // Locked-down browsers may not retain completion; setup still works.
    }
  }, []);

  const closeWalkthrough = useCallback(() => {
    rememberWalkthrough();
    setWalkthroughStep(null);
    window.requestAnimationFrame(() => restartButtonRef.current?.focus());
  }, [rememberWalkthrough]);

  const nextWalkthroughStep = useCallback(() => {
    setWalkthroughStep((current) => {
      if (current === null || current === walkthroughSteps.length - 1) {
        rememberWalkthrough();
        return null;
      }
      return current + 1;
    });
  }, [rememberWalkthrough]);

  const sourcesChanged = useCallback(() => {
    setSourceRevision((value) => value + 1);
    setProjectRevision((value) => value + 1);
  }, []);

  const projectChanged = useCallback(() => {
    setProjectRevision((value) => value + 1);
  }, []);

  const openView = (view: ViewId) => {
    setActiveView(view);
    window.scrollTo({ top: 0, behavior: "auto" });
  };

  return (
    <>
      <div className="console-shell" inert={walkthroughOpen ? true : undefined}>
        <aside className="sidebar">
          <div className="brand-row">
            <span className="brand-mark" aria-hidden="true">A</span>
            <div><strong>AutoTrainer</strong><small>Local training console</small></div>
          </div>

          <div className="project-context">
            <span className="project-avatar" aria-hidden="true">{projectName.slice(0, 2).toUpperCase()}</span>
            <div><strong>{projectName}</strong><small>autotrainer.yaml</small></div>
            <span className="project-state">Local</span>
          </div>

          <nav className="primary-nav" aria-label="Project navigation">
            <button className={activeView === "setup" ? "active" : ""} type="button" onClick={() => openView("setup")} aria-current={activeView === "setup" ? "page" : undefined}>
              <span className="nav-icon" aria-hidden="true">01</span><span className="nav-label">Setup</span>
            </button>
            <button className={activeView === "training" ? "active" : ""} type="button" onClick={() => openView("training")} aria-current={activeView === "training" ? "page" : undefined}>
              <span className="nav-icon" aria-hidden="true">02</span><span className="nav-label">Training</span>{trainingActive && <small>live</small>}
            </button>
          </nav>

          <div className="sidebar-runtime">
            <div><span className={`health-dot ${backendConnected ? "good" : "danger"}`} aria-hidden="true" /><strong>Local backend</strong></div>
            <p>{backendConnected ? "Connected on this machine" : "Not connected"}</p>
          </div>
        </aside>

        <div className="console-main">
          <header className="topbar">
            <div className="breadcrumbs"><span>Projects</span><b>/</b><strong>{projectName}</strong><span className={`status-chip ${backendConnected ? "good" : "danger"}`}>{backendConnected ? "connected" : "offline"}</span></div>
            <div className="topbar-actions">
              {health?.config && <code className="config-source">{health.config}</code>}
              <button ref={restartButtonRef} className="walkthrough-restart" type="button" onClick={() => { setActiveView("setup"); setWalkthroughStep(0); }}>Walkthrough</button>
            </div>
          </header>

          <main className="page-content">
            <header className="page-heading">
              <div>
                <p className="eyebrow">{activeView === "setup" ? "Project setup" : "Local execution"}</p>
                <div className="title-row"><h1>{activeView === "setup" ? "Build the training run" : "Training run"}</h1>{trainingActive && <span className="status-chip info">running</span>}</div>
                <p className="page-description">{activeView === "setup"
                  ? "Choose the exact model, add your work, review useful examples, and prove this machine is ready."
                  : "Watch the durable local job record, completed stages, real trainer metrics, and output paths."}</p>
              </div>
            </header>

            {activeView === "setup" ? (
              <>
                <div className="workspace-grid" aria-label="Training setup">
                  <div className="workspace-main">
                    <ModelSetupPanel onModelChanged={projectChanged} disabled={trainingActive} />
                    <SourceSetupPanel onSourcesChanged={sourcesChanged} disabled={trainingActive} />
                  </div>
                  <PreparePanel revision={projectRevision} onTrainingActiveChange={setTrainingActive} />
                </div>
                <HistoryReviewPanel refreshKey={sourceRevision} onHistoryChanged={projectChanged} disabled={trainingActive} />
              </>
            ) : (
              <TrainingMonitorPanel onOpenSetup={() => openView("setup")} />
            )}
          </main>
        </div>
      </div>

      {walkthroughStep !== null && (
        <Walkthrough
          stepIndex={walkthroughStep}
          onBack={() => setWalkthroughStep((current) => Math.max(0, (current ?? 1) - 1))}
          onNext={nextWalkthroughStep}
          onClose={closeWalkthrough}
        />
      )}
    </>
  );
}
