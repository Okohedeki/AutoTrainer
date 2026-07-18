import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getBackendHealth,
  getProjects,
  getTrainingJob,
  type BackendHealth,
  type ProjectsWorkspace,
} from "./api";
import EvaluationMonitorPanel from "./EvaluationMonitorPanel";
import GrpoEvidencePanel from "./GrpoEvidencePanel";
import HistoryReviewPanel from "./HistoryReviewPanel";
import ModelSetupPanel from "./ModelSetupPanel";
import ProjectsPanel from "./ProjectsPanel";
import ServePanel from "./ServePanel";
import SourceSetupPanel from "./SourceSetupPanel";
import TrainingMonitorPanel from "./TrainingMonitorPanel";

const WALKTHROUGH_STORAGE_KEY = "autotrainer.walkthrough.v3";

type ViewId = "projects" | "data" | "train" | "evaluate" | "serve";
type WalkthroughStep = { view: ViewId; target: string; label: string; title: string; body: string };

// The first run follows the real product lifecycle. Moving between screens is
// part of the walkthrough, so every callout points at an operational control.
const walkthroughSteps: WalkthroughStep[] = [
  { view: "projects", target: '[data-tour="projects"]', label: "1 of 5", title: "One specialist per project", body: "Create a project for the kind of work your local model should master. Each project keeps its own data, runs, proof, and endpoint." },
  { view: "data", target: '[data-tour="model"]', label: "2 of 5", title: "Define the learning material", body: "Choose a supported Hugging Face model, download its pinned revision, then say exactly how each repository may be used." },
  { view: "train", target: '[data-tour="train"]', label: "3 of 5", title: "Train on one GPU", body: "Start the run here. Watch real loss, reward, and rubric signals as the local trainer records them." },
  { view: "evaluate", target: '[data-tour="evaluate"]', label: "4 of 5", title: "Prove the frozen model", body: "Evaluation does not train. Watch each held-out trial move through generation, trusted checks, and scoring." },
  { view: "serve", target: '[data-tour="serve"]', label: "5 of 5", title: "Call your specialist", body: "Load the completed adapter behind a local OpenAI-compatible endpoint and send a real test request." },
];

function Walkthrough({ stepIndex, onBack, onNext, onClose }: { stepIndex: number; onBack: () => void; onNext: () => void; onClose: () => void }) {
  const dialogRef = useRef<HTMLElement>(null);
  const step = walkthroughSteps[stepIndex];
  const finalStep = stepIndex === walkthroughSteps.length - 1;

  useEffect(() => {
    window.requestAnimationFrame(() => document.querySelector<HTMLElement>(step.target)?.scrollIntoView({ block: "center" }));
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
          <div>{stepIndex > 0 && <button className="secondary-button" type="button" onClick={onBack}>Back</button>}<button className="primary-button" type="button" onClick={onNext}>{finalStep ? "Open AutoTrainer" : "Next"}</button></div>
        </div>
      </aside>
    </div>
  );
}

export default function App() {
  const restartButtonRef = useRef<HTMLButtonElement>(null);
  const pageTitleRef = useRef<HTMLHeadingElement>(null);
  const [activeView, setActiveView] = useState<ViewId>("projects");
  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [projects, setProjects] = useState<ProjectsWorkspace | null>(null);
  const [backendConnected, setBackendConnected] = useState(false);
  const [sourceRevision, setSourceRevision] = useState(0);
  const [projectRevision, setProjectRevision] = useState(0);
  const [projectScopeRevision, setProjectScopeRevision] = useState(0);
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
    let timer = 0;
    const refresh = async () => {
      try {
        const [nextHealth, nextProjects, job] = await Promise.all([
          getBackendHealth(),
          getProjects().catch(() => null),
          getTrainingJob().catch(() => null),
        ]);
        if (stopped) return;
        setHealth(nextHealth);
        if (nextProjects) setProjects(nextProjects);
        if (job) setTrainingActive(job.status === "queued" || job.status === "running");
        setBackendConnected(true);
      } catch {
        if (!stopped) setBackendConnected(false);
      } finally {
        if (!stopped) timer = window.setTimeout(() => void refresh(), 2_000);
      }
    };
    void refresh();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, []);

  const activeProject = projects?.projects.find((project) => project.id === projects.active_id);
  const projectName = useMemo(() => {
    if (activeProject?.name) return activeProject.name;
    const parts = health?.config.replaceAll("\\", "/").split("/") ?? [];
    return parts.at(-2) || "Local project";
  }, [activeProject?.name, health]);

  const rememberWalkthrough = useCallback(() => {
    try {
      window.localStorage.setItem(WALKTHROUGH_STORAGE_KEY, "complete");
    } catch {
      // Setup remains usable when a locked-down browser cannot save completion.
    }
  }, []);
  const closeWalkthrough = useCallback(() => {
    rememberWalkthrough();
    setWalkthroughStep(null);
    window.requestAnimationFrame(() => restartButtonRef.current?.focus());
  }, [rememberWalkthrough]);
  const nextWalkthroughStep = useCallback(() => {
    if (walkthroughStep === null) return;
    if (walkthroughStep === walkthroughSteps.length - 1) return closeWalkthrough();
    setWalkthroughStep(walkthroughStep + 1);
  }, [closeWalkthrough, walkthroughStep]);

  useEffect(() => {
    if (walkthroughStep !== null) setActiveView(walkthroughSteps[walkthroughStep].view);
  }, [walkthroughStep]);

  const sourcesChanged = useCallback(() => {
    setSourceRevision((value) => value + 1);
    setProjectRevision((value) => value + 1);
  }, []);
  const projectChanged = useCallback(() => setProjectRevision((value) => value + 1), []);

  const projectsChanged = useCallback((next: ProjectsWorkspace) => {
    if (projects?.active_id !== next.active_id) {
      setProjectScopeRevision((value) => value + 1);
      setSourceRevision((value) => value + 1);
      setProjectRevision((value) => value + 1);
      setActiveView("data");
    }
    setProjects(next);
  }, [projects?.active_id]);

  const openView = (view: ViewId, focusTitle = false) => {
    setActiveView(view);
    window.scrollTo({ top: 0, behavior: "auto" });
    if (focusTitle) window.requestAnimationFrame(() => pageTitleRef.current?.focus());
  };

  const viewCopy: Record<ViewId, { eyebrow: string; title: string; description: string }> = {
    projects: { eyebrow: "Local workspaces", title: "Projects", description: "Create one specialist at a time. Its model, data, training evidence, evaluations, and endpoint stay together." },
    data: { eyebrow: "Define the work", title: "Data", description: "Choose the exact base model and say what every GitHub repository or local file contributes." },
    train: { eyebrow: "Change the adapter", title: "Train", description: "Run the complete local training path and watch its real loss, reward, rubric, and durable events." },
    evaluate: { eyebrow: "Freeze and measure", title: "Evaluate", description: "Generate held-out work, verify it locally, and watch trusted scores arrive. No weights change here." },
    serve: { eyebrow: "Use the result", title: "Serve", description: "Make the completed adapter callable through a local OpenAI-compatible endpoint." },
  };
  const nav: Array<{ id: ViewId; label: string }> = [
    { id: "projects", label: "Projects" }, { id: "data", label: "Data" }, { id: "train", label: "Train" }, { id: "evaluate", label: "Evaluate" }, { id: "serve", label: "Serve" },
  ];

  return (
    <>
      <div className="console-shell" inert={walkthroughOpen ? true : undefined}>
        <aside className="sidebar">
          <div className="brand-row"><span className="brand-mark" aria-hidden="true">A</span><div><strong>AutoTrainer</strong><small>Local specialist console</small></div></div>
          <button className="project-context" type="button" onClick={() => openView("projects")}>
            <span className="project-avatar" aria-hidden="true">{projectName.slice(0, 2).toUpperCase()}</span><span><strong>{projectName}</strong><small>{activeProject?.config_path || "autotrainer.yaml"}</small></span><span className="project-state">Local</span>
          </button>
          <nav className="primary-nav" aria-label="Project lifecycle">
            {nav.map((item, index) => <button key={item.id} className={activeView === item.id ? "active" : ""} type="button" onClick={() => openView(item.id)} aria-current={activeView === item.id ? "page" : undefined}><span className="nav-icon" aria-hidden="true">{String(index + 1).padStart(2, "0")}</span><span className="nav-label">{item.label}</span>{item.id === "train" && trainingActive && <small>live</small>}</button>)}
          </nav>
          <div className="sidebar-runtime"><div><span className={`health-dot ${backendConnected ? "good" : "danger"}`} aria-hidden="true" /><strong>Local backend</strong></div><p>{backendConnected ? "Connected on this machine" : "Not connected"}</p></div>
        </aside>

        <div className="console-main">
          <header className="topbar">
            <div className="breadcrumbs"><span>{projectName}</span><b>/</b><strong>{viewCopy[activeView].title}</strong><span className={`status-chip ${backendConnected ? "good" : "danger"}`}>{backendConnected ? "connected" : "offline"}</span></div>
            <div className="topbar-actions"><button className="secondary-button new-project-button" type="button" onClick={() => openView("projects")}>New project</button><button ref={restartButtonRef} className="walkthrough-restart" type="button" onClick={() => setWalkthroughStep(0)}>Walkthrough</button></div>
          </header>

          <main className="page-content">
            <header className="page-heading"><div><p className="eyebrow">{viewCopy[activeView].eyebrow}</p><div className="title-row"><h1 ref={pageTitleRef} tabIndex={-1}>{viewCopy[activeView].title}</h1>{activeView === "train" && trainingActive && <span className="status-chip info">running</span>}</div><p className="page-description">{viewCopy[activeView].description}</p></div></header>

            {activeView === "projects" ? (
              <div data-tour="projects"><ProjectsPanel workspace={projects} disabled={trainingActive} onWorkspaceChanged={projectsChanged} /></div>
            ) : activeView === "data" ? (
              <div className="data-workspace" key={`data-${projects?.active_id}-${projectScopeRevision}`}>
                <ModelSetupPanel onModelChanged={projectChanged} disabled={trainingActive} />
                <SourceSetupPanel onSourcesChanged={sourcesChanged} disabled={trainingActive} />
                <GrpoEvidencePanel context="data" refreshKey={sourceRevision} />
                <HistoryReviewPanel refreshKey={sourceRevision} onHistoryChanged={projectChanged} disabled={trainingActive} />
              </div>
            ) : activeView === "train" ? (
              <TrainingMonitorPanel key={`train-${projects?.active_id}-${projectScopeRevision}`} revision={projectRevision} onOpenData={() => openView("data", true)} onTrainingActiveChange={setTrainingActive} />
            ) : activeView === "evaluate" ? (
              <EvaluationMonitorPanel key={`evaluate-${projects?.active_id}-${projectScopeRevision}`} onOpenData={() => openView("data", true)} />
            ) : (
              <ServePanel key={`serve-${projects?.active_id}-${projectScopeRevision}`} />
            )}
          </main>
        </div>
      </div>

      {walkthroughStep !== null && <Walkthrough stepIndex={walkthroughStep} onBack={() => setWalkthroughStep((current) => Math.max(0, (current ?? 1) - 1))} onNext={nextWalkthroughStep} onClose={closeWalkthrough} />}
    </>
  );
}
