import { useCallback, useEffect, useRef, useState } from "react";
import HistoryReviewPanel from "./HistoryReviewPanel";
import ModelSetupPanel from "./ModelSetupPanel";
import PreparePanel from "./PreparePanel";
import SourceSetupPanel from "./SourceSetupPanel";

const WALKTHROUGH_STORAGE_KEY = "autotrainer.walkthrough.v2";

type WalkthroughStep = {
  target: string;
  label: string;
  title: string;
  body: string;
};

// Onboarding mirrors the only three decisions on the page. It explains the
// product without creating a second demo flow that can drift from real setup.
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
      <aside
        ref={dialogRef}
        className="walkthrough-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="walkthrough-title"
        aria-describedby="walkthrough-body"
      >
        <span className="walkthrough-progress">{step.label}</span>
        <h2 id="walkthrough-title">{step.title}</h2>
        <p id="walkthrough-body">{step.body}</p>
        <div className="walkthrough-actions">
          <button className="text-button" type="button" onClick={onClose}>Skip</button>
          <div>
            {stepIndex > 0 && <button className="secondary-button" type="button" onClick={onBack}>Back</button>}
            <button className="primary-button" type="button" onClick={onNext}>
              {finalStep ? "Start setup" : "Next"}
            </button>
          </div>
        </div>
      </aside>
    </div>
  );
}

export default function App() {
  const restartButtonRef = useRef<HTMLButtonElement>(null);
  const [walkthroughStep, setWalkthroughStep] = useState<number | null>(() => {
    try {
      return window.localStorage.getItem(WALKTHROUGH_STORAGE_KEY) ? null : 0;
    } catch {
      return 0;
    }
  });
  const walkthroughOpen = walkthroughStep !== null;

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

  return (
    <>
      <div className="app-shell" inert={walkthroughOpen ? true : undefined}>
        <header className="site-header">
          <a className="brand" href="#top" aria-label="AutoTrainer home">
            <span aria-hidden="true">A</span>
            <strong>AutoTrainer</strong>
          </a>
          <button
            ref={restartButtonRef}
            className="text-button walkthrough-restart"
            type="button"
            onClick={() => setWalkthroughStep(0)}
          >
            Walkthrough
          </button>
        </header>

        <main id="top">
          <section className="hero" aria-labelledby="page-title">
            <p className="eyebrow">Local model training</p>
            <h1 id="page-title">Make a small model excellent at your work</h1>
            <p>Choose a model, show it the work that matters, and prepare a training path your machine can run.</p>
          </section>

          <div className="setup-flow" aria-label="Training setup">
            <ModelSetupPanel />
            <SourceSetupPanel />
            <HistoryReviewPanel />
            <PreparePanel />
          </div>

          <section className="proof-note" aria-labelledby="proof-heading">
            <p className="panel-kicker">After training</p>
            <h2 id="proof-heading">Prove the specialist is better</h2>
            <p>Compare it with the original model on work neither version saw during training.</p>
          </section>
        </main>

        <footer className="site-footer">
          <span>AutoTrainer</span>
          <span>Your files stay on this machine.</span>
        </footer>
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
