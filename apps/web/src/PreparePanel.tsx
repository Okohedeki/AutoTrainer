import { useEffect, useState } from "react";
import {
  ApiClientError,
  getTrainingJob,
  prepareProject,
  startTraining,
  type PreparationResult,
  type TrainingJob,
} from "./api";

const recipeLabels: Record<PreparationResult["recipe"], { title: string; detail: string }> = {
  teach: {
    title: "Teach from accepted work",
    detail: "The model will learn from examples of work you already approved.",
  },
  practice: {
    title: "Practice against tests",
    detail: "The model can already attempt your tasks, so verified practice is the useful next step.",
  },
  both: {
    title: "Teach, then practice",
    detail: "Start with accepted work, then improve the same adapter against executable tests.",
  },
  needs_training_data: {
    title: "Add a learning signal",
    detail: "A repository provides context; accepted examples or executable tasks are what change the model.",
  },
};

const trainingStageLabels: Record<NonNullable<TrainingJob["stage"]>, string> = {
  prepare: "Preparing files",
  sft: "Teaching from examples",
  grpo: "Practicing against tests",
};

// Preparation folds the old validate, scan, compile, plan, and Doctor sequence
// into one human action. Detailed evidence remains in the API response for the
// agent CLI, while this panel shows only the next decision.
export default function PreparePanel({ revision = 0 }: { revision?: number }) {
  const [result, setResult] = useState<PreparationResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trainingJob, setTrainingJob] = useState<TrainingJob | null>(null);
  const [trainingBusy, setTrainingBusy] = useState(false);
  const [trainingError, setTrainingError] = useState<string | null>(null);

  useEffect(() => {
    // Preparation is evidence about one exact project snapshot. Hiding it on
    // mutation prevents the GUI from displaying a stale Ready/Start state.
    setResult(null);
    setError(null);
  }, [revision]);

  useEffect(() => {
    const controller = new AbortController();
    getTrainingJob(controller.signal)
      .then(setTrainingJob)
      .catch(() => {
        // Preparation remains usable if there is no prior training job to show.
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (trainingJob?.status !== "queued" && trainingJob?.status !== "running") return;

    let stopped = false;
    // Jobs run outside the page. Polling the shared record keeps this small
    // control truthful without inventing progress or keeping a browser request open.
    const interval = window.setInterval(() => {
      getTrainingJob()
        .then((next) => {
          if (stopped) return;
          setTrainingJob(next);
          setTrainingError(null);
        })
        .catch((reason: unknown) => {
          if (stopped) return;
          setTrainingError(reason instanceof Error ? reason.message : "Training status could not be refreshed.");
        });
    }, 2_000);

    return () => {
      stopped = true;
      window.clearInterval(interval);
    };
  }, [trainingJob?.status]);

  const prepare = async () => {
    setBusy(true);
    setError(null);
    try {
      setResult(await prepareProject());
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "AutoTrainer could not prepare this project.");
    } finally {
      setBusy(false);
    }
  };

  const start = async () => {
    setTrainingBusy(true);
    setTrainingError(null);
    try {
      setTrainingJob(await startTraining());
    } catch (reason) {
      setTrainingError(reason instanceof ApiClientError ? reason.message : "Training could not start.");
    } finally {
      setTrainingBusy(false);
    }
  };

  const recipe = result ? recipeLabels[result.recipe] : null;
  const trainingActive = trainingJob?.status === "queued" || trainingJob?.status === "running";
  const showTrainingControl = result?.status === "ready" || Boolean(trainingJob && trainingJob.status !== "idle");
  const outputDirectory = trainingJob?.result?.stages
    .map((stage) => stage.output_dir)
    .filter((value): value is string => Boolean(value))
    .at(-1);

  return (
    <section className="panel setup-step prepare-panel" aria-labelledby="prepare-heading" data-tour="prepare">
      <header className="step-heading">
        <span className="step-number" aria-hidden="true">3</span>
        <div className="prepare-copy">
          <h2 id="prepare-heading">Prepare training</h2>
          <p>Check the model, your work, the learning data, and this machine in one step.</p>
        </div>
        <button className="primary-button prepare-button" type="button" onClick={() => void prepare()} disabled={busy}>
          {busy ? "Preparing..." : result ? "Check again" : "Prepare training"}
        </button>
      </header>

      {error && <div className="source-error prepare-error" role="alert">{error}</div>}

      {result && recipe && (
        <div className="prepare-result" aria-live="polite">
          <div className="prepare-recipe">
            <span className={`status-chip ${result.status === "ready" ? "good" : "warning"}`}>
              {result.status === "ready" ? "Ready" : "Action needed"}
            </span>
            <div><strong>{recipe.title}</strong><p>{recipe.detail}</p></div>
          </div>
          <ol className="prepare-steps">
            {result.steps.map((step) => (
              <li className={step.status} key={step.id}>
                <span aria-hidden="true">{step.status === "complete" ? "OK" : step.status === "blocked" ? "!" : "-"}</span>
                <div><strong>{step.label}</strong><small>{step.status}</small></div>
              </li>
            ))}
          </ol>
          {result.next_action && (
            <div className="next-action">
              <span>Do this next</span>
              <strong>{result.next_action.title}</strong>
              <p>{result.next_action.detail}</p>
            </div>
          )}

        </div>
      )}

      {showTrainingControl && (
        <div className="training-control" aria-live="polite">
          {(!trainingJob || trainingJob.status === "idle") && result?.status === "ready" && (
            <div className="training-start">
              <div><strong>Ready to train</strong><p>Training uses the path shown above and runs as one local job.</p></div>
              <button className="primary-button" type="button" onClick={() => void start()} disabled={trainingBusy}>
                {trainingBusy ? "Starting..." : "Start training"}
              </button>
            </div>
          )}

          {trainingActive && trainingJob && (
            <div className="training-state active" role="status">
              <span className="training-marker" aria-hidden="true" />
              <div>
                <strong>{trainingJob.stage ? trainingStageLabels[trainingJob.stage] : "Waiting to start"}</strong>
                <p>{trainingJob.message}</p>
              </div>
            </div>
          )}

          {trainingJob?.status === "completed" && (
            <div className="training-state complete" role="status">
              <span aria-hidden="true">OK</span>
              <div>
                <strong>Training output ready</strong>
                <p>{trainingJob.message || "Your trained model output is ready."}</p>
                {outputDirectory && <code className="training-output">{outputDirectory}</code>}
              </div>
              {result?.status === "ready" && (
                <button className="secondary-button" type="button" onClick={() => void start()} disabled={trainingBusy}>
                  {trainingBusy ? "Starting..." : "Train again"}
                </button>
              )}
            </div>
          )}

          {(trainingJob?.status === "failed" || trainingJob?.status === "interrupted") && (
            <div className="training-state failed" role="alert">
              <span aria-hidden="true">!</span>
              <div>
                <strong>{trainingJob.status === "interrupted" ? "Training was interrupted" : "Training stopped"}</strong>
                <p>{trainingJob.message}</p>
              </div>
              {result?.status === "ready" && (
                <button className="secondary-button" type="button" onClick={() => void start()} disabled={trainingBusy}>
                  {trainingBusy ? "Retrying..." : "Retry training"}
                </button>
              )}
            </div>
          )}

          {trainingError && <div className="source-error training-error" role="alert">{trainingError}</div>}
        </div>
      )}
    </section>
  );
}
