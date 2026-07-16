import { useState } from "react";
import { ApiClientError, prepareProject, type PreparationResult } from "./api";

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

// Preparation folds the old validate, scan, compile, plan, and Doctor sequence
// into one human action. Detailed evidence remains in the API response for the
// agent CLI, while this panel shows only the next decision.
export default function PreparePanel() {
  const [result, setResult] = useState<PreparationResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  const recipe = result ? recipeLabels[result.recipe] : null;

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
    </section>
  );
}
