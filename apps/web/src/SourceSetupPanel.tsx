import { useEffect, useState } from "react";
import {
  ApiClientError,
  addProjectSource,
  getProjectSources,
  removeProjectSource,
  type ProjectSource,
} from "./api";

const purposeLabels: Record<ProjectSource["purpose"], string> = {
  work: "Work repository",
  examples: "Accepted examples",
  tasks: "Practice tasks",
};

// One field is deliberate: the backend infers whether the user supplied a
// GitHub repository, local Git checkout, demonstration file, or task pack.
// Those distinctions remain available in YAML without becoming setup chores.
export default function SourceSetupPanel() {
  const [sources, setSources] = useState<ProjectSource[]>([]);
  const [value, setValue] = useState("");
  const [connected, setConnected] = useState<boolean | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getProjectSources(controller.signal)
      .then((next) => {
        setSources(next);
        setConnected(true);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        setConnected(false);
        setError(reason instanceof Error ? reason.message : "AutoTrainer is not connected.");
      });
    return () => controller.abort();
  }, []);

  const addSource = async () => {
    const nextValue = value.trim();
    if (!nextValue) return;
    setBusy("add");
    setError(null);
    try {
      setSources(await addProjectSource(nextValue));
      setValue("");
      setConnected(true);
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "AutoTrainer could not add that source.");
    } finally {
      setBusy(null);
    }
  };

  const removeSource = async (source: ProjectSource) => {
    setBusy(source.id);
    setError(null);
    try {
      setSources(await removeProjectSource(source.id));
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "AutoTrainer could not remove that source.");
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="panel setup-step source-setup" aria-labelledby="source-setup-heading" data-tour="sources">
      <header className="step-heading source-setup-header">
        <span className="step-number" aria-hidden="true">2</span>
        <div>
          <h2 id="source-setup-heading">Add your work</h2>
          <p>Paste a GitHub repository or a local path. AutoTrainer identifies what you added.</p>
        </div>
        <span className={`status-chip ${connected === false ? "danger" : sources.length ? "good" : "muted"}`}>
          {connected === false ? "Backend offline" : `${sources.length} added`}
        </span>
      </header>

      <form
        className="source-entry"
        onSubmit={(event) => {
          event.preventDefault();
          void addSource();
        }}
      >
        <label htmlFor="source-value">GitHub URL or local path</label>
        <div>
          <input
            id="source-value"
            value={value}
            onChange={(event) => setValue(event.target.value)}
            placeholder="github.com/you/project or C:\\path\\to\\work"
            disabled={connected === false || busy !== null}
            spellCheck={false}
            autoComplete="off"
          />
          <button className="primary-button" type="submit" disabled={connected === false || busy !== null || !value.trim()}>
            {busy === "add" ? "Adding…" : "Add"}
          </button>
        </div>
      </form>

      {error && <div className="source-error" role="alert">{error}</div>}

      {sources.length > 0 ? (
        <ul className="source-cards" aria-label="Added work">
          {sources.map((source) => (
            <li key={source.id}>
              <span className="source-origin" aria-hidden="true">{source.origin === "github" ? "GH" : "LOCAL"}</span>
              <div>
                <strong>{source.label}</strong>
                <span>{purposeLabels[source.purpose]}</span>
                <code>{source.value}</code>
              </div>
              <span className="source-ready">Added</span>
              <button
                type="button"
                onClick={() => void removeSource(source)}
                disabled={busy !== null}
                aria-label={`Remove ${source.label}`}
              >
                {busy === source.id ? "Removing…" : "Remove"}
              </button>
            </li>
          ))}
        </ul>
      ) : connected !== false ? (
        <div className="source-empty">
          <strong>No work added yet</strong>
          <p>Start with the repository that best represents how you want the local model to work.</p>
        </div>
      ) : null}

      <p className="source-truth">Repositories show your patterns. Accepted examples and testable tasks teach the model what good work looks like.</p>
    </section>
  );
}
