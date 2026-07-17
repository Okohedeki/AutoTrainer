import { useEffect, useState } from "react";
import {
  ApiClientError,
  addProjectSource,
  getProjectSources,
  removeProjectSource,
  type ProjectSource,
  type SourceMode,
} from "./api";

const modeCopy: Record<SourceMode, { label: string; detail: string }> = {
  accepted_changes: { label: "Accepted changes", detail: "Review useful commits and turn approved work into teaching examples." },
  practice_tasks: { label: "Practice tasks", detail: "Use executable tasks from this repository for reward-driven practice." },
  reference_only: { label: "Reference only", detail: "Learn project structure and conventions without training on its history." },
  evaluation_holdout: { label: "Isolated evaluation holdout", detail: "Keep this source out of training and use it only to measure the frozen model." },
};

function displayedModes(source: ProjectSource): SourceMode[] {
  if (source.modes?.length) return source.modes;
  if (source.partition === "evaluation") return ["evaluation_holdout"];
  if (source.purpose === "examples") return ["accepted_changes"];
  if (source.purpose === "tasks") return ["practice_tasks"];
  return ["reference_only"];
}

function splitPatterns(value: string) {
  return value.split(/[\n,]/).map((item) => item.trim()).filter(Boolean);
}

function hasIntrinsicPurpose(value: string) {
  const normalized = value.trim().toLowerCase().replaceAll("\\", "/");
  return normalized.endsWith(".jsonl") || normalized.endsWith("/tasks.yaml") || normalized.endsWith("/tasks.yml") || normalized.endsWith(".taskpack.json");
}

// Purpose is required because a repository is not training data by itself.
// Accepted changes and practice tasks may be combined; reference and holdout
// stay exclusive so one source cannot silently leak into evaluation.
export default function SourceSetupPanel({
  onSourcesChanged,
  disabled = false,
}: {
  onSourcesChanged?: () => void;
  disabled?: boolean;
}) {
  const [sources, setSources] = useState<ProjectSource[]>([]);
  const [value, setValue] = useState("");
  const [modes, setModes] = useState<SourceMode[]>([]);
  const [revision, setRevision] = useState("");
  const [include, setInclude] = useState("");
  const [exclude, setExclude] = useState("");
  const [licenseSpdx, setLicenseSpdx] = useState("");
  const [licenseAttribution, setLicenseAttribution] = useState("");
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

  const toggleMode = (mode: SourceMode) => {
    setModes((current) => {
      if (current.includes(mode)) return current.filter((item) => item !== mode);
      if (mode === "reference_only" || mode === "evaluation_holdout") return [mode];
      return [...current.filter((item) => item !== "reference_only" && item !== "evaluation_holdout"), mode];
    });
  };

  const intrinsicPurpose = hasIntrinsicPurpose(value);

  const addSource = async () => {
    const nextValue = value.trim();
    if (!nextValue || (!intrinsicPurpose && modes.length === 0)) return;
    setBusy("add");
    setError(null);
    try {
      setSources(await addProjectSource({
        value: nextValue,
        ...(!intrinsicPurpose ? { modes } : {}),
        ...(revision.trim() ? { revision: revision.trim() } : {}),
        ...(splitPatterns(include).length ? { include: splitPatterns(include) } : {}),
        ...(splitPatterns(exclude).length ? { exclude: splitPatterns(exclude) } : {}),
        ...(licenseSpdx.trim() ? { license_spdx: licenseSpdx.trim() } : {}),
        ...(licenseAttribution.trim() ? { license_attribution: licenseAttribution.trim() } : {}),
      }));
      setValue("");
      setModes([]);
      setRevision("");
      setInclude("");
      setExclude("");
      setLicenseSpdx("");
      setLicenseAttribution("");
      setConnected(true);
      onSourcesChanged?.();
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
      onSourcesChanged?.();
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
        <div><h2 id="source-setup-heading">Add a GitHub repo or local folder</h2><p>Tell AutoTrainer exactly how this source may be used. A repository alone never becomes training data.</p></div>
        <span className={`status-chip ${connected === false ? "danger" : sources.length ? "good" : "muted"}`}>{connected === false ? "Backend offline" : `${sources.length} configured`}</span>
      </header>

      <form className="source-entry source-definition" onSubmit={(event) => { event.preventDefault(); void addSource(); }}>
        <label htmlFor="source-value">GitHub URL or local path</label>
        <input id="source-value" value={value} onChange={(event) => setValue(event.target.value)} placeholder="github.com/you/project or C:\\path\\to\\work" disabled={connected !== true || busy !== null || disabled} spellCheck={false} autoComplete="off" />

        {!intrinsicPurpose ? <fieldset className="source-purpose-options">
          <legend>What should AutoTrainer use from it?</legend>
          {Object.entries(modeCopy).map(([id, copy]) => {
            const mode = id as SourceMode;
            return (
              <label key={mode} className={modes.includes(mode) ? "selected" : ""}>
                <input type="checkbox" checked={modes.includes(mode)} onChange={() => toggleMode(mode)} disabled={connected !== true || busy !== null || disabled} />
                <span><strong>{copy.label}</strong><small>{copy.detail}</small></span>
              </label>
            );
          })}
        </fieldset> : <div className="intrinsic-purpose-note"><strong>Purpose comes from this file type</strong><p>AutoTrainer will keep a demonstration JSONL or executable task pack in its intrinsic role.</p></div>}

        <details className="advanced-options source-advanced">
          <summary>Revision, paths, and license</summary>
          <div className="source-advanced-grid">
            <label htmlFor="source-revision"><span>Revision</span><input id="source-revision" value={revision} onChange={(event) => setRevision(event.target.value)} placeholder="main or commit SHA" disabled={connected !== true || busy !== null || disabled} /></label>
            <label htmlFor="source-license"><span>SPDX license</span><input id="source-license" value={licenseSpdx} onChange={(event) => setLicenseSpdx(event.target.value)} placeholder="MIT" disabled={connected !== true || busy !== null || disabled} /></label>
            <label htmlFor="source-include"><span>Include paths</span><input id="source-include" value={include} onChange={(event) => setInclude(event.target.value)} placeholder="src/**, tests/**" disabled={connected !== true || busy !== null || disabled} /></label>
            <label htmlFor="source-exclude"><span>Exclude paths</span><input id="source-exclude" value={exclude} onChange={(event) => setExclude(event.target.value)} placeholder="vendor/**, dist/**" disabled={connected !== true || busy !== null || disabled} /></label>
            <label className="wide" htmlFor="source-attribution"><span>License attribution</span><input id="source-attribution" value={licenseAttribution} onChange={(event) => setLicenseAttribution(event.target.value)} placeholder="Required attribution, if any" disabled={connected !== true || busy !== null || disabled} /></label>
          </div>
        </details>

        <div className="source-submit-row"><p>{connected === null ? "Loading existing sources..." : intrinsicPurpose ? "Intrinsic demonstration or task-pack role" : modes.length === 0 ? "Choose at least one purpose." : modes.map((mode) => modeCopy[mode].label).join(" + ")}</p><button className="primary-button" type="submit" disabled={connected !== true || busy !== null || disabled || !value.trim() || (!intrinsicPurpose && modes.length === 0)}>{busy === "add" ? "Adding..." : "Add source"}</button></div>
      </form>

      {error && <div className="source-error" role="alert">{error}</div>}

      {sources.length > 0 ? (
        <ul className="source-cards" aria-label="Configured sources">
          {sources.map((source) => (
            <li key={source.id}>
              <span className="source-origin" aria-hidden="true">{source.origin === "github" ? "GH" : "LOCAL"}</span>
              <div><strong>{source.label}</strong><span>{displayedModes(source).map((mode) => modeCopy[mode].label).join(" + ")}</span><code>{source.value}{source.revision ? ` @ ${source.revision}` : ""}</code>{source.next_action && <small><b>{source.next_action.title}.</b> {source.next_action.detail}</small>}</div>
              <span className="source-ready">Configured</span>
              <button type="button" onClick={() => void removeSource(source)} disabled={busy !== null || disabled} aria-label={`Remove ${source.label}`}>{busy === source.id ? "Removing..." : "Remove"}</button>
            </li>
          ))}
        </ul>
      ) : connected !== false ? (
        <div className="source-empty"><strong>No sources configured</strong><p>Add the repository or local folder that represents the work this specialist should master.</p></div>
      ) : null}
    </section>
  );
}
