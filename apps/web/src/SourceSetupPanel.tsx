import { useEffect, useState } from "react";
import {
  ApiClientError,
  addProjectSource,
  createAuthoredExample,
  createAuthoredTask,
  getAuthoredExamples,
  getAuthoredTasks,
  getProjectSources,
  removeAuthoredExample,
  removeAuthoredTask,
  removeProjectSource,
  searchGitHubRepositories,
  type AuthoredExample,
  type AuthoredTask,
  type AuthoredTaskWorkspace,
  type ProjectSource,
  type RepositorySearchResult,
  type SourceMode,
} from "./api";

const modeCopy: Record<SourceMode, { label: string; detail: string }> = {
  accepted_changes: { label: "Accepted changes → QLoRA SFT", detail: "Review useful commits and turn approved work into supervised teaching examples." },
  practice_tasks: { label: "Executable tasks → GRPO", detail: "Use resettable tasks and executable verifiers for reward-driven practice." },
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

function shouldSearchGitHub(value: string) {
  const text = value.trim();
  if (text.length < 2 || hasIntrinsicPurpose(text)) return false;
  const lower = text.toLowerCase();
  return !(
    text.includes("\\")
    || /^[a-z]:[\\/]/i.test(text)
    || /^[./~]/.test(text)
    || lower.startsWith("github.com/")
    || lower.startsWith("git@")
    || text.includes("://")
  );
}

function compactStars(value: number) {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function authoredTaskSplit(source: ProjectSource): "train" | "evaluation" | null {
  if (source.kind !== "repository") return null;
  const selected = displayedModes(source);
  if (selected.includes("practice_tasks")) return "train";
  if (selected.includes("evaluation_holdout")) return "evaluation";
  return null;
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
  const [repositoryResults, setRepositoryResults] = useState<RepositorySearchResult[]>([]);
  const [repositorySearching, setRepositorySearching] = useState(false);
  const [repositorySearchError, setRepositorySearchError] = useState<string | null>(null);
  const [repositorySearchEnabled, setRepositorySearchEnabled] = useState(true);
  const [authoredTasks, setAuthoredTasks] = useState<AuthoredTask[]>([]);
  const [taskSummary, setTaskSummary] = useState<AuthoredTaskWorkspace["summary"]>();
  const [authoredExamples, setAuthoredExamples] = useState<AuthoredExample[]>([]);
  const [exampleSourceId, setExampleSourceId] = useState("");
  const [exampleInstruction, setExampleInstruction] = useState("");
  const [exampleResponse, setExampleResponse] = useState("");
  const [exampleRightsConfirmed, setExampleRightsConfirmed] = useState(false);
  const [exampleBusy, setExampleBusy] = useState(false);
  const [exampleError, setExampleError] = useState<string | null>(null);
  const [taskSourceId, setTaskSourceId] = useState("");
  const [taskInstruction, setTaskInstruction] = useState("");
  const [taskWorkingDirectory, setTaskWorkingDirectory] = useState(".");
  const [taskInstall, setTaskInstall] = useState("");
  const [taskBuild, setTaskBuild] = useState("");
  const [taskTests, setTaskTests] = useState("");
  const [taskBrowserTests, setTaskBrowserTests] = useState("");
  const [taskVerifierBundle, setTaskVerifierBundle] = useState("");
  const [taskVerifierCommand, setTaskVerifierCommand] = useState("node /autotrainer-verifier/verify.mjs");
  const [taskVerifierReport, setTaskVerifierReport] = useState(".autotrainer-verifier-report.json");
  const [taskBusy, setTaskBusy] = useState(false);
  const [taskError, setTaskError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      getProjectSources(controller.signal),
      getAuthoredTasks(controller.signal),
      getAuthoredExamples(controller.signal),
    ])
      .then(([nextSources, taskWorkspace, exampleWorkspace]) => {
        setSources(nextSources);
        setAuthoredTasks(taskWorkspace.tasks);
        setTaskSummary(taskWorkspace.summary);
        setAuthoredExamples(exampleWorkspace.examples);
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

  const taskSources = sources.filter((source) => authoredTaskSplit(source) !== null);
  const selectedTaskSource = taskSources.find((source) => source.id === taskSourceId) ?? null;
  const exampleSources = sources.filter((source) => (
    source.kind === "repository" && source.partition !== "evaluation"
  ));
  const selectedExampleSource = exampleSources.find((source) => source.id === exampleSourceId) ?? null;

  useEffect(() => {
    if (exampleSources.some((source) => source.id === exampleSourceId)) return;
    setExampleSourceId(exampleSources[0]?.id ?? "");
  }, [sources, exampleSourceId]);

  useEffect(() => {
    if (taskSources.some((source) => source.id === taskSourceId)) return;
    setTaskSourceId(taskSources[0]?.id ?? "");
  }, [sources, taskSourceId]);

  useEffect(() => {
    if (connected !== true || !repositorySearchEnabled || !shouldSearchGitHub(value)) {
      setRepositoryResults([]);
      setRepositorySearching(false);
      setRepositorySearchError(null);
      return;
    }
    const controller = new AbortController();
    const query = value.trim();
    const timer = window.setTimeout(() => {
      setRepositorySearching(true);
      setRepositorySearchError(null);
      searchGitHubRepositories(query, 8, controller.signal)
        .then((results) => setRepositoryResults(results))
        .catch((reason: unknown) => {
          if (controller.signal.aborted) return;
          setRepositoryResults([]);
          setRepositorySearchError(reason instanceof Error ? reason.message : "GitHub search is unavailable.");
        })
        .finally(() => {
          if (!controller.signal.aborted) setRepositorySearching(false);
        });
    }, 250);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [connected, repositorySearchEnabled, value]);

  const toggleMode = (mode: SourceMode) => {
    setModes((current) => {
      if (current.includes(mode)) return current.filter((item) => item !== mode);
      if (mode === "reference_only" || mode === "evaluation_holdout") return [mode];
      return [...current.filter((item) => item !== "reference_only" && item !== "evaluation_holdout"), mode];
    });
  };

  const intrinsicPurpose = hasIntrinsicPurpose(value);

  const chooseRepository = (repository: RepositorySearchResult) => {
    // A selected identity is already clone-safe. Suppress the next debounced
    // lookup so the result menu does not reopen beneath the completed choice.
    setValue(repository.full_name);
    setRepositorySearchEnabled(false);
    setRepositoryResults([]);
    setRepositorySearchError(null);
    if (!licenseSpdx.trim() && repository.license_spdx !== "UNDECLARED") {
      setLicenseSpdx(repository.license_spdx);
    }
  };

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
      setRepositorySearchEnabled(true);
      setRepositoryResults([]);
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

  const submitTask = async () => {
    if (!taskSourceId || taskInstruction.trim().length < 20 || !taskBuild.trim() || !taskTests.trim() || !taskVerifierBundle.trim() || !taskVerifierCommand.trim()) return;
    setTaskBusy(true);
    setTaskError(null);
    try {
      const workspace = await createAuthoredTask({
        source_id: taskSourceId,
        instruction: taskInstruction.trim(),
        working_directory: taskWorkingDirectory.trim() || ".",
        ...(taskInstall.trim() ? { install: taskInstall.trim() } : {}),
        build: taskBuild.trim(),
        tests: taskTests.trim(),
        ...(taskBrowserTests.trim() ? { browser_tests: taskBrowserTests.trim() } : {}),
        verifier_bundle: taskVerifierBundle.trim(),
        verifier_command: taskVerifierCommand.trim(),
        verifier_report_path: taskVerifierReport.trim() || ".autotrainer-verifier-report.json",
      });
      setAuthoredTasks(workspace.tasks);
      setTaskSummary(workspace.summary);
      // Creating the first task also declares its managed task pack in YAML.
      // Refresh the source list so the GUI immediately reflects that shared state.
      setSources(await getProjectSources());
      setTaskInstruction("");
      setTaskError(null);
      onSourcesChanged?.();
    } catch (reason) {
      setTaskError(reason instanceof ApiClientError ? reason.message : "AutoTrainer could not author that task.");
    } finally {
      setTaskBusy(false);
    }
  };

  const deleteTask = async (task: AuthoredTask) => {
    setTaskBusy(true);
    setTaskError(null);
    try {
      const workspace = await removeAuthoredTask(task.split, task.id);
      setAuthoredTasks(workspace.tasks);
      setTaskSummary(workspace.summary);
      setSources(await getProjectSources());
      onSourcesChanged?.();
    } catch (reason) {
      setTaskError(reason instanceof ApiClientError ? reason.message : "AutoTrainer could not remove that task.");
    } finally {
      setTaskBusy(false);
    }
  };

  const taskFormReady = Boolean(
    taskSourceId
    && taskInstruction.trim().length >= 20
    && taskBuild.trim()
    && taskTests.trim()
    && taskVerifierBundle.trim()
    && taskVerifierCommand.trim(),
  );

  const submitExample = async () => {
    if (!exampleSourceId || exampleInstruction.trim().length < 20 || exampleResponse.trim().length < 20 || !exampleRightsConfirmed) return;
    setExampleBusy(true);
    setExampleError(null);
    try {
      const workspace = await createAuthoredExample({
        source_id: exampleSourceId,
        instruction: exampleInstruction.trim(),
        accepted_response: exampleResponse.trim(),
        rights_confirmed: exampleRightsConfirmed,
      });
      setAuthoredExamples(workspace.examples);
      setSources(await getProjectSources());
      setExampleInstruction("");
      setExampleResponse("");
      setExampleRightsConfirmed(false);
      onSourcesChanged?.();
    } catch (reason) {
      setExampleError(reason instanceof ApiClientError ? reason.message : "AutoTrainer could not author that example.");
    } finally {
      setExampleBusy(false);
    }
  };

  const deleteExample = async (example: AuthoredExample) => {
    setExampleBusy(true);
    setExampleError(null);
    try {
      const workspace = await removeAuthoredExample(example.id);
      setAuthoredExamples(workspace.examples);
      setSources(await getProjectSources());
      onSourcesChanged?.();
    } catch (reason) {
      setExampleError(reason instanceof ApiClientError ? reason.message : "AutoTrainer could not remove that example.");
    } finally {
      setExampleBusy(false);
    }
  };

  const exampleFormReady = Boolean(
    exampleSourceId
    && exampleInstruction.trim().length >= 20
    && exampleResponse.trim().length >= 20
    && exampleRightsConfirmed,
  );

  return (
    <section className="panel setup-step source-setup" aria-labelledby="source-setup-heading" data-tour="sources">
      <header className="step-heading source-setup-header">
        <span className="step-number" aria-hidden="true">2</span>
        <div><h2 id="source-setup-heading">Add a GitHub repo or local folder</h2><p>Accepted examples run QLoRA SFT. Executable tasks run GRPO. Adding both runs SFT first, then GRPO.</p></div>
        <span className={`status-chip ${connected === false ? "danger" : sources.length ? "good" : "muted"}`}>{connected === false ? "Backend offline" : `${sources.length} configured`}</span>
      </header>

      <form className="source-entry source-definition" onSubmit={(event) => { event.preventDefault(); void addSource(); }}>
        <label htmlFor="source-value">Search GitHub or enter a local path</label>
        <div className="source-repository-picker">
          <input
            id="source-value"
            value={value}
            onChange={(event) => { setValue(event.target.value); setRepositorySearchEnabled(true); }}
            placeholder="airflow, apache/airflow, or C:\\path\\to\\work"
            disabled={connected !== true || busy !== null || disabled}
            spellCheck={false}
            autoComplete="off"
            role="combobox"
            aria-autocomplete="list"
            aria-expanded={repositoryResults.length > 0}
            aria-controls="repository-search-results"
          />
          {repositorySearching && <small className="repository-search-status">Searching GitHub...</small>}
          {repositoryResults.length > 0 && (
            <ul id="repository-search-results" className="repository-search-results" role="listbox" aria-label="GitHub repositories">
              {repositoryResults.map((repository) => (
                <li key={repository.full_name} role="option" aria-selected={false}>
                  <button type="button" onClick={() => chooseRepository(repository)}>
                    <span><strong>{repository.full_name}</strong><small>{repository.description || "No description provided."}</small></span>
                    <span className="repository-search-meta">{repository.language || "Code"} · {compactStars(repository.stars)} stars{repository.archived ? " · archived" : ""}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          {repositorySearchError && <small className="repository-search-error">{repositorySearchError} You can still paste owner/repository.</small>}
        </div>

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

      <div className="task-authoring example-authoring" aria-labelledby="example-authoring-heading">
        <header>
          <div>
            <span className="eyebrow">Accepted work</span>
            <h3 id="example-authoring-heading">Create a supervised teaching example</h3>
            <p>Pair a concrete instruction with an accepted response from a locked training repository. AutoTrainer records provenance and compiles it into QLoRA SFT data.</p>
          </div>
          <span className="status-chip muted">{authoredExamples.length} authored</span>
        </header>

        {exampleError && <div className="source-error" role="alert">{exampleError}</div>}

        {exampleSources.length ? (
          <form className="task-authoring-form" onSubmit={(event) => { event.preventDefault(); void submitExample(); }}>
            <div className="task-form-section">
              <span className="task-form-number" aria-hidden="true">1</span>
              <div>
                <label htmlFor="example-source">Locked training source</label>
                <select id="example-source" value={exampleSourceId} onChange={(event) => setExampleSourceId(event.target.value)} disabled={exampleBusy || busy !== null || disabled}>
                  {exampleSources.map((source) => <option key={source.id} value={source.id}>{source.label}</option>)}
                </select>
                {selectedExampleSource && <small>Pinned at <code>{selectedExampleSource.revision}</code>. The compiled record will carry this repository identity and revision.</small>}
              </div>
            </div>
            <div className="task-form-section example-copy-fields">
              <span className="task-form-number" aria-hidden="true">2</span>
              <div>
                <label htmlFor="example-instruction">Instruction</label>
                <textarea id="example-instruction" value={exampleInstruction} onChange={(event) => setExampleInstruction(event.target.value)} placeholder="Describe the concrete change or question that produced the accepted work." rows={3} disabled={exampleBusy || disabled} />
                <label htmlFor="example-response">Accepted response</label>
                <textarea id="example-response" value={exampleResponse} onChange={(event) => setExampleResponse(event.target.value)} placeholder="Paste the reviewed response or solution that the model should learn from." rows={6} disabled={exampleBusy || disabled} />
              </div>
            </div>
            <div className="task-authoring-submit example-authoring-submit">
              <label className="rights-confirmation"><input type="checkbox" checked={exampleRightsConfirmed} onChange={(event) => setExampleRightsConfirmed(event.target.checked)} disabled={exampleBusy || disabled} /><span>I confirm I have the right to use this accepted response for training.</span></label>
              <button className="primary-button" type="submit" disabled={!exampleFormReady || exampleBusy || disabled}>{exampleBusy ? "Creating..." : "Create example"}</button>
            </div>
          </form>
        ) : (
          <div className="source-empty task-authoring-empty"><strong>Add a locked training repository first</strong><p>Evaluation holdouts stay isolated and cannot supply supervised examples.</p></div>
        )}

        {authoredExamples.length > 0 && (
          <ul className="authored-task-list" aria-label="Authored supervised examples">
            {authoredExamples.map((example) => (
              <li key={example.id}>
                <span>SFT</span>
                <div><strong>{example.id}</strong><p>{example.instruction}</p><code>{example.source_id} · rights confirmed</code>{example.next_action && <small><b>{example.next_action.title}.</b> {example.next_action.detail}</small>}</div>
                <span className="status-chip muted">{example.status}</span>
                <button type="button" onClick={() => void deleteExample(example)} disabled={exampleBusy || disabled} aria-label={`Remove example ${example.id}`}>Remove</button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="task-authoring" aria-labelledby="task-authoring-heading">
        <header>
          <div>
            <span className="eyebrow">Executable work</span>
            <h3 id="task-authoring-heading">Create a practice or evaluation task</h3>
            <p>A repository supplies the locked starting state. You supply the exact instruction, commands, and hidden verifier that define success.</p>
          </div>
          <span className={`status-chip ${taskSummary?.evaluation_groups_remaining === 0 ? "good" : "muted"}`}>
            {authoredTasks.length} tasks · {taskSummary?.evaluation_group_count ?? 0}/{taskSummary?.required_evaluation_groups ?? 5} held-out groups
          </span>
        </header>

        {taskError && <div className="source-error" role="alert">{taskError}</div>}

        {taskSources.length ? (
          <form className="task-authoring-form" onSubmit={(event) => { event.preventDefault(); void submitTask(); }}>
            <div className="task-form-section">
              <span className="task-form-number" aria-hidden="true">1</span>
              <div>
                <label htmlFor="task-source">Locked source</label>
                <select id="task-source" value={taskSourceId} onChange={(event) => setTaskSourceId(event.target.value)} disabled={taskBusy || busy !== null || disabled}>
                  {taskSources.map((source) => (
                    <option key={source.id} value={source.id}>{source.label} · {authoredTaskSplit(source) === "train" ? "GRPO practice" : "held-out evaluation"}</option>
                  ))}
                </select>
                {selectedTaskSource && <small>Pinned at <code>{selectedTaskSource.revision}</code>. The selected repository purpose fixes this task to the {authoredTaskSplit(selectedTaskSource)} split.</small>}
              </div>
            </div>

            <div className="task-form-section">
              <span className="task-form-number" aria-hidden="true">2</span>
              <div>
                <label htmlFor="task-instruction">What should the model change?</label>
                <textarea id="task-instruction" value={taskInstruction} onChange={(event) => setTaskInstruction(event.target.value)} placeholder="Describe one observable change, its constraints, and what must remain working." rows={3} disabled={taskBusy || disabled} />
                <small>Use one concrete task. This instruction is the prompt; repository files are not silently converted into training examples.</small>
              </div>
            </div>

            <div className="task-form-section">
              <span className="task-form-number" aria-hidden="true">3</span>
              <div className="task-runtime-fields">
                <label htmlFor="task-workdir"><span>Working directory</span><input id="task-workdir" value={taskWorkingDirectory} onChange={(event) => setTaskWorkingDirectory(event.target.value)} placeholder="." disabled={taskBusy || disabled} /></label>
                <label htmlFor="task-build"><span>Build command</span><input id="task-build" value={taskBuild} onChange={(event) => setTaskBuild(event.target.value)} placeholder="npm run build" disabled={taskBusy || disabled} /></label>
                <label htmlFor="task-tests"><span>Regression tests</span><input id="task-tests" value={taskTests} onChange={(event) => setTaskTests(event.target.value)} placeholder="npm test" disabled={taskBusy || disabled} /></label>
                <details className="advanced-options">
                  <summary>Install and browser commands</summary>
                  <div className="task-runtime-optional">
                    <label htmlFor="task-install"><span>Install command</span><input id="task-install" value={taskInstall} onChange={(event) => setTaskInstall(event.target.value)} placeholder="Leave blank if the runtime image supplies dependencies" disabled={taskBusy || disabled} /></label>
                    <label htmlFor="task-browser-tests"><span>Browser tests</span><input id="task-browser-tests" value={taskBrowserTests} onChange={(event) => setTaskBrowserTests(event.target.value)} placeholder="npm run test:browser" disabled={taskBusy || disabled} /></label>
                  </div>
                </details>
              </div>
            </div>

            <div className="task-form-section">
              <span className="task-form-number" aria-hidden="true">4</span>
              <div className="task-verifier-fields">
                <label htmlFor="task-verifier-bundle"><span>Hidden verifier folder</span><input id="task-verifier-bundle" value={taskVerifierBundle} onChange={(event) => setTaskVerifierBundle(event.target.value)} placeholder="C:\\path\\outside\\the\\repository\\verifier" disabled={taskBusy || disabled} /></label>
                <label htmlFor="task-verifier-command"><span>Verifier command</span><input id="task-verifier-command" value={taskVerifierCommand} onChange={(event) => setTaskVerifierCommand(event.target.value)} disabled={taskBusy || disabled} /></label>
                <label htmlFor="task-verifier-report"><span>Report path in workspace</span><input id="task-verifier-report" value={taskVerifierReport} onChange={(event) => setTaskVerifierReport(event.target.value)} disabled={taskBusy || disabled} /></label>
                <p>The verifier stays outside the editable repository. Its JSON report must contain <code>build_passed</code>, <code>regression_pass_rate</code>, <code>task_pass_rate</code>, <code>responsive_pass_rate</code>, <code>design_rule_pass_rate</code>, and <code>code_quality_pass_rate</code>. AutoTrainer does not generate hidden tests or guess correctness.</p>
              </div>
            </div>

            <div className="task-authoring-submit">
              <p>Creating this task declares its manifest. <strong>Prepare must still execute every gate</strong> before training is ready.</p>
              <button className="primary-button" type="submit" disabled={!taskFormReady || taskBusy || disabled}>{taskBusy ? "Creating..." : "Create task"}</button>
            </div>
          </form>
        ) : (
          <div className="source-empty task-authoring-empty">
            <strong>Choose a task-capable repository first</strong>
            <p>Add a repository as “Executable tasks → GRPO” for practice, or as an isolated evaluation holdout. Reference-only code cannot become a task.</p>
          </div>
        )}

        {authoredTasks.length > 0 && (
          <ul className="authored-task-list" aria-label="Authored executable tasks">
            {authoredTasks.map((task) => (
              <li key={`${task.split}-${task.id}`}>
                <span>{task.split === "train" ? "GRPO" : "EVAL"}</span>
                <div><strong>{task.id}</strong><p>{task.instruction || task.blockers[0]}</p><code>{task.source_id || "unknown source"} · {task.working_directory || "."}</code>{task.next_action && <small><b>{task.next_action.title}.</b> {task.next_action.detail}</small>}</div>
                <span className={`status-chip ${task.status === "blocked" ? "danger" : "muted"}`}>{task.status}</span>
                <button type="button" onClick={() => void deleteTask(task)} disabled={taskBusy || disabled} aria-label={`Remove task ${task.id}`}>Remove</button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
