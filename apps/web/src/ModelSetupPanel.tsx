import { useEffect, useMemo, useState } from "react";
import {
  ApiClientError,
  downloadProjectModel,
  getModelWorkspace,
  selectProjectModel,
  type ModelCacheState,
  type ModelWorkspace,
} from "./api";

type ActionState = "idle" | "saving" | "downloading";
type StatusTone = "good" | "warning" | "danger" | "muted";

const DEFAULT_MODEL = "qwen3.5-9b-text";
const DEFAULT_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a";

function cacheLabel(cache: ModelCacheState | null, selectionChanged: boolean): { label: string; tone: StatusTone } {
  if (selectionChanged) return { label: "Not downloaded", tone: "warning" };
  switch (cache?.status) {
    case "downloaded": return { label: "Downloaded", tone: "good" };
    case "cached_unverified": return { label: "Download needs checking", tone: "warning" };
    case "dependency_missing": return { label: "Download support missing", tone: "danger" };
    case "revision_unresolved": return { label: "Version not pinned", tone: "warning" };
    default: return { label: "Not downloaded", tone: "warning" };
  }
}

function readableBytes(value?: number): string {
  if (!value) return "Size appears after download";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toFixed(unit > 2 ? 1 : 0)} ${units[unit]}`;
}

// This is the human-facing model lifecycle. Every mutation goes through the
// local API and lands in autotrainer.yaml before a model download can begin.
export default function ModelSetupPanel({
  onModelChanged,
  disabled = false,
}: {
  onModelChanged?: () => void;
  disabled?: boolean;
}) {
  const [workspace, setWorkspace] = useState<ModelWorkspace | null>(null);
  const [connected, setConnected] = useState<boolean | null>(null);
  const [action, setAction] = useState<ActionState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL);
  const [revision, setRevision] = useState(DEFAULT_REVISION);
  const [cacheDir, setCacheDir] = useState("./.autotrainer/model-cache");

  const hydrate = (next: ModelWorkspace) => {
    setWorkspace(next);
    const catalogMatch = Object.entries(next.models).find(([, model]) => model.id === next.model.id);
    if (catalogMatch) setSelectedModel(catalogMatch[0]);
    setRevision(next.model.revision);
    setCacheDir(next.model.cache_dir || ".autotrainer/model-cache");
    setConnected(true);
    setError(null);
  };

  useEffect(() => {
    const controller = new AbortController();
    getModelWorkspace(controller.signal)
      .then(hydrate)
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        setConnected(false);
        setError(reason instanceof Error ? reason.message : "AutoTrainer is not connected.");
      });
    return () => controller.abort();
  }, []);

  const trainableModels = useMemo(
    () => Object.entries(workspace?.models || {}).filter(([, model]) => model.trainable_v1),
    [workspace],
  );
  const selectedRecord = workspace?.models[selectedModel];
  const selectionChanged = Boolean(
    workspace
    && (
      selectedRecord?.id !== workspace.model.id
      || revision !== workspace.model.revision
      || cacheDir !== (workspace.model.cache_dir || ".autotrainer/model-cache")
    ),
  );
  const cache = workspace?.cache || null;
  const status = connected === false
    ? { label: "Not connected", tone: "danger" as const }
    : connected === null
      ? { label: "Connecting", tone: "muted" as const }
      : cacheLabel(cache, selectionChanged);
  const busy = action !== "idle";
  const locked = busy || disabled;

  const changeModel = (alias: string) => {
    setSelectedModel(alias);
    const selected = workspace?.models[alias];
    if (selected?.default_revision) setRevision(selected.default_revision);
  };

  const saveSelection = async () => {
    setAction("saving");
    setError(null);
    try {
      await selectProjectModel({ model: selectedModel, revision, cache_dir: cacheDir });
      onModelChanged?.();
      hydrate(await getModelWorkspace());
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "Could not save these settings.");
    } finally {
      setAction("idle");
    }
  };

  const downloadSelection = async () => {
    setAction("downloading");
    setError(null);
    try {
      // Saving first guarantees the GUI downloads precisely what it displays.
      await selectProjectModel({ model: selectedModel, revision, cache_dir: cacheDir });
      onModelChanged?.();
      await downloadProjectModel();
      onModelChanged?.();
      hydrate(await getModelWorkspace());
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "The model download did not complete.");
    } finally {
      setAction("idle");
    }
  };

  return (
    <section className="panel setup-step model-setup" aria-labelledby="model-heading" data-tour="model">
      <header className="step-heading">
        <span className="step-number" aria-hidden="true">1</span>
        <div>
          <h2 id="model-heading">Choose model</h2>
          <p>Pick the small model you want to make better at your work.</p>
        </div>
        <span className={`status-chip ${status.tone}`}>{status.label}</span>
      </header>

      {connected === false && (
        <div className="inline-message danger" role="alert">
          <strong>AutoTrainer is not connected.</strong>
          <span>Start <code>autotrainer serve --config autotrainer.yaml</code>, then refresh.</span>
        </div>
      )}
      {error && connected !== false && <div className="inline-message danger" role="alert">{error}</div>}

      <form
        className="model-form"
        onSubmit={(event) => {
          event.preventDefault();
          void downloadSelection();
        }}
      >
        <div className="model-primary-action">
          <label htmlFor="base-model">
            <span>Model</span>
            <select
              id="base-model"
              value={selectedModel}
              onChange={(event) => changeModel(event.target.value)}
              disabled={!connected || locked}
            >
              {trainableModels.length > 0
                ? trainableModels.map(([alias, model]) => <option key={alias} value={alias}>{model.id} · {model.parameters}</option>)
                : <option value={DEFAULT_MODEL}>Qwen/Qwen3.5-9B · 9B</option>}
            </select>
            <small>Only models prepared for one local GPU appear here.</small>
          </label>
          <button
            className="primary-button model-download"
            type="submit"
            disabled={!connected || locked || (cache?.status === "downloaded" && !selectionChanged)}
          >
            {action === "downloading" ? "Downloading…" : cache?.status === "downloaded" && !selectionChanged ? "Downloaded" : "Select & download"}
          </button>
        </div>

        <details className="advanced-options">
          <summary>Advanced</summary>
          <div className="advanced-fields">
            <label htmlFor="model-revision">
              <span>Exact version</span>
              <input id="model-revision" value={revision} onChange={(event) => setRevision(event.target.value)} disabled={!connected || locked} spellCheck={false} />
            </label>
            <label htmlFor="model-cache">
              <span>Download folder</span>
              <input id="model-cache" value={cacheDir} onChange={(event) => setCacheDir(event.target.value)} disabled={!connected || locked} spellCheck={false} />
            </label>
            <div className="advanced-meta">
              <span>{readableBytes(cache?.logical_bytes)}</span>
              <span>{cache?.hf_token_configured ? "Access key found" : "Public models need no key"}</span>
            </div>
            <button className="secondary-button" type="button" onClick={() => void saveSelection()} disabled={!connected || locked || !selectionChanged}>
              {action === "saving" ? "Saving…" : "Save settings"}
            </button>
          </div>
        </details>
      </form>

      {action === "downloading" && <p className="download-note" role="status">Downloading and checking the complete model. Keep AutoTrainer open.</p>}
    </section>
  );
}
