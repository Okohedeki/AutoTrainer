import { useEffect, useMemo, useState } from "react";
import {
  ApiClientError,
  downloadProjectModel,
  getModelWorkspace,
  selectProjectModel,
  type ModelCacheState,
  type ModelWorkspace,
} from "./api";
import { projectSnapshot, type StatusTone } from "./data";


type ActionState = "idle" | "saving" | "downloading";

function cacheLabel(cache: ModelCacheState | null): { label: string; tone: StatusTone } {
  switch (cache?.status) {
    case "downloaded": return { label: "Downloaded", tone: "good" };
    case "cached_unverified": return { label: "Cached · verify", tone: "warning" };
    case "dependency_missing": return { label: "Hub dependency missing", tone: "danger" };
    case "revision_unresolved": return { label: "Revision unresolved", tone: "warning" };
    default: return { label: "Not downloaded", tone: "warning" };
  }
}

function readableBytes(value?: number): string {
  if (!value) return "Size recorded after download";
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
export default function ModelSetupPanel() {
  const [workspace, setWorkspace] = useState<ModelWorkspace | null>(null);
  const [connected, setConnected] = useState<boolean | null>(null);
  const [action, setAction] = useState<ActionState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState("qwen3.5-9b-text");
  const [revision, setRevision] = useState(projectSnapshot.model.revision);
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
        setError(reason instanceof Error ? reason.message : "Local backend is not connected.");
      });
    return () => controller.abort();
  }, []);

  const trainableModels = useMemo(
    () => Object.entries(workspace?.models || {}).filter(([, model]) => model.trainable_v1),
    [workspace],
  );
  const cache = workspace?.cache || null;
  const status = connected === false
    ? { label: "Backend offline", tone: "danger" as const }
    : connected === null
      ? { label: "Connecting", tone: "muted" as const }
      : cacheLabel(cache);
  const busy = action !== "idle";

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
      hydrate(await getModelWorkspace());
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "Could not save the model configuration.");
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
      await downloadProjectModel();
      hydrate(await getModelWorkspace());
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "The model download did not complete.");
    } finally {
      setAction("idle");
    }
  };

  return (
    <section className="panel model-contract" aria-labelledby="model-contract-heading" data-tour="model-contract">
      <div className="panel-header model-setup-header">
        <div><p className="panel-kicker">Model setup</p><h2 id="model-contract-heading">Choose the training base</h2></div>
        <span className={`status-chip ${status.tone}`}>{status.label}</span>
      </div>

      {connected === false && (
        <div className="info-callout danger model-api-callout">
          <strong>Start the local backend</strong>
          <p><code>autotrainer serve --config examples/frontend-expert/autotrainer.yaml</code></p>
        </div>
      )}
      {error && connected !== false && <div className="info-callout danger model-api-callout"><strong>Model action stopped</strong><p>{error}</p></div>}

      <div className="model-setup-layout">
        <form className="model-form" onSubmit={(event) => { event.preventDefault(); void saveSelection(); }}>
          <label>
            <span>Base model</span>
            <select value={selectedModel} onChange={(event) => changeModel(event.target.value)} disabled={!connected || busy}>
              {trainableModels.length > 0
                ? trainableModels.map(([alias, model]) => <option key={alias} value={alias}>{model.id} · {model.parameters}</option>)
                : <option value="qwen3.5-9b-text">Qwen/Qwen3.5-9B · 9B</option>}
            </select>
            <small>Only V1 profiles validated for one-GPU training appear here.</small>
          </label>
          <label>
            <span>Immutable revision</span>
            <input value={revision} onChange={(event) => setRevision(event.target.value)} disabled={!connected || busy} spellCheck={false} />
            <small>A branch is resolved and pinned before weights are accepted.</small>
          </label>
          <label>
            <span>Model cache</span>
            <input value={cacheDir} onChange={(event) => setCacheDir(event.target.value)} disabled={!connected || busy} spellCheck={false} />
            <small>Used by both the downloader and offline training.</small>
          </label>
          <div className="model-form-actions">
            <button className="secondary-button" type="submit" disabled={!connected || busy}>{action === "saving" ? "Saving…" : "Save model"}</button>
            <button className="primary-button" type="button" onClick={() => void downloadSelection()} disabled={!connected || busy || cache?.status === "downloaded"}>
              {action === "downloading" ? "Downloading…" : cache?.status === "downloaded" ? "Model downloaded" : "Download model"}
            </button>
          </div>
        </form>

        <div className="model-facts" aria-live="polite">
          <div><span>Configured model</span><strong>{workspace?.model.id || projectSnapshot.model.id}</strong><code>{workspace?.model.revision || projectSnapshot.model.revision}</code></div>
          <div><span>Cache state</span><strong>{status.label}</strong><small>{readableBytes(cache?.logical_bytes)}</small></div>
          <div><span>Hugging Face access</span><strong>{cache?.hf_token_configured ? "HF_TOKEN detected" : "No token detected"}</strong><small>Public models need no key; gated models do.</small></div>
          <div><span>Training mode</span><strong>{projectSnapshot.recipe.method}</strong><small>{projectSnapshot.recipe.quantization} · {projectSnapshot.recipe.context}</small></div>
        </div>
      </div>
      {action === "downloading" && <p className="download-note" role="status">The local backend is downloading and verifying the complete snapshot. Keep it running; AutoTrainer reports success only after the receipt is written.</p>}
    </section>
  );
}
