import { useEffect, useMemo, useState } from "react";
import {
  ApiClientError,
  downloadProjectModel,
  downloadReferenceModel,
  getModelStatus,
  getModelWorkspace,
  getReferenceModel,
  searchHuggingFaceModels,
  selectProjectModel,
  type ModelCacheState,
  type ModelDownloadJob,
  type ModelSearchResult,
  type ModelWorkspace,
  type ReferenceModelStatus,
} from "./api";

type ActionState = "idle" | "saving" | "downloading";
type StatusTone = "good" | "warning" | "danger" | "muted" | "info";

const DEFAULT_MODEL = "Qwen/Qwen3.5-9B";
const DEFAULT_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a";

function cacheLabel(cache: ModelCacheState | null, selectionChanged: boolean): { label: string; tone: StatusTone } {
  if (selectionChanged) return { label: "Not downloaded", tone: "warning" };
  switch (cache?.status) {
    case "downloaded": return { label: "Downloaded", tone: "good" };
    case "cached_unverified": return { label: "Needs verification", tone: "warning" };
    case "dependency_missing": return { label: "Download support missing", tone: "danger" };
    case "revision_unresolved": return { label: "Version not pinned", tone: "warning" };
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

function compatibilityLabel(value: string) {
  if (value === "supported") return "Ready for V1 training";
  if (value === "reference_only") return "Evaluation reference only";
  return "Not verified for V1 training";
}

function compatibilityTone(value: string): StatusTone {
  if (value === "supported") return "good";
  if (value === "reference_only") return "warning";
  return "muted";
}

function BenchmarkReferenceRow({ disabled, onActiveChange }: { disabled: boolean; onActiveChange: (active: boolean) => void }) {
  const [reference, setReference] = useState<ReferenceModelStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const jobStatus = reference?.download_job?.status;
  const active = ["queued", "downloading", "running"].includes(jobStatus || "");
  const downloaded = reference?.cache?.status === "downloaded" || reference?.status === "downloaded" || jobStatus === "completed";

  useEffect(() => onActiveChange(active), [active, onActiveChange]);

  useEffect(() => {
    const controller = new AbortController();
    getReferenceModel(controller.signal).then(setReference).catch(() => undefined);
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!active) return;
    let stopped = false;
    const controller = new AbortController();
    let timer = 0;
    const poll = async () => {
      try {
        const next = await getReferenceModel(controller.signal);
        if (stopped) return;
        setReference(next);
        const nextStatus = next.download_job?.status;
        if (["queued", "downloading", "running"].includes(nextStatus || "")) timer = window.setTimeout(() => void poll(), 2_000);
      } catch (reason) {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : "Reference download status could not be refreshed.");
          timer = window.setTimeout(() => void poll(), 2_000);
        }
      }
    };
    timer = window.setTimeout(() => void poll(), 2_000);
    return () => {
      stopped = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [reference?.download_job?.id]);

  const download = async () => {
    setBusy(true);
    setError(null);
    try {
      const job = await downloadReferenceModel();
      setReference((current) => ({ ...(current || {}), download_job: job }));
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "The benchmark reference could not be queued.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="benchmark-reference-row" aria-labelledby="benchmark-reference-heading">
      <div className="reference-mark" aria-hidden="true">REF</div>
      <div><span>Benchmark reference</span><strong id="benchmark-reference-heading">{reference?.model_id || reference?.id || "empero-ai/Qwythos-9B-Claude-Mythos-5-1M"}</strong><code>{reference?.revision || "Pinned revision supplied by AutoTrainer"}</code><small>This model measures the trained specialist. It cannot become the training base.</small></div>
      <span className={`status-chip ${downloaded ? "good" : active ? "info" : jobStatus === "failed" ? "danger" : "muted"}`}>{downloaded ? "downloaded" : jobStatus || reference?.status || "not downloaded"}</span>
      <button className="secondary-button" type="button" onClick={() => void download()} disabled={disabled || busy || active || downloaded}>{busy ? "Queueing..." : active ? "Downloading..." : downloaded ? "Downloaded" : "Download reference"}</button>
      {error && <p role="alert">{error}</p>}
    </section>
  );
}

// Search is deliberately broad while compatibility claims stay narrow. Hub
// results remain visible without AutoTrainer pretending every model can train.
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
  const [downloadJob, setDownloadJob] = useState<ModelDownloadJob | null>(null);
  const [referenceActive, setReferenceActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState<ModelSearchResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<ModelSearchResult | null>(null);
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL);
  const [revision, setRevision] = useState(DEFAULT_REVISION);

  const hydrate = (next: ModelWorkspace) => {
    setWorkspace(next);
    setSelectedModel(next.model.id);
    setRevision(next.model.revision);
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

  useEffect(() => {
    const controller = new AbortController();
    getModelStatus(controller.signal)
      .then((status) => {
        setDownloadJob(status.download_job);
        setWorkspace((current) => current ? { ...current, cache: status.cache } : current);
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const active = ["queued", "downloading", "running"].includes(downloadJob?.status || "");
    if (!active) return;
    let stopped = false;
    const controller = new AbortController();
    let timer = 0;
    const poll = async () => {
      try {
        const status = await getModelStatus(controller.signal);
        if (stopped) return;
        setDownloadJob(status.download_job);
        setWorkspace((current) => current ? { ...current, cache: status.cache } : current);
        if (status.download_job?.status === "completed" || status.cache.status === "downloaded") hydrate(await getModelWorkspace(controller.signal));
        else if (status.download_job?.status === "failed") setError(status.download_job.error || status.download_job.message || "The model download failed.");
        else if (["queued", "downloading", "running"].includes(status.download_job?.status || "")) timer = window.setTimeout(() => void poll(), 2_000);
        else setError("The download job is no longer available. Queue the model again.");
      } catch (reason) {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : "Download status could not be refreshed.");
          timer = window.setTimeout(() => void poll(), 2_000);
        }
      }
    };
    timer = window.setTimeout(() => void poll(), 2_000);
    return () => {
      stopped = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [downloadJob?.id]);

  useEffect(() => {
    const nextQuery = query.trim();
    if (!searchOpen || nextQuery.length < 2) {
      setResults([]);
      setSearching(false);
      return;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setSearching(true);
      searchHuggingFaceModels(nextQuery, 12, controller.signal)
        .then((next) => {
          setResults(next);
          setError(null);
        })
        .catch((reason: unknown) => {
          if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Hugging Face search failed.");
        })
        .finally(() => {
          if (!controller.signal.aborted) setSearching(false);
        });
    }, 300);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [query, searchOpen]);

  const catalogCompatibility = useMemo(() => {
    const record = Object.values(workspace?.models || {}).find((item) => item.id === selectedModel);
    if (!record) return "unverified";
    return record.trainable_v1 ? "supported" : "reference_only";
  }, [selectedModel, workspace]);
  const compatibility = selectedResult?.id === selectedModel ? selectedResult.compatibility : catalogCompatibility;
  const selectionChanged = Boolean(
    workspace
    && (
      selectedModel !== workspace.model.id
      || revision !== workspace.model.revision
    ),
  );
  const cache = workspace?.cache || null;
  const downloadActive = ["queued", "downloading", "running"].includes(downloadJob?.status || "");
  const status = connected === false
    ? { label: "Not connected", tone: "danger" as const }
    : connected === null
      ? { label: "Connecting", tone: "muted" as const }
      : downloadActive
        ? { label: downloadJob?.status === "queued" ? "Download queued" : "Downloading", tone: "info" as const }
      : cacheLabel(cache, selectionChanged);
  const busy = action !== "idle";
  const locked = busy || downloadActive || referenceActive || disabled;

  const choose = (result: ModelSearchResult) => {
    // Updating the input with the selected ID must not trigger the debounced
    // search again and reopen the menu under the user's completed choice.
    setSearchOpen(false);
    setSelectedResult(result);
    setSelectedModel(result.id);
    setRevision(result.revision || result.default_revision || "");
    setQuery(result.id);
    setResults([]);
  };

  const saveSelection = async () => {
    setAction("saving");
    setError(null);
    try {
      await selectProjectModel({ model: selectedModel, revision });
      onModelChanged?.();
      hydrate(await getModelWorkspace());
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "Could not save these model settings.");
    } finally {
      setAction("idle");
    }
  };

  const downloadSelection = async () => {
    setAction("downloading");
    setError(null);
    try {
      await selectProjectModel({ model: selectedModel, revision });
      onModelChanged?.();
      setDownloadJob(await downloadProjectModel());
      onModelChanged?.();
      setWorkspace(await getModelWorkspace());
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
        <div><h2 id="model-heading">Choose the base model</h2><p>Search Hugging Face, pin the exact revision, then download it to this machine.</p></div>
        <span className={`status-chip ${status.tone}`}>{status.label}</span>
      </header>

      {connected === false && <div className="inline-message danger" role="alert"><strong>AutoTrainer is not connected.</strong><span>Start the local backend, then refresh.</span></div>}
      {error && connected !== false && <div className="inline-message danger" role="alert">{error}</div>}

      <form className="model-form" onSubmit={(event) => { event.preventDefault(); void downloadSelection(); }}>
        <label className="hub-search" htmlFor="hub-model-search">
          <span>Hugging Face model</span>
          <input
            id="hub-model-search"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setSearchOpen(true);
            }}
            placeholder="Type a model name or author, for example Qwen 9B"
            disabled={!connected || locked}
            spellCheck={false}
            autoComplete="off"
            aria-describedby="model-compatibility"
          />
          {searchOpen && query.trim() && (
            <div className="model-search-results" role="listbox" aria-label="Hugging Face model results">
              {query.trim().length < 2 ? <p>Type at least two characters.</p> : searching ? <p>Searching Hugging Face...</p> : results.length === 0 ? <p>No matching models returned.</p> : results.map((result) => (
                <button type="button" role="option" aria-selected={result.id === selectedModel} key={result.id} onClick={() => choose(result)} disabled={result.compatibility !== "supported" || !(result.revision || result.default_revision)} title={result.reason}>
                  <span><strong>{result.id}</strong><small>{result.reason || `${result.pipeline_tag || "Model"}${typeof result.downloads === "number" ? ` / ${result.downloads.toLocaleString()} downloads` : ""}`}</small></span>
                  <span className={`status-chip ${compatibilityTone(result.compatibility)}`}>{compatibilityLabel(result.compatibility)}</span>
                </button>
              ))}
            </div>
          )}
        </label>

        <div className="selected-model-row">
          <div><span>Selected model</span><strong>{selectedModel}</strong><code>{revision || "Revision required"}</code></div>
          <span id="model-compatibility" className={`status-chip ${compatibilityTone(compatibility)}`}>{compatibilityLabel(compatibility)}</span>
          <button className="primary-button model-download" type="submit" disabled={!connected || locked || compatibility !== "supported" || !selectedModel || !revision || (cache?.status === "downloaded" && !selectionChanged)}>
            {action === "downloading" ? "Queueing..." : downloadActive ? "Downloading..." : cache?.status === "downloaded" && !selectionChanged ? "Downloaded" : "Use & download"}
          </button>
        </div>

        {selectedResult?.reason && <p className="field-note">{selectedResult.reason}</p>}

        <details className="advanced-options">
          <summary>Revision and storage</summary>
          <div className="advanced-fields model-revision-fields">
            <label htmlFor="model-revision"><span>Exact revision</span><input id="model-revision" value={revision} onChange={(event) => setRevision(event.target.value)} disabled={!connected || locked} spellCheck={false} /></label>
            <div className="advanced-meta"><span>{readableBytes(cache?.logical_bytes)}</span><span>{cache?.cache_dir || "Shared project model cache"}</span><span>{cache?.hf_token_configured ? "Hugging Face access configured" : "Public models need no key"}</span></div>
            <button className="secondary-button" type="button" onClick={() => void saveSelection()} disabled={!connected || locked || !selectionChanged}>{action === "saving" ? "Saving..." : "Save selection"}</button>
          </div>
        </details>
      </form>

      {(action === "downloading" || downloadActive) && <div className="download-state" role="status"><span aria-hidden="true" /><div><strong>{downloadJob?.status === "queued" ? "Download queued" : action === "downloading" ? "Queueing the pinned download" : "Downloading and verifying the complete snapshot"}</strong><p>{downloadJob?.message || "AutoTrainer will report Downloaded only after the pinned files and receipt are present."}</p></div></div>}
      <BenchmarkReferenceRow disabled={busy || downloadActive || disabled} onActiveChange={setReferenceActive} />
    </section>
  );
}
