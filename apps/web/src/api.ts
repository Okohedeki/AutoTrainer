// Typed client for the loopback backend. The GUI never shells out to the CLI;
// both interfaces call the same Python model_service operations instead.

export type ModelCatalogRecord = {
  id: string;
  default_revision?: string;
  loader: string;
  parameters: string;
  license: string;
  purpose: string;
  trainable_v1: boolean;
  minimum_vram_gib: number | null;
  notes: string;
};

export type ProjectModel = {
  provider: string;
  id: string;
  revision: string;
  cache_dir?: string;
  loader: string;
  dtype?: string;
};

export type ModelCacheState = {
  model_id: string;
  revision: string;
  immutable: boolean;
  status: "not_downloaded" | "revision_unresolved" | "dependency_missing" | "cached_unverified" | "downloaded";
  snapshot_path: string | null;
  receipt: string | null;
  hf_token_configured: boolean;
  cache_dir: string;
  file_count?: number;
  logical_bytes?: number;
};

export type ModelWorkspace = {
  models: Record<string, ModelCatalogRecord>;
  model: ProjectModel;
  cache: ModelCacheState;
};

// The normal source picker deals in product concepts, not YAML source kinds.
// The backend keeps revisions and provenance while the GUI only needs enough
// information to show what will teach the model.
export type ProjectSource = {
  id: string;
  kind: "repository" | "sft_jsonl" | "task_pack";
  label: string;
  value: string;
  origin: "github" | "local";
  purpose: "work" | "examples" | "tasks";
  revision?: string;
  status: "ready";
};

type ApiErrorBody = { error?: { code?: string; message?: string } };

export class ApiClientError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = "ApiClientError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiClientError("Local backend is not connected.");
  }
  const payload = await response.json() as T & ApiErrorBody;
  if (!response.ok) {
    throw new ApiClientError(payload.error?.message || `Local backend returned ${response.status}.`, response.status);
  }
  return payload;
}

export async function getModelWorkspace(signal?: AbortSignal): Promise<ModelWorkspace> {
  const [catalog, current] = await Promise.all([
    request<{ models: Record<string, ModelCatalogRecord> }>("/api/v1/models", { signal }),
    request<{ model: ProjectModel; cache: ModelCacheState }>("/api/v1/model", { signal }),
  ]);
  return { models: catalog.models, model: current.model, cache: current.cache };
}

export async function selectProjectModel(input: {
  model: string;
  revision: string;
  cache_dir: string;
}): Promise<{ model: ProjectModel; cache: ModelCacheState }> {
  return request("/api/v1/model/select", { method: "POST", body: JSON.stringify(input) });
}

export async function downloadProjectModel(): Promise<ModelCacheState> {
  return request("/api/v1/model/download", { method: "POST", body: "{}" });
}

export async function getProjectSources(signal?: AbortSignal): Promise<ProjectSource[]> {
  const result = await request<{ sources: ProjectSource[] }>("/api/v1/sources", { signal });
  return result.sources;
}

export async function addProjectSource(value: string): Promise<ProjectSource[]> {
  const result = await request<{ sources: ProjectSource[] }>("/api/v1/sources", {
    method: "POST",
    body: JSON.stringify({ value }),
  });
  return result.sources;
}

export async function removeProjectSource(id: string): Promise<ProjectSource[]> {
  const result = await request<{ sources: ProjectSource[] }>(`/api/v1/sources/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  return result.sources;
}
