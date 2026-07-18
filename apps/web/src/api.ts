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

export type ModelDownloadJob = {
  id: string | null;
  status: "idle" | "queued" | "downloading" | "running" | "completed" | "failed" | string;
  message?: string;
  model_id?: string;
  revision?: string;
  error?: string | null;
};

export type ModelStatus = {
  cache: ModelCacheState;
  download_job: ModelDownloadJob | null;
};

export type ReferenceModelStatus = {
  id?: string;
  model_id?: string;
  revision?: string;
  status?: string;
  message?: string;
  cache?: ModelCacheState;
  download_job?: ModelDownloadJob | null;
};

export type ModelWorkspace = {
  models: Record<string, ModelCatalogRecord>;
  model: ProjectModel;
  cache: ModelCacheState;
};

export type ProjectRecord = {
  id: string;
  name: string;
  config_path?: string;
  active?: boolean;
  managed?: boolean;
  model?: { id: string; revision: string };
};

export type ProjectsWorkspace = {
  active_id: string | null;
  projects: ProjectRecord[];
};

export type ModelSearchResult = {
  id: string;
  revision?: string;
  default_revision?: string;
  pipeline_tag?: string | null;
  downloads?: number;
  likes?: number;
  gated?: boolean | string;
  compatibility: "supported" | "reference_only" | "unverified" | string;
  profile?: string | null;
  reason?: string;
};

export type RepositorySearchResult = {
  full_name: string;
  clone_url: string;
  description: string;
  language: string | null;
  stars: number;
  fork: boolean;
  archived: boolean;
  private: boolean;
  default_branch: string;
  license_spdx: string;
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
  purpose?: "work" | "examples" | "tasks";
  modes?: SourceMode[];
  roles?: string[];
  partition?: string;
  filters?: { include: string[]; exclude: string[] };
  license?: { spdx?: string; attribution?: string } | null;
  next_action?: { title: string; detail: string };
  revision?: string;
  status: "ready" | "configured" | string;
};

export type SourceMode = "accepted_changes" | "practice_tasks" | "reference_only" | "evaluation_holdout";

export type SourceInput = {
  value: string;
  modes?: SourceMode[];
  revision?: string;
  include?: string[];
  exclude?: string[];
  license_spdx?: string;
  license_attribution?: string;
};

export type PreparationResult = {
  status: "ready" | "blocked";
  recipe: "teach" | "practice" | "both" | "needs_training_data";
  summary: string;
  next_action: { title: string; detail: string } | null;
  steps: Array<{
    id: "validate" | "sources" | "compile" | "runtime";
    label: string;
    status: "complete" | "blocked" | "waiting";
  }>;
  details: Record<string, unknown>;
};

export type HistoryFile = {
  path: string;
  status: string;
  additions: number;
  deletions: number;
};

export type HistoryCandidate = {
  candidate_id: string;
  proposed_instruction: string;
  files: HistoryFile[];
  patch: string;
  flags: string[];
};

export type HistoryWorkspace = {
  summary: {
    reviewable_count: number;
    approved_count: number;
    stale_review_count: number;
    blocked_counts?: Record<string, number>;
  };
  candidates: HistoryCandidate[];
};

export type TrainingJob = {
  id: string | null;
  status: "idle" | "queued" | "running" | "completed" | "failed" | "interrupted";
  recipe: "teach" | "practice" | "both" | null;
  stage: "prepare" | "sft" | "grpo" | null;
  message: string;
  result: {
    status: "completed";
    recipe: "teach" | "practice" | "both";
    stages: Array<{
      status: "completed";
      stage: "sft" | "grpo";
      output_dir?: string;
      metrics?: Record<string, number | boolean>;
      trainable_adapter_parameters?: number;
    }>;
  } | null;
};

export type TrainingEvent = {
  sequence: number;
  job_id?: string | null;
  type: string;
  stage?: "prepare" | "sft" | "grpo" | string | null;
  step?: number;
  epoch?: number;
  task_id?: string;
  reward?: number;
  message?: string;
  metrics?: Record<string, number | boolean | string | null>;
  rubric?: Record<string, number>;
  hard_gate_passed?: boolean;
  gate_reason?: string | null;
};

export type TrainingEventPage = {
  job_id: string | null;
  cursor: number;
  events: TrainingEvent[];
  truncated: boolean;
  has_more: boolean;
};

export type BackendHealth = {
  status: "ok";
  config: string;
};

export type EvaluationTrial = {
  trial_id: string;
  task_id: string;
  arm_id: string;
  repetition: number;
  seed: number;
  status?: string;
};

export type EvaluationResult = EvaluationTrial & {
  status: string;
  hard_gate_passed: boolean;
  gate_reason: string | null;
  reward: number;
  components: Record<string, number>;
};

export type EvaluationJob = {
  id: string | null;
  status: "idle" | "queued" | "running" | "completed" | "failed" | "interrupted";
  plan_id: string | null;
  suite: string | null;
  phase: string;
  message: string;
  completed: number;
  total: number;
  current_trial: EvaluationTrial | null;
  planned_trials?: EvaluationTrial[];
  results: EvaluationResult[];
  results_truncated: boolean;
};

export type EvaluationSuite = {
  id: string;
  kind: string;
  runner_type: "builtin" | "command" | "external";
  phase: string;
  message: string;
  completed: number;
  total: number;
  results: EvaluationResult[];
  trials?: EvaluationTrial[];
  trials_truncated?: boolean;
  results_truncated: boolean;
  results_withheld_for_blind_review: boolean;
  review: {
    pairs_exported: boolean;
    review_count: number;
    required_reviews: number;
    complete: boolean;
  } | null;
};

export type EvaluationWorkspace = {
  readiness: {
    status: string;
    ready_task_count: number;
    blockers: string[];
    warnings: string[];
  };
  plan: {
    plan_id: string;
    task_count: number;
    repetitions: number;
    seeds: number[];
    trials?: EvaluationTrial[];
  } | null;
  job: EvaluationJob;
  suites: EvaluationSuite[];
};

export type EvaluationEvent = {
  sequence: number;
  type?: string;
  job_id?: string;
  plan_id?: string;
  suite?: string;
  phase?: string;
  message?: string;
  trial?: EvaluationTrial | null;
  result?: EvaluationResult | null;
  rubric?: {
    hard_gate_passed: boolean;
    reward: number;
    components: Record<string, number>;
  } | null;
  trial_id?: string;
  task_id?: string;
  arm_id?: string;
  reward?: number;
  hard_gate_passed?: boolean;
  gate_reason?: string | null;
  components?: Record<string, number>;
};

export type EvaluationEventPage = {
  events: EvaluationEvent[];
  oldest_sequence: number | null;
  latest_sequence: number | null;
  cursor_reset: boolean;
};

export type HostingStatus = {
  status: "not_ready" | "ready" | "loading" | "live" | "stopped" | "failed" | string;
  message: string;
  endpoint: string | null;
  model?: string | null;
  base_model?: string | null;
  revision?: string | null;
  adapter?: string | null;
  pid?: number | null;
};

export type HostingTestResult = {
  response?: string;
  content?: string;
  text?: string;
  model?: string;
  [key: string]: unknown;
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

export async function getBackendHealth(signal?: AbortSignal): Promise<BackendHealth> {
  return request("/api/v1/health", { signal });
}

export async function getProjects(signal?: AbortSignal): Promise<ProjectsWorkspace> {
  return request("/api/v1/projects", { signal });
}

export async function createProject(name: string): Promise<ProjectRecord> {
  return request("/api/v1/projects", { method: "POST", body: JSON.stringify({ name }) });
}

export async function selectProject(projectId: string): Promise<unknown> {
  return request("/api/v1/projects/select", {
    method: "POST",
    body: JSON.stringify({ project_id: projectId }),
  });
}

export async function getModelWorkspace(signal?: AbortSignal): Promise<ModelWorkspace> {
  const [catalog, current] = await Promise.all([
    request<{ models: Record<string, ModelCatalogRecord> }>("/api/v1/models", { signal }),
    request<{ model: ProjectModel; cache: ModelCacheState }>("/api/v1/model", { signal }),
  ]);
  return { models: catalog.models, model: current.model, cache: current.cache };
}

export async function searchHuggingFaceModels(query: string, limit = 12, signal?: AbortSignal): Promise<ModelSearchResult[]> {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  const result = await request<{ models: ModelSearchResult[] }>(`/api/v1/models/search?${params}`, { signal });
  return result.models;
}

export async function selectProjectModel(input: {
  model: string;
  revision: string;
}): Promise<{ model: ProjectModel; cache: ModelCacheState }> {
  return request("/api/v1/model/select", { method: "POST", body: JSON.stringify(input) });
}

export async function downloadProjectModel(): Promise<ModelDownloadJob> {
  return request("/api/v1/model/download", { method: "POST", body: "{}" });
}

export async function getModelStatus(signal?: AbortSignal): Promise<ModelStatus> {
  const result = await request<ModelStatus & ModelCacheState>("/api/v1/model/status", { signal });
  // Early backends returned cache fields at the top level. Keeping this small
  // fallback makes a reconnect safe during a backend/frontend rolling update.
  return {
    cache: result.cache ?? result,
    download_job: result.download_job ?? null,
  };
}

export async function getReferenceModel(signal?: AbortSignal): Promise<ReferenceModelStatus> {
  return request("/api/v1/reference-model", { signal });
}

export async function downloadReferenceModel(): Promise<ModelDownloadJob> {
  return request("/api/v1/reference-model/download", { method: "POST", body: "{}" });
}

export async function searchGitHubRepositories(query: string, limit = 8, signal?: AbortSignal): Promise<RepositorySearchResult[]> {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  const result = await request<{ repositories: RepositorySearchResult[] }>(`/api/v1/repositories/search?${params}`, { signal });
  return result.repositories;
}

export async function getProjectSources(signal?: AbortSignal): Promise<ProjectSource[]> {
  const result = await request<{ sources: ProjectSource[] }>("/api/v1/sources", { signal });
  return result.sources;
}

export async function addProjectSource(input: SourceInput): Promise<ProjectSource[]> {
  const result = await request<{ sources: ProjectSource[] }>("/api/v1/sources", {
    method: "POST",
    body: JSON.stringify(input),
  });
  return result.sources;
}

export async function removeProjectSource(id: string): Promise<ProjectSource[]> {
  const result = await request<{ sources: ProjectSource[] }>(`/api/v1/sources/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  return result.sources;
}

export async function prepareProject(): Promise<PreparationResult> {
  return request("/api/v1/prepare", { method: "POST", body: "{}" });
}

export async function getReviewHistory(signal?: AbortSignal): Promise<HistoryWorkspace> {
  return request("/api/v1/history", { signal });
}

export async function reviewHistoryCandidate(input: {
  candidate_id: string;
  decision: "approved" | "rejected";
  instruction?: string;
  rights_confirmed?: boolean;
}): Promise<HistoryWorkspace> {
  return request("/api/v1/history/review", { method: "POST", body: JSON.stringify(input) });
}

export async function retireStaleHistoryReviews(): Promise<HistoryWorkspace> {
  return request("/api/v1/history/retire-stale", { method: "POST", body: "{}" });
}

export async function getTrainingJob(signal?: AbortSignal): Promise<TrainingJob> {
  return request("/api/v1/training", { signal });
}

export async function startTraining(): Promise<TrainingJob> {
  return request("/api/v1/training/start", { method: "POST", body: "{}" });
}

export async function getTrainingEvents(after = 0, signal?: AbortSignal): Promise<TrainingEventPage> {
  return request(`/api/v1/training/events?after=${encodeURIComponent(after)}`, { signal });
}

export async function getEvaluationWorkspace(signal?: AbortSignal): Promise<EvaluationWorkspace> {
  return request("/api/v1/evaluation", { signal });
}

export async function planEvaluation(): Promise<EvaluationWorkspace> {
  return request("/api/v1/evaluation/plan", { method: "POST", body: "{}" });
}

export async function startEvaluation(): Promise<EvaluationJob> {
  return request("/api/v1/evaluation/start", {
    method: "POST",
    body: "{}",
  });
}


export async function getEvaluationEvents(after = 0, signal?: AbortSignal): Promise<EvaluationEventPage> {
  return request(`/api/v1/evaluation/events?after=${encodeURIComponent(after)}`, { signal });
}

export async function getHostingStatus(signal?: AbortSignal): Promise<HostingStatus> {
  return request("/api/v1/hosting", { signal });
}

export async function startHosting(adapter: string): Promise<HostingStatus> {
  return request("/api/v1/hosting/start", {
    method: "POST",
    body: JSON.stringify({ adapter }),
  });
}

export async function stopHosting(): Promise<HostingStatus> {
  return request("/api/v1/hosting/stop", { method: "POST", body: "{}" });
}

export async function testHosting(prompt: string): Promise<HostingTestResult> {
  return request("/api/v1/hosting/test", {
    method: "POST",
    body: JSON.stringify({ prompt }),
  });
}
