export type ViewId =
  | "overview"
  | "runs"
  | "data"
  | "environments"
  | "evaluations"
  | "artifacts"
  | "runtime";

export type StatusTone = "good" | "info" | "warning" | "danger" | "muted";

export type PipelineStage = {
  id: string;
  label: string;
  detail: string;
  status: "complete" | "ready" | "blocked" | "waiting";
  meta: string;
};

export type SourceRow = {
  id: string;
  kind: string;
  partition: "Train" | "Evaluation";
  role: string;
  location: string;
  identity: string;
  records: string;
  state: string;
  tone: StatusTone;
};

export type RewardSignal = {
  id: string;
  label: string;
  weight: number;
  description: string;
};

export type CommandDefinition = {
  id: string;
  label: string;
  description: string;
  command: string;
};

export const navigation: Array<{ id: ViewId; label: string; short: string; count?: string }> = [
  { id: "overview", label: "Overview", short: "OV" },
  { id: "runs", label: "Training runs", short: "TR", count: "0" },
  { id: "data", label: "Data sources", short: "DS", count: "5" },
  { id: "environments", label: "Environments", short: "EN", count: "1" },
  { id: "evaluations", label: "Evaluations", short: "EV", count: "2" },
  { id: "artifacts", label: "Artifacts", short: "AR" },
  { id: "runtime", label: "Runtime", short: "RT", count: "2" },
];

// This is an explicit snapshot of the bundled example's last validated local
// state. The UI never upgrades these labels to "running" or "complete" without
// a future backend response from the same artifacts the CLI writes.
export const projectSnapshot = {
  name: "polished-frontend-9b",
  slug: "polished-frontend-9b",
  configPath: "examples/frontend-expert/autotrainer.yaml",
  branch: "main",
  mode: "Local only",
  model: {
    id: "Qwen/Qwen3.5-9B",
    revision: "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    state: "Configured",
    cache: "Not downloaded",
    loader: "Text-only causal LM",
  },
  referenceModel: {
    id: "empero-ai/Qwythos-9B-Claude-Mythos-5-1M",
    revision: "14a29bae5143091aeaf87ad37120de4cd57d592c",
    state: "Deferred benchmark",
  },
  recipe: {
    method: "QLoRA → GRPO",
    quantization: "4-bit NF4",
    rank: 32,
    context: "2,048 tokens",
  },
};

// Recorded local readiness values are separate from project configuration so
// the future API can refresh host and cache health without rewriting YAML.
export const runtimeSnapshot = [
  { label: "GPU", value: "RTX 4090", detail: "24,564 MiB detected", tone: "good" as const },
  { label: "Project models", value: "Not cached", detail: "Qwen and Qwythos weights absent", tone: "warning" as const },
  { label: "Training stack", value: "Blocked", detail: "2 missing · 6 version mismatches", tone: "danger" as const },
  { label: "Sandbox", value: "Missing", detail: "Docker is required for RL", tone: "danger" as const },
];

// These stages match the CLI's executable dependency order. A waiting stage is
// not runnable merely because its static inputs compiled successfully.
export const pipelineStages: PipelineStage[] = [
  {
    id: "validate",
    label: "Validate",
    detail: "YAML, paths, recipes, and declared source shapes accepted.",
    status: "complete",
    meta: "Config valid",
  },
  {
    id: "sources",
    label: "Scan sources",
    detail: "Repositories, demonstrations, and task packs resolved into a source lock.",
    status: "complete",
    meta: "Source lock written",
  },
  {
    id: "compile",
    label: "Compile",
    detail: "Canonical SFT, RL, and evaluation JSONL written with hashes.",
    status: "complete",
    meta: "3 datasets",
  },
  {
    id: "runtime",
    label: "Runtime check",
    detail: "GPU is visible, but Docker and the pinned training stack are not ready.",
    status: "blocked",
    meta: "Resolve locally",
  },
  {
    id: "sft",
    label: "QLoRA / SFT",
    detail: "Inputs and recipe are ready; waits for model weights and runtime.",
    status: "waiting",
    meta: "No adapter",
  },
  {
    id: "grpo",
    label: "GRPO",
    detail: "Continues the same adapter inside executable frontend tasks.",
    status: "waiting",
    meta: "Needs SFT adapter",
  },
  {
    id: "evaluation",
    label: "Proof & package",
    detail: "Freeze evaluation, run both suites, review, report, then package.",
    status: "blocked",
    meta: "3 plan blockers · 1 proof gap",
  },
];

// Evaluation sources remain beside training sources so provenance overlap is
// visible before a comparison plan can be frozen.
export const sources: SourceRow[] = [
  {
    id: "preferred-pricing-ui",
    kind: "Git repository",
    partition: "Train",
    role: "Style + RL seed",
    location: "../..",
    identity: "be2ffc5 · repo 0d396b78",
    records: "10 files",
    state: "Locked from mutable HEAD",
    tone: "warning",
  },
  {
    id: "accepted-frontend-demonstrations",
    kind: "SFT JSONL",
    partition: "Train",
    role: "Demonstrations",
    location: "./data/sft-messages.jsonl",
    identity: "sha256:dcb67ad3",
    records: "1 record",
    state: "Compiled",
    tone: "good",
  },
  {
    id: "frontend-rl-tasks",
    kind: "Task pack",
    partition: "Train",
    role: "RL environment",
    location: "./tasks/train",
    identity: "snapshot be2ffc5",
    records: "1 task",
    state: "Ready",
    tone: "good",
  },
  {
    id: "held-out-newsletter-site",
    kind: "Git repository",
    partition: "Evaluation",
    role: "Held-out candidate",
    location: "../..",
    identity: "be2ffc5 · repo 0d396b78",
    records: "11 files",
    state: "Holdout conflict",
    tone: "danger",
  },
  {
    id: "held-out-frontend",
    kind: "Task pack",
    partition: "Evaluation",
    role: "Final benchmark",
    location: "./tasks/evaluation",
    identity: "snapshot be2ffc5",
    records: "1 task",
    state: "Authoring only",
    tone: "warning",
  },
];

export const preparationEvents = [
  { label: "Configuration validated", detail: "No schema errors", tone: "good" as const },
  { label: "Source inventory locked", detail: "21 eligible files", tone: "good" as const },
  { label: "Trainer datasets compiled", detail: "1 SFT · 1 RL · 1 evaluation", tone: "good" as const },
  { label: "SFT recipe dry-run", detail: "QLoRA settings accepted", tone: "info" as const },
  { label: "Evaluation readiness blocked", detail: "Adapter, holdout, and runner identities unresolved", tone: "danger" as const },
];

export const environment = {
  id: "FrontendEnvironment",
  factory: "autotrainer.environments.frontend:FrontendEnvironment",
  image: "autotrainer/frontend-runtime:0.1",
  backend: "Docker · network disabled",
  task: "Repair responsive frontend behavior while preserving desktop regressions.",
  tools: ["list_files", "read_file", "search_code", "apply_patch", "run_check"],
  limits: [
    ["Tool calls", "40"],
    ["Command timeout", "120 seconds"],
    ["Episode timeout", "900 seconds"],
    ["Network", "Disabled"],
  ],
};

export const rewardSignals: RewardSignal[] = [
  { id: "tests", label: "Task tests", weight: 0.35, description: "Hidden requirements and browser behavior." },
  { id: "regression", label: "Regression safety", weight: 0.2, description: "Existing project behavior remains intact." },
  { id: "responsive", label: "Responsive rules", weight: 0.2, description: "Layout constraints pass at defined viewports." },
  { id: "design", label: "Design rules", weight: 0.15, description: "Components and tokens follow project policy." },
  { id: "quality", label: "Patch quality", weight: 0.1, description: "The change is typed, focused, and maintainable." },
];

export const evaluationSuites = [
  {
    label: "Model benchmark",
    comparison: "Configured: base Qwen3.5 9B vs AutoTrainer adapter",
    plannedReference: "Deferred baseline: Qwythos 9B Claude Mythos",
    metric: "Verified task success",
    status: "Blocked",
    planBlocker: "Needs a frozen candidate adapter, independent held-out repositories, and a pinned local runner.",
    proofRequirement: "Execute every paired seed and verify the scored task evidence.",
  },
  {
    label: "Fable A/B",
    comparison: "Base 9B + Fable vs trained 9B + Fable",
    plannedReference: "Same configured base and frozen candidate",
    metric: "Blind preference rate",
    status: "Blocked",
    planBlocker: "Needs the same frozen candidate, independent holdout, and immutable Fable orchestration.",
    proofRequirement: "Generate both sites, ingest results, and collect three distinct blind reviews per pair.",
  },
];

// Every enabled web action currently resolves to one of these reproducible CLI
// commands. Credentials and other secrets must never be embedded here.
export const commands: CommandDefinition[] = [
  {
    id: "backend",
    label: "Start human backend",
    description: "Expose the same model and training operations to the local GUI on loopback only.",
    command: "autotrainer serve --config examples/frontend-expert/autotrainer.yaml",
  },
  {
    id: "model-status",
    label: "Inspect model cache",
    description: "Agent equivalent of the GUI model status, without making a network request.",
    command: "autotrainer model status --config examples/frontend-expert/autotrainer.yaml",
  },
  {
    id: "model-download",
    label: "Download base model",
    description: "Agent equivalent of Download model: pin, materialize, verify, and record the snapshot.",
    command: "autotrainer model download --config examples/frontend-expert/autotrainer.yaml",
  },
  {
    id: "validate",
    label: "Validate inputs",
    description: "Check YAML, source declarations, recipes, and paths without loading the model.",
    command: "autotrainer validate --config examples/frontend-expert/autotrainer.yaml",
  },
  {
    id: "doctor",
    label: "Check local runtime",
    description: "Inspect the GPU, Docker sandbox, Python, and pinned training packages.",
    command: "autotrainer doctor --config examples/frontend-expert/autotrainer.yaml",
  },
  {
    id: "sft-dry-run",
    label: "Dry-run QLoRA",
    description: "Resolve the exact SFT recipe and dataset before downloading weights.",
    command: "autotrainer train sft --dry-run --config examples/frontend-expert/autotrainer.yaml",
  },
  {
    id: "sft",
    label: "Start QLoRA",
    description: "Launch supervised adapter training after Doctor reports ready.",
    command: "autotrainer train sft --config examples/frontend-expert/autotrainer.yaml",
  },
  {
    id: "grpo",
    label: "Start GRPO",
    description: "Continue the completed SFT adapter with executable rewards.",
    command: "autotrainer train rl --config examples/frontend-expert/autotrainer.yaml",
  },
];
