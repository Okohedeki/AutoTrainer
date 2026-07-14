export type ModelDefinition = {
  id: string;
  name: string;
  shortName: string;
  parameters: string;
  architecture: "dense" | "moe";
  trainingMode: "qlora";
  status: "ready" | "planned";
  description: string;
};

export type PipelineStage = {
  id: "baseline" | "sft" | "rl" | "benchmark" | "fable";
  label: string;
  detail: string;
};

export type RewardSignal = {
  id: string;
  label: string;
  weight: number;
  description: string;
};

export const modelCatalog: ModelDefinition[] = [
  {
    id: "google/gemma-2-9b-it",
    name: "Gemma 2 9B IT",
    shortName: "Gemma 9B",
    parameters: "9B",
    architecture: "dense",
    trainingMode: "qlora",
    status: "ready",
    description: "The first compatibility target for the single-GPU pipeline.",
  },
  {
    id: "custom/9b-checkpoint",
    name: "Custom 9B checkpoint",
    shortName: "Custom 9B",
    parameters: "9B",
    architecture: "dense",
    trainingMode: "qlora",
    status: "planned",
    description: "A future adapter for compatible Hugging Face checkpoints.",
  },
];

export const pipelineStages: PipelineStage[] = [
  {
    id: "baseline",
    label: "Baseline",
    detail: "Measure the untouched 9B model on held-out frontend tasks.",
  },
  {
    id: "sft",
    label: "QLoRA",
    detail: "Warm-start the adapter from successful frontend examples.",
  },
  {
    id: "rl",
    label: "Reinforcement",
    detail: "Optimize the same adapter against executable outcomes.",
  },
  {
    id: "benchmark",
    label: "Benchmark",
    detail: "Compare Base, QLoRA, and RL on unseen environments.",
  },
  {
    id: "fable",
    label: "Fable A/B",
    detail: "Run the base and winning model through identical orchestration.",
  },
];

export const rewardSignals: RewardSignal[] = [
  {
    id: "task-tests",
    label: "Task tests",
    weight: 0.35,
    description: "Hidden requirements and browser behavior.",
  },
  {
    id: "regressions",
    label: "Regression safety",
    weight: 0.2,
    description: "Existing behavior remains intact.",
  },
  {
    id: "responsive",
    label: "Responsive rules",
    weight: 0.2,
    description: "Layout constraints pass at defined viewports.",
  },
  {
    id: "design-rules",
    label: "Design rules",
    weight: 0.15,
    description: "Components and tokens follow the project policy.",
  },
  {
    id: "quality",
    label: "Patch quality",
    weight: 0.1,
    description: "The change is typed, focused, and maintainable.",
  },
];

export const defaultEnvironment = {
  id: "frontend-vite-v1",
  stack: ["React", "TypeScript", "Vite", "Tailwind", "Playwright"],
  task: "Repair a responsive pricing section without changing desktop behavior.",
  tools: ["read_file", "search_code", "apply_patch", "run_command", "inspect_browser"],
  limits: {
    toolCalls: 40,
    tokenBudget: 12_000,
    commandTimeoutSeconds: 120,
    networkAccess: false,
  },
};
