import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

async function source(path) {
  return readFile(new URL(path, root), "utf8");
}

test("build emits the AutoTrainer console", async () => {
  const indexUrl = new URL("dist/index.html", root);
  await access(indexUrl);
  const html = await readFile(indexUrl, "utf8");
  assert.match(html, /<title>AutoTrainer - Train a local specialist<\/title>/);
  assert.match(html, /<div id="root"><\/div>/);
  const assets = await readdir(new URL("dist/assets/", root));
  assert.ok(assets.some((name) => name.endsWith(".js")));
  assert.ok(assets.some((name) => name.endsWith(".css")));
});

test("the operating console follows Projects, Data, Train, Evaluate, Serve", async () => {
  const app = await source("src/App.tsx");
  assert.match(app, /type ViewId = "projects" \| "data" \| "train" \| "evaluate" \| "serve"/);
  for (const label of ["Projects", "Data", "Train", "Evaluate", "Serve"]) assert.match(app, new RegExp(`label: "${label}"`));
  assert.match(app, /<ProjectsPanel/);
  assert.match(app, /<ModelSetupPanel/);
  assert.match(app, /<SourceSetupPanel/);
  assert.match(app, /<GrpoEvidencePanel context="data"/);
  assert.match(app, /<DatasetWorkspacePanel/);
  assert.match(app, /<TrainingMonitorPanel/);
  assert.match(app, /<EvaluationMonitorPanel/);
  assert.match(app, /<ServePanel/);
  assert.doesNotMatch(app, /<PreparePanel/);
  assert.doesNotMatch(app, /className="hero"|site-footer|CommandDrawer/);
  assert.ok(app.indexOf("<SourceSetupPanel") < app.indexOf("<DatasetWorkspacePanel"));
  assert.ok(app.indexOf("<DatasetWorkspacePanel") < app.indexOf("<GrpoEvidencePanel context=\"data\""));
});

test("Data treats merged-PR datasets as an inspectable local training asset", async () => {
  const panel = await source("src/DatasetWorkspacePanel.tsx");
  const api = await source("src/api.ts");
  for (const rule of ["Merged PR", "main / master", "Required", "Local only"]) assert.match(panel, new RegExp(rule));
  for (const operation of ["getDatasetWorkspace", "syncDatasetSources", "designDatasetCandidate", "freezeDataset"]) assert.match(panel, new RegExp(operation));
  assert.match(panel, /Local model/);
  assert.match(panel, /Anthropic Claude/);
  for (const step of ["Import", "Analyze", "Review", "Lock"]) assert.match(panel, new RegExp(step));
  assert.match(panel, /Accepted patch/);
  assert.match(panel, /Add to QLoRA dataset/);
  assert.match(panel, /GRPO task/);
  assert.match(panel, /Lock dataset for training/);
  assert.match(api, /\/api\/v1\/dataset\/sync/);
  assert.match(api, /\/api\/v1\/dataset\/design/);
  assert.match(api, /\/api\/v1\/dataset\/freeze/);
  assert.doesNotMatch(panel, /Math\.random|estimated time|token counter/i);
});

test("projects can be created and switched through durable backend state", async () => {
  const panel = await source("src/ProjectsPanel.tsx");
  const api = await source("src/api.ts");
  const createHandler = panel.slice(panel.indexOf("const create"), panel.indexOf("const choose"));
  assert.match(api, /request\("\/api\/v1\/projects", \{ signal \}\)/);
  assert.match(api, /body: JSON\.stringify\(\{ name \}\)/);
  assert.match(api, /"\/api\/v1\/projects\/select"/);
  assert.match(api, /project_id: projectId/);
  assert.match(panel, /Create a project/);
  assert.match(panel, /await createProject/);
  assert.match(panel, /await selectProject/);
  assert.doesNotMatch(createHandler, /selectProject/);
  assert.match(createHandler, /await refresh/);
  assert.match(panel, /Finish the active GPU job/);
});

test("model search distinguishes training support and observes background downloads", async () => {
  const panel = await source("src/ModelSetupPanel.tsx");
  const api = await source("src/api.ts");
  assert.match(api, /\/api\/v1\/models\/search\?\$\{params\}/);
  assert.match(api, /\/api\/v1\/model\/status/);
  assert.match(api, /\/api\/v1\/reference-model/);
  assert.match(api, /\/api\/v1\/reference-model\/download/);
  assert.match(panel, /searchHuggingFaceModels/);
  assert.match(panel, /searchOpen/);
  assert.match(panel, /setSearchOpen\(false\)/);
  assert.match(panel, /searchOpen && query\.trim\(\)/);
  assert.match(panel, /result\.compatibility !== "supported"/);
  assert.match(panel, /Not verified for V1 training/);
  assert.match(panel, /downloadJob\?\.status/);
  assert.match(panel, /Download queued/);
  assert.match(panel, /Benchmark reference/);
  assert.match(panel, /It cannot become the training base/);
  assert.doesNotMatch(panel, /Math\.random|\bETA\b/i);
});

test("model setup detects and adopts existing local base models without downloading", async () => {
  const panel = await source("src/ModelSetupPanel.tsx");
  const api = await source("src/api.ts");
  const localHandler = panel.slice(panel.indexOf("const adoptLocalModel"), panel.indexOf("const downloadSelection"));
  assert.match(api, /request\("\/api\/v1\/models\/local", \{ signal \}\)/);
  assert.match(api, /request\("\/api\/v1\/model\/use-local"/);
  assert.match(api, /body: JSON\.stringify\(\{ candidate_id: candidateId \}\)/);
  for (const field of ["candidate_id", "catalog_key", "availability", "cache_label", "scanned_cache_count", "ignored_incomplete_count"]) assert.match(api, new RegExp(field));
  assert.match(panel, /getLocalModels/);
  assert.match(panel, /On this machine/);
  for (const state of ["Checking this machine", "Found locally", "No supported base models found", "Ready locally", "Could not check known model caches"]) assert.match(panel, new RegExp(state));
  assert.match(localHandler, /await useLocalModel\(localModel\.candidate_id\)/);
  assert.match(localHandler, /await getModelWorkspace\(\)/);
  assert.match(localHandler, /await scanLocalModels\(\)/);
  assert.doesNotMatch(localHandler, /downloadProjectModel|cache_dir|snapshot_path/);
});

test("repository search defaults selected GitHub results to the merged-PR workflow", async () => {
  const panel = await source("src/SourceSetupPanel.tsx");
  const api = await source("src/api.ts");
  for (const mode of ["accepted_changes", "practice_tasks", "reference_only", "evaluation_holdout"]) {
    assert.match(panel, new RegExp(mode));
  }
  assert.match(panel, /Repository or local folder/);
  assert.match(panel, /setModes\(\["accepted_changes"\]\)/);
  assert.match(panel, /searchGitHubRepositories/);
  assert.match(panel, /repository\.full_name/);
  assert.match(panel, /setRepositorySearchEnabled\(false\)/);
  assert.match(api, /\/api\/v1\/repositories\/search\?\$\{params\}/);
  assert.match(panel, /include: splitPatterns\(include\)/);
  assert.match(panel, /exclude: splitPatterns\(exclude\)/);
  assert.match(panel, /license_spdx/);
  assert.match(panel, /license_attribution/);
  assert.match(panel, /hasIntrinsicPurpose/);
  assert.match(panel, /disabled={connected !== true/);
  assert.match(panel, /Loading existing sources/);
  assert.match(api, /modes\?: SourceMode\[\]/);
  assert.match(api, /body: JSON\.stringify\(input\)/);
});

test("Data keeps manual examples and verifier authoring behind an advanced path", async () => {
  const panel = await source("src/SourceSetupPanel.tsx");
  const api = await source("src/api.ts");
  assert.match(api, /request\("\/api\/v1\/tasks", \{ signal \}\)/);
  assert.match(api, /export async function createAuthoredTask/);
  assert.match(api, /export async function removeAuthoredTask/);
  assert.match(api, /request\("\/api\/v1\/examples", \{ signal \}\)/);
  assert.match(api, /export async function createAuthoredExample/);
  assert.match(api, /export async function removeAuthoredExample/);
  assert.match(panel, /Advanced: add training records by hand/);
  assert.match(panel, /Most projects can skip this/);
  assert.match(panel, /Add an instruction and accepted answer/);
  assert.match(panel, /I confirm I have the right to use this accepted response for training/);
  assert.match(panel, /createAuthoredExample/);
  assert.match(panel, /deleteExample/);
  assert.match(panel, /Add a GRPO or evaluation task/);
  assert.match(panel, /held-out groups/);
  for (const field of [
    "Locked source",
    "What should the model change",
    "Working directory",
    "Build command",
    "Regression tests",
    "Hidden verifier folder",
    "Verifier command",
    "Report path in workspace",
  ]) assert.match(panel, new RegExp(field));
  assert.match(panel, /repository files are not silently converted into training examples/i);
  assert.match(panel, /AutoTrainer does not generate hidden tests or guess correctness/);
  for (const signal of ["build_passed", "regression_pass_rate", "task_pass_rate", "responsive_pass_rate", "design_rule_pass_rate", "code_quality_pass_rate"]) assert.match(panel, new RegExp(signal));
  assert.match(panel, /In Train, <strong>Check readiness/);
  assert.match(panel, /authoredTaskSplit/);
  assert.match(panel, /createAuthoredTask/);
  assert.match(panel, /deleteTask/);
  assert.match(panel, /Remove task/);
});

test("Data and Train share one truthful three-level GRPO evidence surface", async () => {
  const panel = await source("src/GrpoEvidencePanel.tsx");
  const api = await source("src/api.ts");
  const training = await source("src/TrainingMonitorPanel.tsx");
  for (const label of ["Overview", "Tasks", "Rollouts"]) assert.match(panel, new RegExp(`label: "${label}"`));
  for (const field of ["episode_id", "task_id", "tool_call_count", "tool_calls_by_name", "changed_file_count", "elapsed_seconds"]) assert.match(panel, new RegExp(field));
  for (const key of ["design_rules", "patch_quality", "regression_safety", "responsive_rules", "task_tests"]) assert.match(panel, new RegExp(key));
  assert.match(panel, /aria-pressed={value === item\.id}/);
  assert.match(panel, /getCurriculumWorkspace/);
  assert.match(api, /request\("\/api\/v1\/curriculum", \{ signal \}\)/);
  assert.match(panel, /task\.split === "train"/);
  assert.match(panel, /protected_holdout_count/);
  assert.match(panel, /function TaskDetail/);
  for (const aspect of ["Locked snapshot", "Bounded tools", "Hidden verifier", "Verifier and reward"]) assert.match(panel, new RegExp(aspect));
  assert.match(panel, /task\.checks/);
  assert.match(panel, /Latest retained rollout window/);
  assert.match(panel, /unmatched_observations/);
  assert.match(panel, /never fabricates progress/);
  assert.match(panel, /never change granularity/);
  assert.match(panel, /No activity is simulated here/);
  assert.match(training, /<GrpoEvidencePanel context="training"/);
  assert.doesNotMatch(panel, /Math\.random|type="range"|token counter|\bETA\b|estimated time/i);
});

test("Train owns one-click start and plots only observed training systems evidence", async () => {
  const panel = await source("src/TrainingMonitorPanel.tsx");
  const runtime = await source("src/RuntimeSetupPanel.tsx");
  const app = await source("src/App.tsx");
  const evidence = await source("src/GrpoEvidencePanel.tsx");
  const api = await source("src/api.ts");
  assert.match(panel, />Start training</);
  assert.match(panel, /Check readiness/);
  assert.match(panel, /Actual GPU training happens here/);
  assert.match(panel, /Learn from merged PRs/);
  assert.match(panel, /Practice with verified tasks/);
  assert.match(panel, /verified practice then continues that same adapter/);
  assert.match(panel, /onClick={onOpenData}>Open Data/);
  assert.match(panel, /const recipeCopy/);
  assert.match(panel, /await startTraining\(\)/);
  assert.match(panel, /await prepareProject\(\)/);
  assert.match(panel, /setEvents\(\[\]\);\s+\/\/ Readiness[\s\S]+setPreparation\(null\);\s+setJob\(await startTraining\(\)\)/);
  assert.match(panel, /\}, \[revision, job\?\.id\]\);/);
  assert.match(api, /\/api\/v1\/training\/events\?after=/);
  assert.match(panel, /getTrainingEvents\(cursorRef\.current/);
  assert.match(panel, /jobRolledOver/);
  assert.match(panel, /getTrainingEvents\(0, controller\.signal\)/);
  assert.match(panel, /const sftLoss = useMemo/);
  assert.match(panel, /teachingLoss={sftLoss}/);
  assert.match(panel, /systemTelemetry={systemTelemetry}/);
  for (const chart of ["Teaching loss", "GPU memory", "Optimization throughput", "Verified GRPO reward and rubric"]) assert.match(evidence, new RegExp(chart));
  for (const metric of ["observed_steps_per_second", "vram_allocated_gib", "vram_reserved_gib", "vram_limit_gib"]) assert.match(panel, new RegExp(metric));
  assert.match(evidence, /no completion time is inferred/);
  assert.match(panel, /Training receipt/);
  assert.match(app, /<RuntimeSetupPanel disabled={trainingActive}/);
  assert.match(app, /<RefinementResourcePanel disabled={trainingActive}/);
  assert.match(api, /request\("\/api\/v1\/runtime\/setup", \{ signal \}\)/);
  assert.match(api, /export async function applyRuntimeSetup/);
  for (const component of ["Python 3.11", "Pinned ML packages", "One CUDA GPU", "Container backend", "Pinned rollout image"]) assert.match(runtime, new RegExp(component));
  assert.match(runtime, /workspace\.actions\.map/);
  assert.match(runtime, /action\.command\.join/);
  assert.match(runtime, /applyRuntimeSetup\(action\.id\)/);
  assert.match(runtime, /Administrator approval is required/);
  assert.doesNotMatch(runtime, /exec\(|spawn\(|shell/i);
  assert.doesNotMatch(panel, /Practice reward and rubric|events\.slice\(-24\)|Durable event rail/);
  assert.doesNotMatch(panel, /Math\.random|type="range"|token counter|\bETA\b/i);
});

test("Train exposes adapter-only hard or soft VRAM governance", async () => {
  const panel = await source("src/RefinementResourcePanel.tsx");
  const api = await source("src/api.ts");
  assert.match(panel, /Adapter-only refinement/);
  assert.match(panel, /Base weights frozen/);
  assert.match(panel, /VRAM limit or target/);
  assert.match(panel, /validated minimum/);
  assert.match(panel, /min={minimumVram}/);
  assert.match(panel, /settings === null \? "Loading the model's VRAM requirements/);
  assert.match(panel, /minimum_vram_gib == null/);
  assert.doesNotMatch(panel, /minimum_vram_gib === null/);
  assert.match(panel, /Hard limit/);
  assert.match(panel, /CUDA allocator cap/);
  assert.match(panel, /Soft target/);
  assert.match(panel, /Training may use more/);
  assert.match(panel, /Save GPU setting/);
  assert.match(panel, /both hard limits and soft monitoring targets/);
  assert.match(api, /request\("\/api\/v1\/refinement", \{ signal \}\)/);
  assert.match(api, /minimum_vram_gib/);
  assert.match(api, /max_vram_gib/);
});

test("Evaluate freezes weights, runs once, and renders real trial and verifier evidence", async () => {
  const panel = await source("src/EvaluationMonitorPanel.tsx");
  const language = await source("src/LanguageEvaluationPanel.tsx");
  const api = await source("src/api.ts");
  assert.match(panel, /Weights are frozen\. Nothing learns here\./);
  assert.match(panel, /Run held-out evaluation/);
  assert.match(panel, /runnableReadinessStatuses = new Set\(\["ready", "inputs_ready"\]\)/);
  assert.match(panel, /runnableReadinessStatuses\.has\(workspace\?\.readiness\.status/);
  assert.match(panel, /const hasEvaluationEvidence/);
  assert.match(panel, /Nothing is missing or broken/);
  assert.match(panel, /await startEvaluation\(\)/);
  assert.match(api, /body: "\{\}"/);
  assert.match(api, /\/api\/v1\/evaluation\/events\?after=/);
  assert.match(panel, /benchmark\?\.trials/);
  assert.match(panel, /Planned trial matrix/);
  assert.match(panel, /Reward and verified success by model arm/);
  assert.match(panel, /Separate per-arm means/);
  assert.match(panel, /no values are pooled across models/);
  assert.match(panel, /Rubric components for one model/);
  assert.match(panel, /Did training improve the model\?/);
  assert.match(panel, /Candidate minus reference/);
  assert.match(panel, /confidence interval/);
  assert.match(panel, /report\.comparison\.candidates/);
  assert.match(api, /export type EvaluationReport/);
  assert.match(panel, /Live verification/);
  assert.match(panel, /planIdRef/);
  assert.match(panel, /page\.cursor_reset \|\| planRolledOver/);
  assert.match(panel, /hardGatePassed === false/);
  assert.match(panel, /<LanguageEvaluationPanel/);
  assert.match(api, /\/api\/v1\/evaluation\/language/);
  assert.match(language, /Language-matched code proof/);
  assert.match(language, /workspace\.available\.map/);
  assert.match(language, /Open benchmark inspiration/);
  assert.match(language, /Auto-detect from frozen dataset/);
  assert.match(panel, /resultsTruncated/);
  assert.match(panel, /trialsTruncated/);
  assert.match(panel, /visible-window means rather than whole-run means/);
  assert.doesNotMatch(panel, /planEvaluation|Math\.random|type="range"|token counter|\bETA\b/i);
});

test("Serve manages a real local OpenAI-compatible host and test request", async () => {
  const panel = await source("src/ServePanel.tsx");
  const api = await source("src/api.ts");
  assert.match(api, /\/api\/v1\/hosting\/start/);
  assert.match(api, /\/api\/v1\/hosting\/stop/);
  assert.match(api, /\/api\/v1\/hosting\/test/);
  assert.match(panel, /Start local endpoint/);
  assert.match(panel, /OpenAI-compatible endpoint/);
  assert.match(panel, /chat\/completions/);
  assert.match(panel, /Copy curl/);
  assert.match(panel, /Send test request/);
  assert.match(panel, /No response yet/);
  assert.match(panel, /adapterChoiceTouched/);
  assert.match(panel, /Backend resolved output/);
  assert.match(panel, /Loaded output/);
});

test("the first run walkthrough covers all five real lifecycle screens", async () => {
  const app = await source("src/App.tsx");
  const walkthrough = app.slice(app.indexOf("const walkthroughSteps"), app.indexOf("function Walkthrough"));
  assert.equal(walkthrough.match(/title: "/g)?.length, 5);
  for (const target of ["projects", "model", "train", "evaluate", "serve"]) assert.match(app, new RegExp(`data-tour="${target}"`));
  assert.match(app, /autotrainer\.walkthrough\.v3/);
  assert.match(app, /role="dialog"/);
  assert.match(app, /aria-modal="true"/);
  assert.match(app, /event\.key === "Escape"/);
  assert.match(app, /window\.localStorage\.setItem/);
});

test("charts render observed values and truthful empty states", async () => {
  const chart = await source("src/TelemetryChart.tsx");
  const visibleUi = await Promise.all([
    source("src/App.tsx"), source("src/GrpoEvidencePanel.tsx"), source("src/TrainingMonitorPanel.tsx"), source("src/EvaluationMonitorPanel.tsx"), source("src/ServePanel.tsx"),
  ]).then((files) => files.join("\n"));
  assert.match(chart, /plots only observed backend values/);
  assert.match(chart, /Waiting for observed values/);
  assert.match(chart, /item\.points\.map/);
  assert.doesNotMatch(visibleUi, /Math\.random|fake data|demo values|token counter|estimated time/i);
});
