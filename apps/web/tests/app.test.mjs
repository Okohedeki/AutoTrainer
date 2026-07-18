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
  assert.match(app, /<HistoryReviewPanel/);
  assert.match(app, /<TrainingMonitorPanel/);
  assert.match(app, /<EvaluationMonitorPanel/);
  assert.match(app, /<ServePanel/);
  assert.doesNotMatch(app, /<PreparePanel/);
  assert.doesNotMatch(app, /className="hero"|site-footer|CommandDrawer/);
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

test("repository search resolves names before explicit purpose and advanced scope", async () => {
  const panel = await source("src/SourceSetupPanel.tsx");
  const api = await source("src/api.ts");
  for (const mode of ["accepted_changes", "practice_tasks", "reference_only", "evaluation_holdout"]) {
    assert.match(panel, new RegExp(mode));
  }
  assert.match(panel, /Search GitHub or enter a local path/);
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

test("Train owns one-click start plus real loss, reward, rubric, and event telemetry", async () => {
  const panel = await source("src/TrainingMonitorPanel.tsx");
  const api = await source("src/api.ts");
  assert.match(panel, />Start training</);
  assert.match(panel, /Check readiness/);
  assert.match(panel, /Actual GPU training happens here/);
  assert.match(panel, /Accepted examples → QLoRA SFT/);
  assert.match(panel, /Executable tasks → GRPO/);
  assert.match(panel, /GRPO continues training the same adapter/);
  assert.match(panel, /onClick={onOpenData}>Open Data/);
  assert.match(panel, /const recipeCopy/);
  assert.match(panel, /await startTraining\(\)/);
  assert.match(panel, /await prepareProject\(\)/);
  assert.match(api, /\/api\/v1\/training\/events\?after=/);
  assert.match(panel, /getTrainingEvents\(cursorRef\.current/);
  assert.match(panel, /jobRolledOver/);
  assert.match(panel, /getTrainingEvents\(0, controller\.signal\)/);
  assert.match(panel, /event\.hard_gate_passed === false/);
  assert.match(panel, /Teaching loss/);
  assert.match(panel, /Practice reward and rubric/);
  for (const key of ["design_rules", "patch_quality", "regression_safety", "responsive_rules", "task_tests"]) assert.match(panel, new RegExp(key));
  assert.match(panel, /No training events yet/);
  assert.doesNotMatch(panel, /Math\.random|type="range"|token counter|\bETA\b/i);
});

test("Evaluate freezes weights, runs once, and renders real trial and verifier evidence", async () => {
  const panel = await source("src/EvaluationMonitorPanel.tsx");
  const api = await source("src/api.ts");
  assert.match(panel, /Weights are frozen\. Nothing learns here\./);
  assert.match(panel, /Run held-out evaluation/);
  assert.match(panel, /await startEvaluation\(\)/);
  assert.match(api, /body: "\{\}"/);
  assert.match(api, /\/api\/v1\/evaluation\/events\?after=/);
  assert.match(panel, /benchmark\?\.trials/);
  assert.match(panel, /Planned trial matrix/);
  assert.match(panel, /Reward and verified success/);
  assert.match(panel, /Rubric components/);
  assert.match(panel, /Live verification/);
  assert.match(panel, /planIdRef/);
  assert.match(panel, /page\.cursor_reset \|\| planRolledOver/);
  assert.match(panel, /hardGatePassed === false/);
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
    source("src/App.tsx"), source("src/TrainingMonitorPanel.tsx"), source("src/EvaluationMonitorPanel.tsx"), source("src/ServePanel.tsx"),
  ]).then((files) => files.join("\n"));
  assert.match(chart, /plots only observed backend values/);
  assert.match(chart, /Waiting for observed values/);
  assert.match(chart, /item\.points\.map/);
  assert.doesNotMatch(visibleUi, /Math\.random|fake data|demo values|token counter|estimated time/i);
});
