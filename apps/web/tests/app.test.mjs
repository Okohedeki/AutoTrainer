import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

async function source(path) {
  return readFile(new URL(path, root), "utf8");
}

test("build emits the focused AutoTrainer application", async () => {
  const indexUrl = new URL("dist/index.html", root);
  await access(indexUrl);
  const html = await readFile(indexUrl, "utf8");
  assert.match(html, /<title>AutoTrainer · Train a local specialist<\/title>/);
  assert.match(html, /<div id="root"><\/div>/);

  const assets = await readdir(new URL("dist/assets/", root));
  assert.ok(assets.some((name) => name.endsWith(".js")));
  assert.ok(assets.some((name) => name.endsWith(".css")));
});

test("the page is one clear three-step setup flow", async () => {
  const app = await source("src/App.tsx");
  assert.match(app, /Make a small model excellent at your work/);
  assert.match(app, /<ModelSetupPanel \/>/);
  assert.match(app, /<SourceSetupPanel onSourcesChanged=\{sourcesChanged\} \/>/);
  assert.match(app, /<HistoryReviewPanel refreshKey=\{sourceRevision\} \/>/);
  assert.match(app, /<PreparePanel \/>/);
  assert.ok(app.indexOf("<SourceSetupPanel") < app.indexOf("<HistoryReviewPanel"));
  assert.ok(app.indexOf("<HistoryReviewPanel") < app.indexOf("<PreparePanel />"));
  assert.match(app, /aria-label="Training setup"/);
  assert.doesNotMatch(app, /sidebar|CommandDrawer|Training runs|Training overview/);
  await assert.rejects(access(new URL("src/data.ts", root)));
});

test("model selection and download use the real shared lifecycle", async () => {
  const panel = await source("src/ModelSetupPanel.tsx");
  const api = await source("src/api.ts");
  const vite = await source("vite.config.ts");
  assert.match(panel, /<h2 id="model-heading">Choose model<\/h2>/);
  assert.match(panel, /Select & download/);
  assert.match(panel, /await selectProjectModel/);
  assert.match(panel, /await downloadProjectModel/);
  assert.match(panel, /<details className="advanced-options">/);
  assert.match(panel, /<summary>Advanced<\/summary>/);
  assert.match(panel, /Save settings/);
  assert.match(api, /\/api\/v1\/model\/select/);
  assert.match(api, /\/api\/v1\/model\/download/);
  assert.match(vite, /127\.0\.0\.1:8765/);
});

test("work is added and removed through the real source contract", async () => {
  const panel = await source("src/SourceSetupPanel.tsx");
  const api = await source("src/api.ts");
  assert.match(panel, /<h2 id="source-setup-heading">Add your work<\/h2>/);
  assert.match(panel, /GitHub URL or local path/);
  assert.match(panel, /await addProjectSource/);
  assert.match(panel, /await removeProjectSource/);
  assert.match(panel, /aria-label={`Remove \${source\.label}`}/);
  assert.match(api, /GET|ProjectSource/);
  assert.match(api, /\/api\/v1\/sources/);
  assert.match(api, /method: "DELETE"/);
  assert.match(panel, /onSourcesChanged\?\.\(\)/);
});

test("reviewed history approves one real change at a time", async () => {
  const panel = await source("src/HistoryReviewPanel.tsx");
  const api = await source("src/api.ts");
  assert.match(panel, /workspace\?\.candidates\[0\]/);
  assert.match(panel, /\[refreshKey\]/);
  assert.doesNotMatch(panel, /workspace\?\.candidates\.map|workspace\.candidates\.map/);
  assert.match(panel, /<textarea/);
  assert.match(panel, /<pre className="history-diff"/);
  assert.match(panel, /type="checkbox"/);
  assert.match(panel, /I have the right to use this change for training/);
  assert.match(panel, /Approve example/);
  assert.match(panel, />\s*Skip\s*</);
  assert.match(panel, /review\("approved"\)/);
  assert.match(panel, /review\("rejected"\)/);
  assert.match(
    panel,
    /!candidate && workspace\.summary\.approved_count === 0 && workspace\.summary\.stale_review_count === 0/,
  );
  assert.doesNotMatch(panel, /role="dialog"|modal|bulk|ranking|filter/i);
  assert.match(api, /\/api\/v1\/history/);
  assert.match(api, /\/api\/v1\/history\/review/);
  assert.match(api, /\/api\/v1\/history\/retire-stale/);
  assert.match(api, /rights_confirmed\?: boolean/);
  assert.match(panel, /Retire old approval/);
  assert.match(panel, /retireStaleHistoryReviews/);
});

test("one preparation action returns an honest next step", async () => {
  const panel = await source("src/PreparePanel.tsx");
  const api = await source("src/api.ts");
  assert.match(panel, /<h2 id="prepare-heading">Prepare training<\/h2>/);
  assert.match(panel, /await prepareProject/);
  assert.match(panel, /Teach from accepted work/);
  assert.match(panel, /Practice against tests/);
  assert.match(panel, /Action needed/);
  assert.match(panel, /Do this next/);
  assert.match(api, /\/api\/v1\/prepare/);
});

test("training starts only after readiness and follows the real job record", async () => {
  const panel = await source("src/PreparePanel.tsx");
  const api = await source("src/api.ts");
  assert.match(api, /export type TrainingJob/);
  assert.match(api, /status: "idle" \| "queued" \| "running" \| "completed" \| "failed"/);
  assert.match(api, /request\("\/api\/v1\/training", \{ signal \}\)/);
  assert.match(api, /request\("\/api\/v1\/training\/start", \{ method: "POST", body: "\{\}" \}\)/);
  assert.match(panel, /result\.status === "ready"/);
  assert.match(panel, /await startTraining\(\)/);
  assert.match(panel, /window\.setInterval/);
  assert.match(panel, /2_000/);
  assert.match(panel, /\)\}\s*\{showTrainingControl && \(/);
  assert.match(panel, /Teaching from examples/);
  assert.match(panel, /Practicing against tests/);
  assert.match(panel, /trainingJob\.message/);
  assert.match(panel, /Training output ready/);
  assert.match(panel, /Retry training/);
  assert.doesNotMatch(panel, /type="range"|percentage|progress bar|view logs|learning rate|rank|alpha/i);
});

test("the first run has exactly three accessible walkthrough steps", async () => {
  const app = await source("src/App.tsx");
  assert.equal(app.match(/title: "/g)?.length, 3);
  assert.match(app, /autotrainer\.walkthrough\.v2/);
  assert.match(app, /data-tour="model"/);
  assert.match(app, /data-tour="sources"/);
  assert.match(app, /data-tour="prepare"/);
  assert.match(app, /role="dialog"/);
  assert.match(app, /aria-modal="true"/);
  assert.match(app, /event\.key === "Escape"/);
  assert.match(app, /inert={walkthroughOpen \? true : undefined}/);
  assert.match(app, /Walkthrough/);
  assert.match(app, /window\.localStorage\.setItem/);
});

test("normal UI stays plain, truthful, and free of research jargon", async () => {
  const visibleUi = await Promise.all([
    source("src/App.tsx"),
    source("src/ModelSetupPanel.tsx"),
    source("src/SourceSetupPanel.tsx"),
    source("src/HistoryReviewPanel.tsx"),
    source("src/PreparePanel.tsx"),
    source("index.html"),
  ]).then((files) => files.join("\n"));

  assert.doesNotMatch(visibleUi, /QLoRA|GRPO|Fable|control plane|training lab|dashboard/);
  assert.doesNotMatch(visibleUi, /status:\s*["']running["']|training started|job queued/i);
  assert.match(visibleUi, /Training never starts by accident/);
});
