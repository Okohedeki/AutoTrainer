import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("build emits a conventional static web application", async () => {
  const indexUrl = new URL("dist/index.html", root);
  await access(indexUrl);
  const html = await readFile(indexUrl, "utf8");
  assert.match(html, /<title>AutoTrainer · Local training control plane<\/title>/);
  assert.match(html, /<div id="root"><\/div>/);

  const assets = await readdir(new URL("dist/assets/", root));
  assert.ok(assets.some((name) => name.endsWith(".js")));
  assert.ok(assets.some((name) => name.endsWith(".css")));
});

test("the control plane exposes truthful training operations", async () => {
  const source = await readFile(new URL("src/App.tsx", root), "utf8");
  const snapshot = await readFile(new URL("src/data.ts", root), "utf8");
  assert.match(source, /Training overview/);
  assert.match(source, /Training runs/);
  assert.match(source, /Backend not connected/);
  assert.match(source, /Configured ≠ downloaded ≠ trained ≠ verified/);
  assert.match(snapshot, /Model benchmark/);
  assert.match(snapshot, /Fable A\/B/);
  assert.match(source, /QLoRA/);
  assert.match(source, /GRPO/);
  assert.doesNotMatch(source, /Build a better 9B frontend model/);
});

test("the example snapshot cannot imply a downloaded or active model", async () => {
  const snapshot = await readFile(new URL("src/data.ts", root), "utf8");
  assert.match(snapshot, /cache: "Not downloaded"/);
  assert.match(snapshot, /label: "Training stack", value: "Blocked"/);
  assert.match(snapshot, /Configured: base Qwen3\.5 9B vs AutoTrainer adapter/);
  assert.match(snapshot, /Deferred baseline: Qwythos 9B Claude Mythos/);
  assert.doesNotMatch(snapshot, /Qwythos 9B reference vs AutoTrainer adapter/);
  assert.doesNotMatch(snapshot, /status: "running"/i);
  assert.doesNotMatch(snapshot, /cache: "Downloaded"/);
});

test("navigation and the command drawer keep their accessible contracts", async () => {
  const source = await readFile(new URL("src/App.tsx", root), "utf8");
  assert.match(source, /aria-label=\{item\.label\}/);
  assert.match(source, /inert=\{drawerOpen \? true : undefined\}/);
  assert.match(source, /event\.key === "Escape"/);
  assert.match(source, /previouslyFocused\?\.focus\(\)/);
});
