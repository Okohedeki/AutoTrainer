import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("build emits a conventional static web application", async () => {
  const indexUrl = new URL("dist/index.html", root);
  await access(indexUrl);
  const html = await readFile(indexUrl, "utf8");
  assert.match(html, /<title>AutoTrainer · Single-GPU model foundry<\/title>/);
  assert.match(html, /<div id="root"><\/div>/);

  const assets = await readdir(new URL("dist/assets/", root));
  assert.ok(assets.some((name) => name.endsWith(".js")));
  assert.ok(assets.some((name) => name.endsWith(".css")));
});

test("the control plane exposes the required experiment decisions", async () => {
  const source = await readFile(new URL("src/App.tsx", root), "utf8");
  assert.match(source, /Build a better 9B frontend model/);
  assert.match(source, /Create first environment/);
  assert.match(source, /Model benchmark/);
  assert.match(source, /Fable website A\/B/);
  assert.match(source, /QLoRA/);
  assert.match(source, /GRPO/);
});
