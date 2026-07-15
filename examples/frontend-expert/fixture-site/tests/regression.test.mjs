import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("keeps the three-column desktop rule", async () => {
  const css = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");
  assert.match(css, /repeat\(3,\s*minmax\(0,\s*1fr\)\)/);
});

test("keeps all purchase actions", async () => {
  const source = await readFile(new URL("../src/main.tsx", import.meta.url), "utf8");
  assert.match(source, /Choose \{name\}/);
});
