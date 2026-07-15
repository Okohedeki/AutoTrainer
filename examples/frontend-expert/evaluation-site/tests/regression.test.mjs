import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const sourceUrl = new URL("../src/main.tsx", import.meta.url);
const stylesUrl = new URL("../src/styles.css", import.meta.url);

test("preserves the publication identity and core editorial copy", async () => {
  const source = await readFile(sourceUrl, "utf8");
  for (const phrase of [
    "Signal &amp; Story",
    "Make space for ideas worth keeping.",
    "A calmer way to follow the ideas shaping creative work.",
    "The next note arrives Thursday.",
  ]) {
    assert.ok(source.includes(phrase), `missing preserved phrase: ${phrase}`);
  }
});
test("retains the three held-out archive entries", async () => {
  const source = await readFile(sourceUrl, "utf8");
  const issueEntries = source.match(/number:\s*"Issue \d+"/g) ?? [];
  assert.equal(issueEntries.length, 3);
  for (const title of ["The patient city", "Tools that leave room", "A practice of noticing"]) {
    assert.ok(source.includes(title), `missing archive title: ${title}`);
  }
});

test("keeps subscription feedback local and deterministic", async () => {
  const source = await readFile(sourceUrl, "utf8");
  assert.match(source, /event\.preventDefault\(\)/);
  assert.match(source, /new FormData\(event\.currentTarget\)/);
  assert.match(source, /setMessage\(/);
  assert.doesNotMatch(source, /\b(?:fetch|XMLHttpRequest|axios)\b/);
});

test("preserves the wide editorial composition", async () => {
  const css = await readFile(stylesUrl, "utf8");
  assert.match(css, /\.hero\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1\.15fr\)\s+minmax\(20rem,\s*0\.85fr\)/s);
  assert.match(css, /\.issue-grid\s*\{[^}]*grid-template-columns:\s*repeat\(3,\s*minmax\(12rem,\s*1fr\)\)/s);
});
