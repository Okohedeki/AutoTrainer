import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { scoreSources } from "./scoring.mjs";

const fixture = new URL("../../../../evaluation-site/", import.meta.url);

test("baseline preserves regressions while exposing independent task deficits", async () => {
  const [jsx, css] = await Promise.all([
    readFile(new URL("src/main.tsx", fixture), "utf8"),
    readFile(new URL("src/styles.css", fixture), "utf8"),
  ]);
  const score = scoreSources(jsx, css);
  assert.equal(score.regression_pass_rate, 1);
  assert.equal(score.diagnostics.accessibility.associated_email_label, false);
  assert.equal(score.diagnostics.accessibility.announced_feedback, false);
  assert.equal(score.diagnostics.responsive.narrow_breakpoint, false);
  assert.ok(score.task_pass_rate > 0 && score.task_pass_rate < 1);
  assert.ok(score.responsive_pass_rate > 0 && score.responsive_pass_rate < 1);
});
test("multi-signal scores improve for semantic and responsive repairs", async () => {
  const [baselineJsx, baselineCss] = await Promise.all([
    readFile(new URL("src/main.tsx", fixture), "utf8"),
    readFile(new URL("src/styles.css", fixture), "utf8"),
  ]);
  const jsx = baselineJsx
    .replace('<form className="signup-form"', '<form className="signup-form" aria-labelledby="signup-title"')
    .replace('<span className="form-label">Email address</span>', '<label className="form-label" htmlFor="subscriber-email">Email address</label>')
    .replace('name="email"\n                  type="email"', 'name="email"\n                  autoComplete="email"\n                  type="email"')
    .replace('<p className="form-status">{message}</p>', '<p className="form-status" role="status" aria-live="polite">{message}</p>');
  const css = `${baselineCss}
.form-controls input:focus-visible { outline: 3px solid var(--peach); outline-offset: 3px; }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
@media (max-width: 48rem) {
  .hero { grid-template-columns: 1fr; }
  .form-controls { grid-template-columns: 1fr; }
  .issue-grid { grid-template-columns: 1fr; }
  .form-controls button { min-height: 2.75rem; }
}
`;
  const before = scoreSources(baselineJsx, baselineCss);
  const after = scoreSources(jsx, css);
  assert.equal(after.responsive_pass_rate, 1);
  assert.equal(after.diagnostics.accessibility.associated_email_label, true);
  assert.equal(after.diagnostics.accessibility.announced_feedback, true);
  assert.ok(after.task_pass_rate > before.task_pass_rate);
  assert.ok(after.design_rule_pass_rate > before.design_rule_pass_rate);
});
