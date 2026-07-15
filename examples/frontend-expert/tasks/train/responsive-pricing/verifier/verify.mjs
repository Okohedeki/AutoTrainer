import { existsSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const workspace = "/workspace";

const fixture = path.join(workspace, "examples", "frontend-expert", "fixture-site");
const css = await readFile(path.join(fixture, "src", "styles.css"), "utf8");
const hasMobileColumn = /\.pricing-grid\s*\{[\s\S]*?grid-template-columns:\s*1fr/.test(css);
const hasDesktopBreakpoint = /@media\s*\(min-width:\s*768px\)[\s\S]*?repeat\(3,\s*minmax\(0,\s*1fr\)\)/.test(css);
const taskPassed = hasMobileColumn && hasDesktopBreakpoint;

const report = {
  build_passed: existsSync(path.join(fixture, "dist", "index.html")),
  regression_pass_rate: hasDesktopBreakpoint ? 1 : 0,
  task_pass_rate: taskPassed ? 1 : 0,
  responsive_pass_rate: taskPassed ? 1 : 0,
  design_rule_pass_rate: css.includes("var(--space)") ? 1 : 0,
  code_quality_pass_rate: css.includes("!important") ? 0 : 1
};

await writeFile("/workspace/.autotrainer-verifier-report.json", `${JSON.stringify(report, null, 2)}\n`);
