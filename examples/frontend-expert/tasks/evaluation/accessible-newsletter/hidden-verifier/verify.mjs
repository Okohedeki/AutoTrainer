import { existsSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { scoreSources } from "./scoring.mjs";

const workspace = path.resolve(process.argv[2] ?? process.env.AUTOTRAINER_WORKSPACE ?? "/workspace");
const fixture = path.join(workspace, "examples", "frontend-expert", "evaluation-site");
// The environment removes any policy-supplied report immediately before this
// trusted process runs; command-line overrides keep the verifier independently
// testable by maintainers.
const reportPath = path.resolve(
  process.argv[3] ?? process.env.AUTOTRAINER_REPORT_PATH ?? path.join(workspace, ".autotrainer-verifier-report.json"),
);
const [jsx, css] = await Promise.all([
  readFile(path.join(fixture, "src", "main.tsx"), "utf8"),
  readFile(path.join(fixture, "src", "styles.css"), "utf8"),
]);

const report = {
  build_passed: existsSync(path.join(fixture, "dist", "index.html")),
  ...scoreSources(jsx, css),
  verifier_version: "newsletter-accessibility-v1",
};

await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
