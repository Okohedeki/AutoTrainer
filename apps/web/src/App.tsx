import { useMemo, useState } from "react";
import {
  defaultEnvironment,
  modelCatalog,
  pipelineStages,
  rewardSignals,
} from "./data";

const benchmarkRows = [
  { name: "Base 9B", detail: "Untouched reference", tone: "neutral" },
  { name: "QLoRA", detail: "Supervised warm start", tone: "blue" },
  { name: "RL adapter", detail: "Executable reward", tone: "green" },
];

export default function App() {
  const [selectedModelId, setSelectedModelId] = useState(modelCatalog[0].id);
  const [showCommands, setShowCommands] = useState(false);

  const selectedModel = useMemo(
    () => modelCatalog.find((model) => model.id === selectedModelId) ?? modelCatalog[0],
    [selectedModelId],
  );

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="AutoTrainer home">
          <span className="brand-mark" aria-hidden="true">A</span>
          <span>AutoTrainer</span>
        </a>
        <div className="topbar-status">
          <span className="status-dot" aria-hidden="true" />
          Local mode
          <span className="topbar-divider" aria-hidden="true" />
          RTX 4090 · 24 GB
        </div>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">Single-GPU model foundry</p>
          <h1>Build a better 9B frontend model.</h1>
          <p className="hero-description">
            Warm-start with QLoRA, improve inside executable RL environments,
            then prove the result against the base model and inside Fable.
          </p>
        </div>
        <div className="hero-proof" aria-label="Training constraint">
          <span className="proof-number">01</span>
          <div>
            <strong>GPU from start to finish</strong>
            <p>No cloud trainer. No distributed cluster. One reproducible run.</p>
          </div>
        </div>
      </section>

      <section className="workspace" aria-label="Experiment setup">
        <article className="panel setup-panel">
          <div className="panel-heading">
            <div>
              <p className="section-label">Experiment 001</p>
              <h2>Frontend design expert</h2>
            </div>
            <span className={showCommands ? "pill pill-ready" : "pill"}>
              {showCommands ? "CLI shown" : "Read-only preview"}
            </span>
          </div>

          <div className="field-grid">
            <label className="field">
              <span>Base model</span>
              <select
                value={selectedModelId}
                onChange={(event) => setSelectedModelId(event.target.value)}
              >
                {modelCatalog.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}{model.status === "custom" ? " · configure in YAML" : ""}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Specialization</span>
              <input value="Frontend design" readOnly aria-label="Specialization" />
            </label>
          </div>

          <div className="model-card">
            <div className="model-monogram" aria-hidden="true">9B</div>
            <div className="model-copy">
              <div className="model-title-row">
                <strong>{selectedModel.shortName}</strong>
                <span>{selectedModel.id}</span>
              </div>
              <p>{selectedModel.description}</p>
            </div>
            <dl className="model-specs">
              <div><dt>Weights</dt><dd>4-bit</dd></div>
              <div><dt>Adapter</dt><dd>QLoRA</dd></div>
              <div><dt>Target</dt><dd>24 GB</dd></div>
            </dl>
          </div>

          <div className="recipe">
            <div className="recipe-step active"><span>1</span>Baseline</div>
            <i aria-hidden="true" />
            <div className="recipe-step"><span>2</span>QLoRA</div>
            <i aria-hidden="true" />
            <div className="recipe-step"><span>3</span>GRPO</div>
          </div>

          <button
            className="primary-action"
            type="button"
            onClick={() => setShowCommands((current) => !current)}
          >
            {showCommands ? "Hide project commands" : "Show project commands"}
            <span aria-hidden="true">→</span>
          </button>
          <p className="action-note">
            {showCommands
              ? "autotrainer init → source add → compile → doctor → train sft → train rl"
              : "The dashboard does not start jobs yet. autotrainer.yaml and the CLI are the source of truth."}
          </p>
        </article>

        <article className="panel environment-panel">
          <div className="panel-heading compact">
            <div>
              <p className="section-label">RL environment</p>
              <h2>{defaultEnvironment.id}</h2>
            </div>
            <span className="code-badge">v1.0</span>
          </div>

          <div className="environment-stack">
            {defaultEnvironment.stack.map((item) => <span key={item}>{item}</span>)}
          </div>

          <div className="task-brief">
            <span>Task brief</span>
            <p>{defaultEnvironment.task}</p>
          </div>

          <div className="environment-details">
            <div>
              <span className="detail-label">Tools</span>
              <ul>
                {defaultEnvironment.tools.map((tool) => <li key={tool}>{tool}</li>)}
              </ul>
            </div>
            <div>
              <span className="detail-label">Limits</span>
              <dl>
                <div><dt>Tool calls</dt><dd>{defaultEnvironment.limits.toolCalls}</dd></div>
                <div><dt>Token budget</dt><dd>12k</dd></div>
                <div><dt>Network</dt><dd>Blocked</dd></div>
              </dl>
            </div>
          </div>

          <div className="gate-row">
            <span className="gate-icon" aria-hidden="true">×</span>
            <div><strong>Hard gates</strong><p>Build and regression suite must pass before reward.</p></div>
          </div>
        </article>
      </section>

      <section className="pipeline-section" aria-labelledby="pipeline-title">
        <div className="section-heading-row">
          <div>
            <p className="section-label">End-to-end proof</p>
            <h2 id="pipeline-title">One model. Three checkpoints. Two decisions.</h2>
          </div>
          <p>Every stage earns its place on held-out work.</p>
        </div>
        <ol className="pipeline-list">
          {pipelineStages.map((stage, index) => (
            <li key={stage.id}>
              <span className="stage-number">0{index + 1}</span>
              <strong>{stage.label}</strong>
              <p>{stage.detail}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="results-grid">
        <article className="panel benchmark-panel">
          <div className="panel-heading compact">
            <div>
              <p className="section-label">Decision one</p>
              <h2>Model benchmark</h2>
            </div>
            <span className="pill">Awaiting baseline</span>
          </div>
          <div className="benchmark-table" role="table" aria-label="Checkpoint benchmark">
            <div className="benchmark-header" role="row">
              <span role="columnheader">Checkpoint</span>
              <span role="columnheader">Verified success</span>
              <span role="columnheader">Result</span>
            </div>
            {benchmarkRows.map((row) => (
              <div className="benchmark-row" role="row" key={row.name}>
                <span role="cell"><i className={`model-dot ${row.tone}`} /> <strong>{row.name}</strong><small>{row.detail}</small></span>
                <span role="cell" className="metric-empty">—</span>
                <span role="cell" className="result-empty">Not run</span>
              </div>
            ))}
          </div>
        </article>

        <article className="panel fable-panel">
          <div className="panel-heading compact">
            <div>
              <p className="section-label">Decision two</p>
              <h2>Fable website A/B</h2>
            </div>
            <span className="code-badge">Blind review</span>
          </div>
          <div className="versus">
            <div><span>A</span><strong>Fable + Base 9B</strong><small>Control</small></div>
            <b>vs</b>
            <div><span>B</span><strong>Fable + RL winner</strong><small>Candidate</small></div>
          </div>
          <p className="fable-note">
            Same briefs, tools, orchestration, token budget, and time limit. Human preference decides whether the improvement is visible in real work.
          </p>
        </article>
      </section>

      <section className="reward-section" aria-labelledby="reward-title">
        <div className="section-heading-row">
          <div>
            <p className="section-label">Deterministic reward</p>
            <h2 id="reward-title">What the adapter is allowed to optimize</h2>
          </div>
          <p>Signals stay separate so reward hacking is visible.</p>
        </div>
        <div className="reward-grid">
          {rewardSignals.map((signal) => (
            <article key={signal.id}>
              <span>{Math.round(signal.weight * 100)}%</span>
              <strong>{signal.label}</strong>
              <p>{signal.description}</p>
              <i style={{ width: `${signal.weight * 100}%` }} aria-hidden="true" />
            </article>
          ))}
        </div>
      </section>

      <footer>
        <span>AutoTrainer · local development</span>
        <span>Base → QLoRA → RL → proof</span>
      </footer>
    </main>
  );
}
