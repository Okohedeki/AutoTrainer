import { useEffect, useMemo, useState } from "react";
import { getTrainingJob, type TrainingJob } from "./api";

const liveStatuses = new Set<TrainingJob["status"]>(["queued", "running"]);

const stageNames: Record<NonNullable<TrainingJob["stage"]>, string> = {
  prepare: "Prepare",
  sft: "Teach from examples",
  grpo: "Practice against tests",
};

type TrainingStage = "prepare" | "sft" | "grpo";

function observedStages(job: TrainingJob): TrainingStage[] {
  if (job.status === "idle") return [];
  const stages: TrainingStage[] = ["prepare"];
  // While a run is live the durable record exposes only the stage it has
  // actually reached. Completion adds exactly the stages the trainer returned.
  const reached = job.result?.stages.map((stage) => stage.stage) ?? (job.stage && job.stage !== "prepare" ? [job.stage] : []);
  for (const stage of reached) {
    if (!stages.includes(stage)) stages.push(stage);
  }
  return stages;
}

function stageState(job: TrainingJob, stage: TrainingStage) {
  const completedStages = new Set(job.result?.stages.map((item) => item.stage) ?? []);
  if (stage === "prepare" && (job.stage === "sft" || job.stage === "grpo" || job.status === "completed")) {
    return "complete";
  }
  if (completedStages.has(stage as "sft" | "grpo")) return "complete";
  if (liveStatuses.has(job.status) && job.stage === stage) return "active";
  if (job.status === "failed" && job.stage === stage) return "failed";
  return "waiting";
}

function readableMetric(key: string) {
  return key.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

// The monitor renders only the durable job record exposed by the shared Python
// service. There is no guessed percentage, ETA, loss curve, or hidden browser job.
export default function TrainingMonitorPanel({ onOpenSetup }: { onOpenSetup: () => void }) {
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stopped = false;
    const controller = new AbortController();
    let timer = 0;

    const refresh = async () => {
      try {
        const next = await getTrainingJob(controller.signal);
        if (stopped) return;
        setJob(next);
        setError(null);
      } catch (reason) {
        if (stopped || controller.signal.aborted) return;
        setError(reason instanceof Error ? reason.message : "Training status could not be refreshed.");
      } finally {
        // Schedule after completion so a slow request cannot be overtaken by a
        // newer poll and regress a terminal job back to an older live state.
        if (!stopped) timer = window.setTimeout(() => void refresh(), 2_000);
      }
    };

    void refresh();
    return () => {
      stopped = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, []);

  const metricRows = useMemo(() => (
    job?.result?.stages.flatMap((stage) =>
      Object.entries(stage.metrics ?? {}).map(([name, value]) => ({ stage: stage.stage, name, value })),
    ) ?? []
  ), [job]);

  const statusTone = job?.status === "completed"
    ? "good"
    : job?.status === "failed" || job?.status === "interrupted"
      ? "danger"
      : liveStatuses.has(job?.status ?? "idle")
        ? "info"
        : "muted";

  return (
    <section className="run-workspace" aria-labelledby="training-monitor-heading">
      <article className="panel run-monitor-panel">
        <header className="panel-header run-monitor-header">
          <div>
            <p className="panel-kicker">Current job</p>
            <h2 id="training-monitor-heading">Training monitor</h2>
          </div>
          <span className={`status-chip ${statusTone}`}>{job?.status ?? "connecting"}</span>
        </header>

        {error && <div className="source-error" role="alert">{error}</div>}

        {job && (
          <>
            <div className="run-message" role={job.status === "failed" || job.status === "interrupted" ? "alert" : "status"} aria-live="polite">
              <span className={`health-dot ${statusTone}`} aria-hidden="true" />
              <div>
                <strong>{job.stage ? stageNames[job.stage] : "No training job"}</strong>
                <p>{job.message}</p>
              </div>
              {job.id && <code>{job.id.slice(0, 12)}</code>}
            </div>

            <ol className="stage-list" aria-label="Observed training stages">
              {observedStages(job).map((stage, index) => {
                const state = stageState(job, stage);
                return (
                  <li className={`stage-row ${state}`} key={stage}>
                    <span className="stage-marker" aria-hidden="true">
                      {state === "complete" ? "OK" : state === "failed" ? "!" : index + 1}
                    </span>
                    <div className="stage-copy">
                      <strong>{stageNames[stage]}</strong>
                      <p>{stage === "prepare"
                        ? "Validate the project, compile approved data, and check the local runtime."
                        : stage === "sft"
                          ? "Train the adapter on examples you accepted."
                          : "Improve the adapter against executable rewards when tasks are available."}</p>
                    </div>
                    <span className={`status-chip ${state === "complete" ? "good" : state === "active" ? "info" : state === "failed" ? "danger" : "muted"}`}>
                      {state}
                    </span>
                  </li>
                );
              })}
            </ol>
          </>
        )}

        {job?.status === "idle" && (
          <div className="monitor-empty">
            <strong>No run yet</strong>
            <p>Finish setup and Prepare will prove the exact local training path before it can start.</p>
            <button className="primary-button" type="button" onClick={onOpenSetup}>Open setup</button>
          </div>
        )}
      </article>

      <aside className="panel run-evidence-panel">
        <header className="panel-header">
          <div><p className="panel-kicker">Recorded evidence</p><h2>Outputs</h2></div>
        </header>
        {!job?.result && (
          <div className="evidence-empty">
            <strong>Nothing fabricated</strong>
            <p>Metrics and adapter paths appear here only after the trainer writes them.</p>
          </div>
        )}
        {job?.result?.stages.map((stage) => (
          <section className="stage-result" key={stage.stage}>
            <div><strong>{stageNames[stage.stage]}</strong><span className="status-chip good">complete</span></div>
            {stage.output_dir && <code>{stage.output_dir}</code>}
            {stage.trainable_adapter_parameters !== undefined && (
              <p>{stage.trainable_adapter_parameters.toLocaleString()} trainable adapter parameters</p>
            )}
          </section>
        ))}
        {metricRows.length > 0 && (
          <dl className="metric-list">
            {metricRows.map((metric) => (
              <div key={`${metric.stage}-${metric.name}`}>
                <dt>{readableMetric(metric.name)}</dt>
                <dd>{String(metric.value)}</dd>
              </div>
            ))}
          </dl>
        )}
      </aside>
    </section>
  );
}
