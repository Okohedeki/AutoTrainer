import { useEffect, useMemo, useRef, useState } from "react";
import {
  ApiClientError,
  getTrainingEvents,
  getTrainingJob,
  prepareProject,
  startTraining,
  type PreparationResult,
  type TrainingEvent,
  type TrainingJob,
} from "./api";
import GrpoEvidencePanel from "./GrpoEvidencePanel";
import type { ChartPoint } from "./TelemetryChart";

const liveStatuses = new Set<TrainingJob["status"]>(["queued", "running"]);

const stageNames: Record<string, string> = {
  prepare: "Prepare run",
  sft: "Teach from examples",
  grpo: "Practice against the rubric",
};

// Readiness returns a recipe derived only from the configured data. Keeping
// this mapping beside the Train UI makes the exact optimization work explicit.
const recipeCopy: Record<PreparationResult["recipe"], { label: string; detail: string }> = {
  teach: {
    label: "QLoRA supervised fine-tuning (SFT)",
    detail: "The model learns from instruction-and-accepted-response examples.",
  },
  practice: {
    label: "Verifier-backed GRPO",
    detail: "The model attempts resettable tasks and learns from executable rewards.",
  },
  both: {
    label: "QLoRA SFT, then verifier-backed GRPO",
    detail: "Examples teach the adapter first; verified practice then continues that same adapter.",
  },
  needs_training_data: {
    label: "Training path not selected",
    detail: "Add accepted examples, executable tasks, or both in Data.",
  },
};

function numeric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function metric(event: TrainingEvent, keys: string[]) {
  for (const key of keys) {
    const value = numeric(event.metrics?.[key]);
    if (value !== null) return value;
  }
  return null;
}

function pointsFor(events: TrainingEvent[], read: (event: TrainingEvent) => number | null): ChartPoint[] {
  const points: ChartPoint[] = [];
  for (const event of events) {
    const value = read(event);
    if (value === null) continue;
    points.push({ x: event.step ?? points.length + 1, y: value });
  }
  return points;
}

// Train owns the one-click start action. The backend repeats the complete
// preflight inside that action; Check readiness is optional evidence, not a gate.
export default function TrainingMonitorPanel({
  revision = 0,
  onOpenData,
  onTrainingActiveChange,
}: {
  revision?: number;
  onOpenData: () => void;
  onTrainingActiveChange?: (active: boolean) => void;
}) {
  const cursorRef = useRef(0);
  const jobIdRef = useRef<string | null>(null);
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [events, setEvents] = useState<TrainingEvent[]>([]);
  const [preparation, setPreparation] = useState<PreparationResult | null>(null);
  const [action, setAction] = useState<"check" | "start" | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setPreparation(null);
  }, [revision]);

  useEffect(() => {
    let stopped = false;
    const controller = new AbortController();
    let timer = 0;

    const poll = async () => {
      try {
        const [nextJob, initialPage] = await Promise.all([
          getTrainingJob(controller.signal),
          getTrainingEvents(cursorRef.current, controller.signal),
        ]);
        if (stopped) return;
        let page = initialPage;
        const nextJobId = nextJob.id ?? initialPage.job_id;
        const jobRolledOver = jobIdRef.current !== null && nextJobId !== jobIdRef.current;
        const pageBelongsToOtherJob = Boolean(
          nextJob.id && initialPage.job_id && nextJob.id !== initialPage.job_id,
        );
        if (jobRolledOver || pageBelongsToOtherJob) {
          // Training sequences restart for each job. A cursor from the prior
          // job can sit beyond every event in the new job, so reconnect from
          // zero before adopting the replacement stream.
          page = await getTrainingEvents(0, controller.signal);
          if (stopped) return;
        }
        setJob(nextJob);
        const pageJobId = page.job_id ?? nextJob.id;
        if (jobRolledOver || pageBelongsToOtherJob || jobIdRef.current !== pageJobId || page.truncated) {
          jobIdRef.current = pageJobId;
          setEvents(page.events);
        } else if (page.events.length > 0) {
          setEvents((current) => {
            const merged = new Map(current.map((event) => [event.sequence, event]));
            page.events.forEach((event) => merged.set(event.sequence, event));
            return [...merged.values()].sort((a, b) => a.sequence - b.sequence).slice(-500);
          });
        }
        cursorRef.current = page.cursor;
        setError(null);
      } catch (reason) {
        if (!stopped && !controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Training status could not be refreshed.");
      } finally {
        if (!stopped) timer = window.setTimeout(() => void poll(), 2_000);
      }
    };

    void poll();
    return () => {
      stopped = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, []);

  const trainingActive = liveStatuses.has(job?.status ?? "idle");
  useEffect(() => onTrainingActiveChange?.(trainingActive), [onTrainingActiveChange, trainingActive]);

  const check = async () => {
    setAction("check");
    setError(null);
    try {
      setPreparation(await prepareProject());
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "Readiness could not be checked.");
    } finally {
      setAction(null);
    }
  };

  const start = async () => {
    setAction("start");
    setError(null);
    try {
      cursorRef.current = 0;
      jobIdRef.current = null;
      setEvents([]);
      setJob(await startTraining());
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "Training could not start.");
    } finally {
      setAction(null);
    }
  };

  const sftLoss = useMemo(() => pointsFor(
    events.filter((event) => event.stage === "sft"),
    (event) => metric(event, ["loss", "train_loss", "observed_loss"]),
  ), [events]);

  const completedStages = new Set(job?.result?.stages.map((stage) => stage.stage) ?? []);
  const calibrationProgress = [...events].reverse().find((event) =>
    event.type === "calibration_round_started" || event.type === "calibration_round_completed",
  );
  const calibrationDetail = job?.status === "running" && job.stage === "grpo" && calibrationProgress?.round && calibrationProgress.total_rounds
    ? calibrationProgress.type === "calibration_round_completed"
      ? `Calibration round ${calibrationProgress.round} of ${calibrationProgress.total_rounds} verified.`
      : `Calibration round ${calibrationProgress.round} of ${calibrationProgress.total_rounds} is sampling the frozen starting policy.`
    : null;
  const statusTone = job?.status === "completed"
    ? "good"
    : job?.status === "failed" || job?.status === "interrupted"
      ? "danger"
      : trainingActive ? "info" : "muted";
  const outputDirectory = job?.result?.stages.map((stage) => stage.output_dir).filter((value): value is string => Boolean(value)).at(-1);
  const selectedRecipe = preparation ? recipeCopy[preparation.recipe] : null;

  return (
    <section className="training-workspace" aria-labelledby="training-monitor-heading">
      {error && <div className="source-error training-page-error" role="alert">{error}</div>}

      <article className="panel training-command-panel" data-tour="train">
        <header className="panel-header">
          <div><p className="panel-kicker">One local GPU</p><h2 id="training-monitor-heading">Start training</h2></div>
          <span className={`status-chip ${statusTone}`}>{job?.status ?? "connecting"}</span>
        </header>
        <div className="training-command-copy">
          <div><strong>Actual GPU training happens here.</strong><p>Start training runs preflight, then changes the model with QLoRA SFT, GRPO, or both—based only on what you configured in Data.</p></div>
          <div className="training-command-actions">
            <button className="secondary-button" type="button" onClick={() => void check()} disabled={trainingActive || action !== null}>{action === "check" ? "Checking..." : "Check readiness"}</button>
            <button className="primary-button" type="button" onClick={() => void start()} disabled={trainingActive || action !== null}>{action === "start" ? "Starting..." : job?.status === "failed" || job?.status === "interrupted" ? "Retry training" : job?.status === "completed" ? "Train again" : "Start training"}</button>
          </div>
        </div>
        {preparation && (
          <div className={`readiness-result ${preparation.status}`} role="status">
            <div><strong>{preparation.status === "ready" ? "Ready on this machine" : "Action needed"}</strong><p>{preparation.summary}</p></div>
            <div><span>Training selected</span><strong>{selectedRecipe?.label}</strong><p>{selectedRecipe?.detail}</p></div>
          </div>
        )}
        {preparation?.recipe === "needs_training_data" && (
          <div className="training-data-guide" role="note">
            <div className="training-data-guide-heading">
              <div><span>Do this next</span><strong>{preparation.next_action?.title ?? "Choose how this model will learn"}</strong></div>
              <button className="secondary-button" type="button" onClick={onOpenData}>Open Data</button>
            </div>
            <div className="training-data-options">
              <section><strong>Accepted examples → QLoRA SFT</strong><p>Add instruction-and-accepted-response JSONL, or choose Accepted changes on a repository and approve useful commits.</p></section>
              <section><strong>Executable tasks → GRPO</strong><p>Add resettable code tasks with an instruction and an executable verifier that scores the result.</p></section>
            </div>
            <p className="training-data-sequence"><strong>Add both:</strong> AutoTrainer runs SFT first, then GRPO continues training the same adapter.</p>
          </div>
        )}
        {preparation?.recipe !== "needs_training_data" && preparation?.next_action && (
          <div className="training-next-action" role="note"><span>Do this next</span><strong>{preparation.next_action.title}</strong><p>{preparation.next_action.detail}</p></div>
        )}
        {job && job.status !== "idle" && <div className="run-message training-run-message" role={job.status === "failed" || job.status === "interrupted" ? "alert" : "status"}><span className={`health-dot ${statusTone}`} aria-hidden="true" /><div><strong>{stageNames[job.stage || ""] || "Training run"}</strong><p>{calibrationDetail ?? job.message}</p></div></div>}
      </article>

      <GrpoEvidencePanel context="training" refreshKey={`${revision}-${job?.id ?? "idle"}`} live={trainingActive} teachingLoss={sftLoss} />

      <article className="panel training-output-panel">
        <header className="panel-header"><div><p className="panel-kicker">Post-training artifact</p><h2>Adapter output</h2></div>{job?.status === "completed" && <span className="status-chip good">ready to evaluate</span>}</header>
        {!job?.result ? <div className="evidence-empty"><strong>No adapter output yet</strong><p>Completed stage paths and metrics appear here only after the trainer writes them.</p></div> : (
          <div className="training-output-grid">
            {job.result.stages.map((stage) => (
              <section key={stage.stage}><div><strong>{stageNames[stage.stage]}</strong><span className={`status-chip ${completedStages.has(stage.stage) ? "good" : "muted"}`}>complete</span></div>{stage.output_dir && <code>{stage.output_dir}</code>}{stage.trainable_adapter_parameters !== undefined && <p>{stage.trainable_adapter_parameters.toLocaleString()} trainable adapter parameters</p>}</section>
            ))}
          </div>
        )}
        {outputDirectory && <div className="next-step-note"><strong>Next: prove it.</strong><p>Run the frozen held-out evaluation before treating this adapter as verified.</p></div>}
        {job?.status === "idle" && <button className="text-button" type="button" onClick={onOpenData}>Review Data</button>}
      </article>
    </section>
  );
}
