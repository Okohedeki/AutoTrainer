import { useEffect, useState } from "react";
import {
  ApiClientError,
  applyRuntimeSetup,
  getRuntimeSetup,
  type RuntimeSetupAction,
  type RuntimeSetupWorkspace,
} from "./api";

const liveStatuses = new Set(["queued", "running"]);

function componentStatus(workspace: RuntimeSetupWorkspace) {
  const packagesReady = workspace.doctor.packages.every((item) => item.status === "ready");
  return [
    {
      label: "Python 3.11",
      ready: workspace.doctor.python.status === "ready",
      detail: `${workspace.doctor.python.version} · expected ${workspace.doctor.python.expected}`,
    },
    {
      label: "Pinned ML packages",
      ready: packagesReady,
      detail: packagesReady
        ? `${workspace.doctor.packages.length} exact packages import successfully`
        : `${workspace.doctor.packages.filter((item) => item.status !== "ready").length} missing, mismatched, or unusable`,
    },
    {
      label: "One CUDA GPU",
      ready: workspace.doctor.gpu.status === "ready",
      detail: workspace.doctor.gpu.device_name || workspace.doctor.gpu.detail || "CUDA is not ready",
    },
    {
      label: "Container backend",
      ready: workspace.doctor.sandbox.status === "ready",
      detail: workspace.doctor.sandbox.detail || "Docker or Podman is not ready",
    },
    {
      label: "Pinned rollout image",
      ready: workspace.doctor.environment_image.status === "ready",
      detail: workspace.doctor.environment_image.detail || "The configured image has not been built",
    },
  ];
}

export default function RuntimeSetupPanel({ disabled = false }: { disabled?: boolean }) {
  const [workspace, setWorkspace] = useState<RuntimeSetupWorkspace | null>(null);
  const [busy, setBusy] = useState<RuntimeSetupAction["id"] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = async (signal?: AbortSignal) => {
    const next = await getRuntimeSetup(signal);
    setWorkspace(next);
    setError(null);
    return next;
  };

  useEffect(() => {
    const controller = new AbortController();
    refresh(controller.signal).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Runtime setup could not be inspected.");
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!workspace || !liveStatuses.has(workspace.job.status)) return;
    let stopped = false;
    const controller = new AbortController();
    let timer = 0;
    const poll = async () => {
      try {
        const next = await refresh(controller.signal);
        if (stopped) return;
        if (liveStatuses.has(next.job.status)) {
          timer = window.setTimeout(() => void poll(), 2_000);
        } else {
          setBusy(null);
        }
      } catch (reason) {
        if (!stopped && !controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Runtime setup status could not be refreshed.");
      }
    };
    timer = window.setTimeout(() => void poll(), 2_000);
    return () => {
      stopped = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [workspace?.job.status, workspace?.job.id]);

  const apply = async (action: RuntimeSetupAction) => {
    setBusy(action.id);
    setError(null);
    try {
      const job = await applyRuntimeSetup(action.id);
      setWorkspace((current) => current ? { ...current, job } : current);
    } catch (reason) {
      setBusy(null);
      setError(reason instanceof ApiClientError ? reason.message : "Runtime setup could not start.");
    }
  };

  const components = workspace ? componentStatus(workspace) : [];
  const jobLive = Boolean(workspace && liveStatuses.has(workspace.job.status));

  return (
    <section className="panel runtime-setup-panel" aria-labelledby="runtime-setup-heading">
      <header className="panel-header">
        <div>
          <p className="eyebrow">This machine</p>
          <h2 id="runtime-setup-heading">Training runtime</h2>
          <p>Doctor checks the exact stack. AutoTrainer can apply each supported setup action and then check again.</p>
        </div>
        <button className="secondary-button" type="button" onClick={() => void refresh().catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "Runtime setup could not be inspected."))} disabled={jobLive || disabled}>Check again</button>
      </header>

      {error && <div className="source-error" role="alert">{error}</div>}

      {workspace && (
        <>
          <ul className="runtime-component-list" aria-label="Runtime components">
            {components.map((component) => (
              <li key={component.label}>
                <span className={`runtime-component-mark ${component.ready ? "ready" : "blocked"}`} aria-hidden="true">{component.ready ? "OK" : "!"}</span>
                <div><strong>{component.label}</strong><small>{component.detail}</small></div>
                <span className={`status-chip ${component.ready ? "good" : "warning"}`}>{component.ready ? "Ready" : "Action needed"}</span>
              </li>
            ))}
          </ul>

          {workspace.job.status !== "idle" && (
            <div className={`runtime-setup-job ${workspace.job.status}`} role={workspace.job.status === "failed" ? "alert" : "status"}>
              <strong>{workspace.job.status === "completed" ? "Setup action completed" : workspace.job.status === "failed" ? "Setup action stopped" : "Setup in progress"}</strong>
              <p>{workspace.job.message}</p>
              {workspace.job.result?.restart_required && <small>Restart Windows or the AutoTrainer backend before relying on the new runtime.</small>}
            </div>
          )}

          {workspace.actions.length > 0 ? (
            <div className="runtime-actions">
              {workspace.actions.map((action) => (
                <article key={action.id}>
                  <div><strong>{action.title}</strong><p>{action.detail}</p><code>{action.command.join(" ")}</code>{action.requires_admin && <small>Administrator approval is required.</small>}</div>
                  <button className={action.status === "available" ? "primary-button" : "secondary-button"} type="button" onClick={() => void apply(action)} disabled={action.status !== "available" || jobLive || busy !== null || disabled}>{busy === action.id ? "Starting..." : action.status === "available" ? "Apply" : "Waiting"}</button>
                </article>
              ))}
            </div>
          ) : (
            <div className="runtime-ready"><span aria-hidden="true">OK</span><div><strong>Runtime ready</strong><p>The pinned packages, CUDA device, container backend, and rollout image passed Doctor.</p></div></div>
          )}
        </>
      )}
    </section>
  );
}
