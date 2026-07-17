import { useState } from "react";
import {
  ApiClientError,
  createProject,
  getProjects,
  selectProject,
  type ProjectsWorkspace,
} from "./api";

export default function ProjectsPanel({
  workspace,
  disabled = false,
  onWorkspaceChanged,
}: {
  workspace: ProjectsWorkspace | null;
  disabled?: boolean;
  onWorkspaceChanged: (workspace: ProjectsWorkspace) => void;
}) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    const next = await getProjects();
    onWorkspaceChanged(next);
    return next;
  };

  const create = async () => {
    const nextName = name.trim();
    if (!nextName) return;
    setBusy("create");
    setError(null);
    try {
      // Project creation is atomic on the backend: the returned project is
      // already the active working context, so a second select could only
      // leave an orphaned project if another local job starts in between.
      await createProject(nextName);
      setName("");
      await refresh();
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "The project could not be created.");
    } finally {
      setBusy(null);
    }
  };

  const choose = async (projectId: string) => {
    if (projectId === workspace?.active_id) return;
    setBusy(projectId);
    setError(null);
    try {
      await selectProject(projectId);
      await refresh();
    } catch (reason) {
      setError(reason instanceof ApiClientError ? reason.message : "The project could not be opened.");
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="projects-workspace" aria-labelledby="projects-heading">
      <article className="panel project-create-panel">
        <header className="panel-header">
          <div><p className="panel-kicker">New specialist</p><h2 id="projects-heading">Create a project</h2></div>
          <span className="status-chip muted">local</span>
        </header>
        <p className="panel-lead">Each project keeps its own model choice, data, training record, evaluation proof, and serving state.</p>
        <form
          className="project-create-form"
          onSubmit={(event) => {
            event.preventDefault();
            void create();
          }}
        >
          <label htmlFor="project-name">Project name</label>
          <div>
            <input id="project-name" value={name} onChange={(event) => setName(event.target.value)} placeholder="Frontend specialist" autoComplete="off" disabled={disabled || busy !== null} />
            <button className="primary-button" type="submit" disabled={disabled || busy !== null || !name.trim()}>{busy === "create" ? "Creating..." : "Create project"}</button>
          </div>
        </form>
        {disabled && <p className="field-note warning-text">Finish the active GPU job before switching projects.</p>}
        {error && <div className="source-error project-error" role="alert">{error}</div>}
      </article>

      <article className="panel project-list-panel">
        <header className="panel-header">
          <div><p className="panel-kicker">On this machine</p><h2>Your projects</h2></div>
          <span className="status-chip muted">{workspace?.projects.length ?? 0} total</span>
        </header>
        {!workspace ? (
          <div className="monitor-empty compact"><strong>Connecting to projects</strong><p>The local backend owns this list.</p></div>
        ) : workspace.projects.length === 0 ? (
          <div className="monitor-empty compact"><strong>No projects yet</strong><p>Create the first local specialist above.</p></div>
        ) : (
          <ul className="project-list">
            {workspace.projects.map((project) => {
              const active = project.id === workspace.active_id;
              return (
                <li key={project.id} className={active ? "active" : ""}>
                  <span className="project-avatar" aria-hidden="true">{project.name.slice(0, 2).toUpperCase()}</span>
                  <div><strong>{project.name}</strong><code>{project.config_path || project.id}</code></div>
                  <span className={`status-chip ${active ? "good" : "muted"}`}>{active ? "current" : "saved"}</span>
                  <button className="secondary-button" type="button" disabled={disabled || busy !== null || active} onClick={() => void choose(project.id)}>
                    {busy === project.id ? "Opening..." : active ? "Open" : "Open project"}
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </article>
    </section>
  );
}
