# AutoTrainer architecture

AutoTrainer is one local system with two clients:

- the GUI is the primary interface for people;
- the CLI is the scriptable interface for agents.

They call the same Python services and persist the same project files. The GUI does not maintain a second training workflow.

```text
                         autotrainer.yaml
                               |
               +---------------+---------------+
               |                               |
        GUI -> local /api/v1                 CLI commands
               |                               |
               +---------- shared services ----+
                               |
       projects / model cache / sources / compilation
                               |
            training / evaluation / local model host
                               |
                 durable artifacts and evidence
```

## Product lifecycle

The console is organized around the work, not internal commands:

1. **Projects** - create or select one local specialist workspace.
2. **Data** - select/download models and declare learning or held-out sources.
3. **Train** - prepare, select SFT/GRPO stages, execute, and graph observed metrics.
4. **Evaluate** - freeze the benchmark and show trusted generation/verifier results.
5. **Serve** - load the downloaded base or a completed adapter behind a local endpoint.

`autotrainer prepare` is an internal boundary inside Train, not a separate product destination. It validates, scans, compiles, plans, and runs machine checks before the GPU job begins.

## Project boundary

`autotrainer.yaml` declares model identity, source intent, QLoRA/SFT/GRPO settings, sandbox policy, evaluation arms, and package settings. Relative paths resolve from that file.

The dashboard starts with one explicit config and can manage additional projects under its trusted `.autotrainer/projects` root. Project names are converted to one safe directory segment. Creation writes and validates the new config before atomically making it active; a failed creation is not exposed as selectable state. New managed projects share the workspace model cache but keep their configs, runs, evidence, and host receipts separate.

Agents can address any project directly with `--config`. The `autotrainer projects` commands expose the same bounded workspace operations when an explicit projects root is supplied.

## Data boundary

Source kinds remain distinct:

- `repository` - Git code, accepted-history candidates, or task starting states;
- `sft_jsonl` - explicit text demonstrations;
- `task_pack` - resettable tasks with bounded tools and hidden verification.

Human-facing repository modes map to the canonical YAML roles:

| Mode | YAML role | Meaning |
|---|---|---|
| `accepted_changes` | `history` | review Git changes and instructions for SFT |
| `practice_tasks` | `rl_seed` | repository can supply task starting states |
| `reference_only` | `style` | inspect code without turning it into training rows |
| `evaluation_holdout` | `evaluation` | isolate the repository from optimization |

Adding a GitHub source clones it into project-managed storage and pins a detached commit before the declaration is saved. Local sources remain in place. Scanning is read-only; compilation produces explicit JSONL and provenance below the project artifact directory.

## Model boundary

The GUI and `autotrainer models search` can query Hugging Face, but search results carry an explicit compatibility label. Before any remote search or download, both clients can use the same bounded local discovery service. It inspects only the configured cache and known Hugging Face cache roots, returns opaque candidates for structurally complete supported snapshots, and revalidates a candidate before recording its exact revision and server-owned cache root. It never recursively searches the machine or accepts a browser-provided path.

The guarded V1 training implementation supports the pinned text-only `Qwen/Qwen3.5-9B` profile. Selecting, adopting an existing snapshot, and downloading are separate durable operations; real training and hosting load only the recorded local snapshot.

The benchmark reference is code-owned product configuration:

- model: `empero-ai/Qwythos-9B-Claude-Mythos-5-1M`;
- revision: `14a29bae5143091aeaf87ad37120de4cd57d592c`;
- role: evaluation reference only.

It has its own download receipt and cannot become the project training base.

## Training boundary

QLoRA keeps the base weights frozen and trains a PEFT adapter. Compiled evidence selects one path:

- demonstrations only -> SFT;
- executable tasks only -> GRPO from the base or an explicitly compatible adapter;
- both -> SFT, then GRPO continues that exact adapter.

GRPO tasks execute in disposable, network-disabled Docker or Podman environments. The policy receives bounded file/patch/check tools, never a host shell or the hidden verifier. Reward components and hard-gate failures are stored separately.

Training records are durable. The UI polls observed trainer events and renders their real steps and metrics; it does not interpolate an optimizer curve or invent progress.

## Evaluation boundary

`evaluate plan --write` freezes the held-out task matrix, model revisions, adapter digest, environment, runner identity, seeds, and fairness policy. The plan and trial matrix are cryptographically checked when loaded.

The `model_benchmark` suite uses AutoTrainer's built-in text-only producer. It:

1. loads the pinned Qwythos reference and candidate adapter one arm at a time;
2. uses bounded source context and generation limits;
3. stores each producer result and evidence durably;
4. applies the patch and runs the trusted local verifier;
5. reports real rubric components and failures to the UI.

Arms are grouped so only one 9B model occupies the GPU. Analysis positions remain deterministically counterbalanced; V1 does not claim per-trial GPU arm randomization.

The `fable_ab` suite is external. AutoTrainer can hash and pin a supplied Fable
runtime bundle, freeze/export its requests, background-ingest envelopes,
locally verify patches, manage blind-review rows, and report a separate
decision. It does not include or emulate a Fable runtime. Placeholder Fable
identity is marked deferred and does not block the built-in local model
benchmark.

## GPU coordination

Training, built-in evaluation, and local hosting share one exclusive GPU-0 lease. The lease is process-aware and cross-project, preventing two dashboard projects or a GUI/CLI pair from loading separate 9B models concurrently. A busy operation fails clearly instead of risking an out-of-memory race.

The model host holds both its project lease and GPU lease for its lifetime. Shutdown waits for the current generation, releases model references, clears the CUDA allocator when possible, and only then releases the lease.

## Two loopback servers

- `autotrainer serve` runs the dashboard control API at `http://127.0.0.1:8765/api/v1` by default. It performs project operations and spawns jobs; it does not load weights.
- `autotrainer host start` launches a separate model process at `http://127.0.0.1:8791` by default. It exposes `/health`, `/v1/models`, and `/v1/chat/completions`.

The V1 model endpoint is OpenAI-compatible only for the implemented subset: text messages, one serialized bounded request, no streaming, and loopback access. It is a local use surface, not public deployment.

## Proof status

The services, contracts, UI, and local job/evidence paths are implemented. This repository does not bundle weights and has not yet completed a real 9B training run, a production held-out benchmark, or the external Fable comparison. Those results remain the release proof, not assumptions encoded in the architecture.
