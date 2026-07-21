# Training

AutoTrainer V1 trains a QLoRA adapter around one supported 9B text model. SFT and GRPO are conditional stages, not a requirement to run two algorithms on every project.

```text
accepted examples only -> QLoRA SFT
executable tasks only  -> QLoRA GRPO
both signals           -> SFT, then GRPO continues the SFT adapter
```

The base weights remain frozen in every path. AutoTrainer verifies after PEFT setup that only LoRA adapter parameters are trainable and fails before optimization if that invariant is broken.

## Prepare before loading weights

The GUI performs preparation as part of **Start training**. Agents can inspect it separately:

```bash
autotrainer prepare --config autotrainer.yaml
autotrainer train auto --config autotrainer.yaml
```

Preparation:

- validates YAML, paths, source intent, and recipes;
- scans and locks repositories;
- compiles explicit SFT and task JSONL;
- verifies that the operator inspected and froze the current local dataset;
- verifies the recorded Hugging Face snapshot;
- selects teach, practice, or both;
- checks Python, CUDA/GPU, training packages, and the container runtime;
- does not load model weights.

`train auto` repeats preparation so a stale Ready result cannot start a changed project.

## Compiled inputs

Trainers read generated files, not arbitrary repository trees:

```yaml
sft:
  dataset: .autotrainer/compiled/sft/train.jsonl

grpo:
  dataset: .autotrainer/compiled/rl/train.jsonl
  # Optional training feedback; never reuse the final benchmark.
  # eval_dataset: ./data/rl-validation.jsonl

evaluation:
  dataset: .autotrainer/compiled/rl/evaluation.jsonl
```

Review the Data workspace, compiled rows, language counts, patches, and compile report before a costly run, then explicitly freeze the dataset. SFT rows are text conversations. GRPO rows carry the conversational prompt, task manifest, locked source identity, and sandbox settings. Final evaluation tasks stay outside optimization and checkpoint selection.

Any change to the configuration, PR catalog, LLM designs, approvals, task definitions, or compiled bytes makes the receipt stale. `prepare` and every real training entry point fail closed until the local dataset is frozen again.

## SFT

```bash
autotrainer train sft --dry-run --config autotrainer.yaml
autotrainer train sft --config autotrainer.yaml
```

The supported loader uses 4-bit NF4 base weights, BF16 compute, gradient checkpointing, and a PEFT adapter. Completion-only and assistant-only loss keep instructions and tool observations as context rather than targets.

SFT is useful when accepted examples show the behavior directly. It can also warm-start GRPO so verifier-backed rollouts receive varied, informative rewards. It is not proof that the specialist improved on held-out work.

The completed output contains `adapter_config.json` and adapter weights under `sft.output_dir`.

## GRPO

```bash
autotrainer train rl --dry-run --config autotrainer.yaml
autotrainer train rl --config autotrainer.yaml
```

The command is `train rl`; the selected V1 algorithm is configured under `grpo`.

For each row, the environment:

1. materializes a disposable checkout of the locked starting revision;
2. keeps the hidden verifier outside the editable tree;
3. starts a network-disabled Docker/Podman container;
4. exposes bounded `list_files`, `read_file`, `search_code`, `apply_patch`, and named `run_check` tools;
5. enforces generation, tool, process, output, and wall-time limits;
6. runs trusted build/regression/hidden checks;
7. records raw reward components and destroys the workspace.

The policy never receives an unrestricted host terminal or the verifier bundle.

### Reward gates

Build failure or a regression rate below one forces reward to zero. A passing rollout uses the task's declared components; the reference weights are:

| Signal | Weight |
|---|---:|
| Hidden task tests | 35% |
| Regression safety | 20% |
| Responsive rules | 20% |
| Design rules | 15% |
| Patch/accessibility quality | 10% |

Keep component values in artifacts. A single scalar without its verifier evidence is not auditable.

### Continue the SFT adapter

For a combined path, `grpo.start_from` must equal `sft.output_dir`. `train auto` passes the just-completed adapter into GRPO and writes the result to a different output directory. GRPO does not silently create a second unrelated adapter after SFT.

A practice-only project may use `start_from: base` to create a fresh QLoRA policy, or an explicitly compatible completed adapter.

## Observed telemetry

Training events are written durably and exposed through the same service contract used by the GUI and CLI:

```bash
autotrainer curriculum --config autotrainer.yaml --json
```

The curriculum view has three levels. **Overview** shows the compiled task and retained signal distribution. **Tasks** shows each instruction, locked snapshot, bounded tools, hidden-verifier contract, reward weights, and observed outcome pattern. **Rollouts** shows one verifier-scored attempt with safe tool counts, changed-file count, elapsed time, gate result, reward, and rubric components.

Rollout telemetry does not persist model reasoning, patches, tool arguments/output, verifier output, or source text. The task instruction already frozen in the compiled curriculum is the only prompt material shown. The view also does not fill gaps with simulated values. Empty measurements remain empty rather than becoming zero.

Every GRPO job records the fingerprint and SHA-256 of the compiled task dataset it actually used. Rollouts are merged into task summaries only when that binding matches the current compiled curriculum. Stored events outside that match remain visibly unmatched. Metrics describe the current job's retained event window; a truncated window is labeled as truncated rather than presented as the complete run.

Outcome labels are descriptive, not improvement claims. `unobserved` has no scored evidence, `uncalibrated` has too few retained attempts to read a pattern, `varied` has reward spread, and `flat` does not. Before optimization, AutoTrainer now runs and records at least four frozen starting-policy rollouts per task. A task must show within-group reward variation in that calibration or GRPO stops before updating the adapter. Use the retained training rollouts for later curriculum analysis, but do not confuse them with the frozen starting-policy evidence in `starting_policy_calibration.json`.

If the backend stops during a job, the durable record is marked interrupted when recovered. Full optimizer/checkpoint resume is not automatic for every combined-path interruption; retry can repeat a stage. Preserve output directories and receipts before deciding whether a retry is comparable.

## One-GPU policy

The reference recipe uses one 9B model, two GRPO generations, no vLLM, and one environment at a time. It is designed to complete on one GPU, not to keep a trainer, reference model, serving process, and multiple sandboxes resident together.

For comparable refinement experiments, AutoTrainer explicitly requests the
PyTorch SDPA attention backend and records the optimizer, scheduler, warmup,
weight decay, gradient clipping, and seed in each resolved recipe. The eager
attention backend is available as a controlled baseline. Unsupported kernels
fail closed instead of silently falling back to another implementation.

An exclusive cross-process GPU-0 lease covers:

- SFT/GRPO training;
- built-in model evaluation;
- the local callable model host.

Only one of those operations may run across all local AutoTrainer projects. This prevents the GUI and an agent command from accidentally loading competing 9B models.

The Train screen exposes the adapter-only resource policy:

```yaml
refinement:
  mode: adapter_only
  vram:
    max_gib: 20
    enforcement: hard
```

`hard` sets the CUDA per-process memory fraction before weights load and also passes a GPU `max_memory` map to the model loader. It is the default when sharing the computer matters most. `soft` records and displays the same ceiling as an operating target but does not ask CUDA to reject allocations beyond it. Both modes report allocated, reserved, and configured GiB in observed telemetry. The limit applies to the AutoTrainer process; GPU memory already consumed by unrelated applications remains outside its control.

Every completed SFT or GRPO adapter directory also contains
`training_receipt.json`. It binds the resolved recipe and exact dependency
versions to observed coarse phase wall times, trainer metrics, current and peak
VRAM, device policy, and trainable adapter parameter count. The receipt contains
no prompts, completions, tool arguments, or model reasoning. These measurements
make later same-machine comparisons possible; they are observations, not a
performance claim.

For out-of-memory failures, reduce the VRAM ceiling only if headroom exists; otherwise reduce sequence/completion length, generation count, or the generation batch first. Record every change in the resolved recipe.

## Evaluation after training

An optimizer completion is not the success criterion. Freeze the held-out plan and run the built-in benchmark:

```bash
autotrainer model reference-download --config autotrainer.yaml
autotrainer evaluate plan --write --config autotrainer.yaml
autotrainer evaluate run --suite model_benchmark --config autotrainer.yaml
```

The benchmark compares:

- pinned Qwythos 9B reference at `14a29bae5143091aeaf87ad37120de4cd57d592c`;
- the project model plus the selected completed adapter.

Both use the same held-out task snapshots, instructions, tools, limits, seeds, sampling settings, and trusted verifier. The built-in runner loads arms in a frozen grouped order so one 9B model occupies the GPU at a time. Each result, patch, verifier report, and rubric component is durable and appears in the Evaluation view only when observed.

Before planning, AutoTrainer selects the shipped evaluation profile that matches the primary frozen training language: Python, TypeScript/React, C#, or C++. It blocks a manually selected mismatch and a held-out repository set with no code in that language. The profiles use language-appropriate build/test expectations and common code-generation metrics, informed by HumanEval, MBPP, MultiPL-E, and HumanEval-X without delegating execution or trust to those projects.

Fable is an optional external compatibility add-on, not a V1 prerequisite. Running `autotrainer fable pin` explicitly adds its control arm, suite, and decision to that project; only then does Fable become part of that project's proof contract.

## Serve the result

Serving is a post-training use step, not another training stage:

```bash
autotrainer host start --adapter auto --config autotrainer.yaml
autotrainer host test "Improve this component." --config autotrainer.yaml
autotrainer host stop --config autotrainer.yaml
```

`auto` prefers a completed GRPO adapter, then SFT, then the base snapshot. The host is text-only, loopback-only, non-streaming, and serializes one bounded generation request. It implements a small `/v1/chat/completions` compatibility surface, not public deployment.

## Reproducibility

Retain:

- original and resolved config;
- exact base revision and download receipt;
- package/CUDA/GPU identity;
- source locks and compiled-data fingerprints;
- container and task/verifier identity;
- seeds and generation settings;
- trainer events, checkpoints, rollouts, patches, and raw verifier reports;
- final adapter and evidence hashes.

The Train screen plots only retained observations. Step throughput is a
wall-time window between actual trainer logs, and peak memory comes from CUDA's
allocator counters. A missing series stays visibly empty instead of being
interpolated or reconstructed.

Changing model/source revisions, compiler code, adapter architecture, reward policy, tasks, or limits creates a new experiment identity.

## V1 exclusions

- multimodal or screenshot-conditioned training;
- cloud, multi-GPU, or distributed training;
- full-weight fine-tuning or merged-base redistribution;
- arbitrary model/framework compatibility;
- unreviewed conversion of any repository into demonstrations or tasks;
- improvement claims without language-matched held-out model evidence.
