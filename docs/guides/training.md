# Training

AutoTrainer's V1 training surface is intentionally narrow, but its learning
path is conditional:

```text
supported 9B text causal LM + 4-bit QLoRA
        |
        +-- accepted examples only --> SFT
        +-- executable tasks only --> GRPO practice
        +-- both signals -----------> SFT, then GRPO on the same adapter
        |
declared 9B reference vs trained candidate comparison
```

The frozen base weights are not rewritten. QLoRA creates or updates a PEFT
adapter in every path. SFT teaches from accepted responses. GRPO practices
against executable rewards; after SFT it continues the same adapter, while a
practice-only project starts a fresh QLoRA adapter from the selected base.

## Before using the GPU

Run the shared preparation stage first:

```bash
autotrainer prepare --config autotrainer.yaml
autotrainer train auto --config autotrainer.yaml
```

`prepare` performs validation, source scanning, compilation, recipe resolution,
and local runtime checks without loading model weights. `train auto` repeats
preparation before starting, so the GUI and CLI cannot launch from a stale Ready
result. Evaluation-only blockers do not prevent a useful training run. Resolve
them before `evaluate plan --write`.

At minimum, the preparation report should show:

- The exact model ID and immutable revision.
- A CUDA-capable GPU and compatible package matrix.
- At least one learning signal: a valid text-only SFT dataset, valid training
  task manifests, or both.
- For a practice path, a Docker or Podman runtime capable of network isolation.
- Resolvable repository locks and working directories.
- For a practice path, verifier bundles outside editable workspaces.

Final evaluation additionally requires a held-out task set at
`evaluation.dataset`, distinct from `grpo.eval_dataset`, and no detected
train/evaluation group collision.

The included example has a small evaluation fixture in a separate directory for exercising the contract. Its starting project is still part of the same AutoTrainer Git repository as the training fixture, so the planner correctly rejects it as a repository holdout. Add multiple genuinely independent held-out repository families and pin the real evaluation runners before claiming an improvement.

## Compiled trainer inputs

Authored inputs are declared in `sources`; trainers read compiled files:

```yaml
sft:
  dataset: ./.autotrainer/compiled/sft/train.jsonl

grpo:
  dataset: ./.autotrainer/compiled/rl/train.jsonl
  # Optional training-loop validation. Keep it separate from the final proof set.
  eval_dataset: ./data/rl-validation.jsonl

evaluation:
  dataset: ./.autotrainer/compiled/rl/evaluation.jsonl
```

Compilation is the boundary where source paths and mutable revisions become locked provenance. It also prevents a trainer from silently sweeping every file in a repository into a dataset. `evaluation.dataset` contains held-out task-pack records for the final two-suite proof. `grpo.eval_dataset` is optional trainer feedback and must be a different file; do not let final benchmark tasks affect optimization or checkpoint selection.

Inspect the JSONL before training. Each SFT line must contain text-only conversational `messages`, or conversational `prompt` and `completion`. Each GRPO or evaluation-task line contains a conversational `prompt`, `task_id`, the validated manifest, resolved source path/revision, and sandbox settings.

## QLoRA supervised tuning

Run:

```bash
autotrainer train sft --config autotrainer.yaml
```

The base-model loader owns 4-bit quantization:

```yaml
model:
  quantization:
    method: bitsandbytes-4bit
    quant_type: nf4
    double_quant: true
    compute_dtype: bfloat16
```

The trainable adapter configuration is:

```yaml
qlora:
  rank: 32
  alpha: 32
  dropout: 0.0
  bias: none
  target_modules: all-linear
```

The supervised stage uses assistant-only, completion-only loss. The prompt and tool observations provide context; accepted assistant behavior is the target. The example uses batch size one, gradient accumulation, gradient checkpointing, BF16, and a 2,048-token maximum as conservative starting values for a 9B model.

SFT is a warm start, not the final proof. Its purpose is to make useful behavior frequent enough that multiple GRPO rollouts receive different, informative rewards. If every rollout receives zero, increasing RL steps will not manufacture a learning signal.

### SFT output

The selected SFT checkpoint must contain standard PEFT artifacts, including `adapter_config.json` and adapter weights. Update `grpo.sft_adapter` to that directory:

```yaml
grpo:
  sft_adapter: ./.autotrainer/runs/sft
```

The reference runner saves the final SFT adapter at `sft.output_dir`. Retain periodic checkpoints, select the adapter under test explicitly in `evaluation.arms`, and record why it was chosen; V1 evaluation does not search training checkpoints automatically.

## GRPO reinforcement learning

Validate the adapter and environment again, then run:

```bash
autotrainer validate --config autotrainer.yaml
autotrainer doctor --config autotrainer.yaml
autotrainer train rl --config autotrainer.yaml
```

The command is named `train rl`; its algorithm configuration lives under `grpo`.

For each dataset row, the frontend environment:

1. Resolves the task manifest and its locked repository source.
2. Materializes a disposable checkout.
3. Uses the task’s `workingDirectory` inside that checkout.
4. Mounts the verifier bundle outside the editable repository.
5. Starts the container with no external network.
6. Exposes bounded `list_files`, `read_file`, `search_code`, `apply_patch`, and `run_check` tools.
7. Enforces model token, tool-call, command, and episode limits.
8. Runs trusted verification and reads the structured report.
9. Persists raw reward signals and destroys the workspace.

The policy can run only named checks. It does not receive the hidden verifier or an unrestricted host terminal.

### Reward gates and signals

Build and regression safety are hard gates. A rollout receives total reward zero when the build fails or the regression pass rate is below one.

For a passing rollout, the initial weights are:

| Signal | Weight |
|---|---:|
| Hidden task tests | 35% |
| Regression safety | 20% |
| Responsive rules | 20% |
| Design-system rules | 15% |
| Accessibility and patch quality | 10% |

The task manifest calls the final signal `patchQuality`; a verifier may combine auditable accessibility and focused-patch checks within it. Store its components separately when possible.

The verifier report contains:

```json
{
  "build_passed": true,
  "regression_pass_rate": 1.0,
  "task_pass_rate": 0.75,
  "responsive_pass_rate": 1.0,
  "design_rule_pass_rate": 0.8,
  "code_quality_pass_rate": 0.9
}
```

Never reward only shorter patches, fewer tokens, or static-analysis scores. Those signals are easy to exploit without completing the frontend task.

### Sequential single-GPU execution

The initial GRPO configuration uses two generations, disables vLLM, and runs one environment at a time. The intended schedule is sequential:

```text
load policy and generate grouped rollouts
        ↓
release rollout-only memory
        ↓
execute builds and verifiers on CPU/container runtime
        ↓
load/update the trainable adapter
        ↓
save state and repeat
```

“Single GPU” means one GPU can complete the job. It does not promise that policy serving, a reference model, a trainer, a vision judge, and multiple sandboxes remain resident concurrently.

If the job runs out of memory, first reduce `sft.max_length`, `grpo.max_completion_length`, `grpo.num_generations`, or the generation batch. Do not increase gradient accumulation expecting it to reduce the memory needed for one forward pass. Record every change in the resolved recipe so checkpoint comparisons remain meaningful.

## Reproducibility and recovery

Every run should retain:

- The original and resolved configuration.
- Exact base-model revision and tokenizer identity.
- Python package and CUDA versions.
- GPU name and available VRAM.
- Source locks and compiled dataset fingerprints.
- Environment/container identity.
- Random seeds and generation settings.
- Periodic adapter checkpoints.
- Rollout prompts, tool trajectories, patches, and raw verifier reports where licensing permits.
- Failure and exclusion reasons.

Resume only when all locked inputs match. If the model revision, source revision, compiler version, reward weights, task limits, or adapter architecture changed, start a new run identity rather than appending incomparable steps.

## Evaluation protocol

A successful optimizer run is not the product success criterion. The required model benchmark compares:

- The immutable 9B reference declared for the benchmark.
- The project model with the adapter produced by the prepared teach, practice,
  or combined QLoRA path.

Use the same held-out tasks, model prompt template, tools, starting revisions, completion limit, tool-call limit, generation settings, and verifier. The primary metric is verified task success; also retain build rate, task-test pass rate, regressions, accessibility, responsive checks, tokens per success, and wall time per success.

The second proof runs:

- Fable orchestrator with the base 9B.
- The identical Fable orchestrator with that same trained candidate adapter.

Both receive identical website briefs, context, tools, time limits, and completion limits. Blind reviewers compare whether the finished sites satisfy the brief and which is better. This final rendering/review step does not make the training pipeline multimodal.

The CLI implements immutable evaluation planning, model-benchmark execution, external Fable request/result exchange, local result verification, blind-review import/export, reporting, and winner-gated packaging. `benchmark` is an alias for `evaluate` and accepts the same subcommands. Implementation alone is not evidence: the configured runners must be pinned, every planned trial must be completed, the model benchmark must satisfy its unique-task and confidence rules, the Fable review must satisfy its unique-task and completeness rules, and both decisions must report `verified_better`. The bundled placeholders and fixtures do not meet that bar. See the [V1 handoff plan](../V1-HANDOFF.md) for the remaining run work.

## Current runtime matrix

The 2026-07-14 reference combination is Python 3.11 with PyTorch 2.13.0, Transformers 5.13.1, TRL 1.8.0, PEFT 0.19.1, Accelerate 1.14.0, Datasets 5.0.0, bitsandbytes 0.49.2, and jmespath 1.1.0. See [Getting started](getting-started.md) for authoritative package links and installation notes.

This matrix is recorded because agentic GRPO depends on evolving library interfaces. Upgrade it as a tested set, not one dependency at a time in an unrecorded environment.

## Out of scope for V1

- Multimodal or screenshot-conditioned model training.
- Cloud, distributed, or multi-GPU training.
- A GPU vision judge.
- Arbitrary framework and arbitrary-model compatibility.
- Full-weight fine-tuning or merged-base redistribution.
- Unreviewed automatic task generation from arbitrary Git history.
- Claims of improvement without held-out model and Fable comparisons.
