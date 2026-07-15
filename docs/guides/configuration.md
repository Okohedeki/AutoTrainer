# Configuration reference

`autotrainer.yaml` is the source of truth for an AutoTrainer experiment. Keep it in the experiment directory, review it like code, and preserve the resolved lock and recipe with published adapters.

The JSON Schema is [`schemas/autotrainer.schema.json`](../../schemas/autotrainer.schema.json). The complete example is [`examples/frontend-expert/autotrainer.yaml`](../../examples/frontend-expert/autotrainer.yaml).

```bash
autotrainer validate --config autotrainer.yaml
```

Schema validation checks shape and value types. `validate`, `source scan`, `compile`, `plan`, and `doctor` also apply semantic checks that JSON Schema cannot express, such as source-role compatibility, path resolution, immutable revisions, train/evaluation separation, task-source links, adapter existence, and machine readiness.

## Path and revision rules

- Relative paths resolve from the directory containing `autotrainer.yaml`.
- Absolute Linux paths are accepted. Under WSL2, use `/mnt/c/...` or `/mnt/h/...`, not Windows backslash paths.
- Repository URLs may be HTTPS or SSH Git URLs.
- A local repository may use `startingRevision: locked` in an authoring task. `compile` resolves it to the exact content or Git revision in the generated lock.
- An upstream model or Git repository revision should be an immutable commit SHA. A branch such as `main` may be accepted while authoring; `plan` warns and `autotrainer lock` resolves the model revision.
- Secrets, access tokens, and Hugging Face credentials do not belong in YAML. Supply them through the relevant credential helper or environment.

## Top-level structure

Every schema version `1` file contains these sections:

```yaml
schema_version: 1
project: {}
model: {}
sources: []
qlora: {}
sft: {}
grpo: {}
environment: {}
evaluation: {}
package: {}
```

Unknown fields are rejected. This prevents a misspelled option from being silently ignored.

## `project`

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Stable lowercase experiment/package slug. |
| `seed` | non-negative integer | Project-level deterministic seed. Stage-specific seeds remain explicit. |
| `artifact_dir` | path | Root for locks, compiled data, runs, reports, and packages. |

Example:

```yaml
project:
  name: polished-frontend-9b
  seed: 42
  artifact_dir: ./.autotrainer
```

Generated data should not be committed by default. The configuration, source/license manifest, resolved recipe, and final evaluation report should be retained with a published expert.

## `model`

The base model is selected here—not in a TypeScript dropdown or hidden Python constant.

```yaml
model:
  provider: huggingface
  id: Qwen/Qwen3.5-9B
  revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a
  loader: qwen3_5_text
  trust_remote_code: false
  dtype: bfloat16
  max_sequence_length: 2048
  quantization:
    method: bitsandbytes-4bit
    quant_type: nf4
    double_quant: true
    compute_dtype: bfloat16
```

| Field | Required value or meaning |
|---|---|
| `provider` | `huggingface` in V1. |
| `id` | Hugging Face model ID. The example uses `Qwen/Qwen3.5-9B`. |
| `revision` | Exact model revision. The bundled example is pinned; resolve intentional updates with `autotrainer lock`. |
| `loader` | `qwen3_5_text` selects the supported text-only causal-LM loader. |
| `trust_remote_code` | Must be `false` in the supported V1 security boundary. |
| `dtype` | `bfloat16` for compute on the target GPU. |
| `max_sequence_length` | Maximum model context used by the initial runtime. |
| `quantization` | Frozen-base 4-bit bitsandbytes/NF4 loader settings. |

The model ID is replaceable; compatibility is not automatic. A different model needs a compatible Transformers architecture, causal-LM head, tokenizer/chat template, PEFT target modules, redistribution license, and a measured single-GPU memory profile.

`autotrainer models list` reports catalog entries. `autotrainer model use ID --revision SHA` writes the project selection; direct YAML editing remains supported.

## `sources`

`sources` is a flat list. Each item declares one repository, supervised dataset, or executable task pack.

Common fields:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Unique source ID used by reports and task manifests. |
| `kind` | enum | `repository`, `sft_jsonl`, or `task_pack`. |
| `uri` | string | Relative/absolute path, or a Git URL for repositories. |
| `partition` | enum | `train` or `evaluation`. |
| `roles` | list | One or more declared uses. |
| `revision` | string, optional | Repository revision to lock. |
| `include` | glob list, optional | Files eligible for scanning. |
| `exclude` | glob list, optional | Files excluded after include matching. |
| `runtime` | object, optional | Frontend setup and named checks for a repository. |
| `license` | object, optional | SPDX expression and attribution record. |

Supported roles are:

- `style`: final code may provide conventions or corpus evidence.
- `history`: accepted changes may be candidates for supervised compilation.
- `rl_seed`: the repository can seed mutation or reconstructed tasks.
- `demonstrations`: explicit SFT records.
- `rl_tasks`: executable training tasks.
- `evaluation`: held-out repositories or task packs.

The semantic validator checks sensible combinations. For example, `demonstrations` normally belongs to `sft_jsonl`, while `rl_tasks` normally belongs to `task_pack`.

Repository runtime fields are `preset`, `working_directory`, `install`, `build`, `test`, and `browser_test`. They declare commands; scanning must not execute them. Execution happens later in a sandbox.

See [Data sources](data-sources.md) for examples and compilation behavior.

## `qlora`

The model section quantizes the frozen base to 4-bit NF4. `qlora` declares only the trainable adapter:

```yaml
qlora:
  rank: 32
  alpha: 32
  dropout: 0.0
  bias: none
  target_modules: all-linear
```

`target_modules: all-linear` delegates architecture-specific linear-module discovery to PEFT. The resolved module list belongs in the run recipe. Changing adapter rank, target modules, or base revision creates a different experiment and invalidates checkpoint comparisons unless all candidates are rerun.

## `sft`

```yaml
sft:
  enabled: true
  dataset: ./.autotrainer/compiled/sft/train.jsonl
  output_dir: ./.autotrainer/runs/sft
  max_length: 2048
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 0.0001
  num_train_epochs: 1
  bf16: true
  tf32: true
  gradient_checkpointing: true
  completion_only_loss: true
  assistant_only_loss: true
  packing: false
  seed: 42
  logging_steps: 5
  save_steps: 50
```

`dataset` is required for execution and should point to the canonical file produced by `compile`. Direct demonstration files remain declared under `sources`. `eval_dataset` is optional but strongly recommended once a separate supervised holdout exists. Records must be text-only conversational examples using `messages`, or `prompt` plus `completion`. Both completion-only and assistant-only loss are enabled so user instructions are not trained as target tokens.

The initial values are conservative starting points, not universal optima. Increase length or batch-related values only after measuring peak memory on the selected model.

## `grpo`

```yaml
grpo:
  enabled: true
  algorithm: grpo
  dataset: ./.autotrainer/compiled/rl/train.jsonl
  sft_adapter: ./.autotrainer/runs/sft
  output_dir: ./.autotrainer/runs/grpo
  max_steps: 500
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 2
  num_generations: 2
  generation_batch_size: 2
  max_completion_length: 2048
  max_tool_calling_iterations: 8
  learning_rate: 0.00001
  beta: 0.0
  loss_type: dapo
  bf16: true
  tf32: true
  gradient_checkpointing: true
  use_vllm: false
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  seed: 42
```

`sft_adapter` must resolve to a completed PEFT adapter containing adapter configuration and weights. `train rl` loads it as trainable and continues updating it; starting GRPO from an uninitialized adapter is rejected.

`num_generations` controls the candidates compared for each prompt. The V1 defaults use two and disable vLLM to keep the policy trainer and rollout path within the single-GPU boundary. `loss_type: dapo` is the selected loss variant. `beta: 0.0` disables the explicit reference-model KL term in the initial memory-conscious recipe; that is a deliberate experiment choice and must be recorded.

Every compiled GRPO JSONL record has a conversational `prompt`, the validated manifest, resolved source path/revision, and sandbox settings passed to `environment.reset(**row)`. Image fields are rejected.

## `environment`

```yaml
environment:
  factory: autotrainer.environments.frontend:FrontendEnvironment
  backend: docker
  image: autotrainer/frontend-runtime:0.1
  network: none
  max_tool_output_chars: 20000
  episode_timeout_seconds: 900
```

`factory` selects the built-in frontend environment. `backend` may be Docker or Podman; the pinned image contains the frontend dependencies needed while `network: none` is enforced. `max_tool_output_chars` bounds observations returned to the model, and `episode_timeout_seconds` is the outer wall-clock limit.

Only named checks declared by the task are available through `run_check`. The policy does not receive an arbitrary host shell. The current runner executes one environment at a time as part of the single-machine V1 contract.

## `evaluation`

```yaml
evaluation:
  task_pack: ./tasks/evaluation
  candidates: [base, sft, grpo]
  holdout_unit: repository
  primary_metric: verified_task_success
  metrics:
    - verified_task_success
    - build_rate
    - task_pass_rate
    - regression_rate
    - accessibility
    - responsive
    - tokens_per_success
    - wall_time_per_success
  fable_a_b:
    enabled: false
    blind_review: true
```

Evaluation must hold out entire repository/project families. Random commit splitting can place near-identical components on both sides and overstate improvement.

The model benchmark must keep prompts, tools, task starting states, token/tool limits, generation settings, and verification identical for Base, SFT, and GRPO candidates. The Fable A/B is a second test with identical orchestration around the base and winning model.

The evaluation configuration is versioned now, but automatic baseline/evaluation execution is the next CLI milestone after `0.1.0` training.

## `package`

```yaml
package:
  name: polished-frontend-9b
  type: lora_adapter
  winner: best
  output_dir: ./.autotrainer/packages/polished-frontend-9b
  include:
    - adapter
    - system_prompt
    - tool_schema
    - evaluation_report
    - source_license_manifest
    - resolved_recipe
```

The default deliverable is an adapter package, not a copy of the base weights. `winner: best` means the future evaluation step selects between eligible SFT and GRPO checkpoints by the declared primary metric. A package must identify the exact base model revision; adapter weights without that identity are incomplete.

Package export is a declared next milestone, not evidence that the current CLI can already publish or serve an expert.
