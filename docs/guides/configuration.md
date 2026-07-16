# Configuration reference

`autotrainer.yaml` is the source of truth for an AutoTrainer experiment. Keep it in the experiment directory, review it like code, and preserve the resolved lock and recipe with published adapters.

The published JSON Schema is [`schemas/autotrainer.schema.json`](../../schemas/autotrainer.schema.json). It is useful for editors and independent tooling. The complete example is [`examples/frontend-expert/autotrainer.yaml`](../../examples/frontend-expert/autotrainer.yaml).

```bash
autotrainer validate --config autotrainer.yaml
```

The CLI does not currently load that JSON Schema. `validate` and the commands that load a project run Python semantic validation instead, including required shapes plus checks the schema cannot express: source-role compatibility, path resolution, immutable revisions, train/evaluation separation, task-source links, adapter existence, and machine readiness. Run an independent JSON Schema validator when schema-level conformance itself is required.

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

The semantic validator rejects unknown top-level fields. The published JSON Schema additionally closes nested objects for editor or independent-validator checks; the CLI does not yet apply that schema automatically.

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
| `id` | Must be `Qwen/Qwen3.5-9B` for V1 SFT/GRPO execution. |
| `revision` | Exact model revision. The bundled example is pinned; resolve intentional updates with `autotrainer lock`. |
| `loader` | `qwen3_5_text` selects the supported text-only causal-LM loader. |
| `trust_remote_code` | Must be `false` in the supported V1 security boundary. |
| `dtype` | `bfloat16` for compute on the target GPU. |
| `max_sequence_length` | Maximum model context used by the initial runtime. |
| `quantization` | Frozen-base 4-bit bitsandbytes/NF4 loader settings. |

The V1 training backend rejects other model IDs and verifies the loaded `Qwen3_5ForCausalLM` class. Adding another model requires explicit implementation and testing of its architecture, tokenizer/chat template, PEFT targets, license, and single-GPU memory profile. Evaluation may still declare a different immutable 9B reference for an external pinned runner; that is an evaluation arm, not a trainable project-model option.

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
  # Optional training-loop validation; never point this at evaluation.dataset.
  # eval_dataset: ./data/rl-validation.jsonl
  start_from: ./.autotrainer/runs/sft
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

`start_from` is either `base` for a practice-only fresh QLoRA adapter, or a
completed PEFT adapter containing adapter configuration and weights. In a
combined path it must equal `sft.output_dir`. `train auto` selects the correct
value in memory from the compiled learning signal; a manual `train rl` run uses
the value declared here. The legacy `sft_adapter` key remains readable for older
projects.

`num_generations` controls the candidates compared for each prompt. The V1 defaults use two and disable vLLM to keep the policy trainer and rollout path within the single-GPU boundary. `loss_type: dapo` is the selected loss variant. `beta: 0.0` disables the explicit reference-model KL term in the initial memory-conscious recipe; that is a deliberate experiment choice and must be recorded.

Every GRPO JSONL record has a conversational `prompt`, the validated manifest, resolved source path/revision, and sandbox settings passed to `environment.reset(**row)`. Image fields are rejected. `grpo.dataset` supplies optimization episodes. `grpo.eval_dataset` is optional training-loop validation and, when declared, must be a different file from the final held-out `evaluation.dataset`; never reuse benchmark tasks for checkpoint feedback.

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
  # This names the declared evaluation task-pack source ID, not its filesystem path.
  task_pack: held-out-frontend
  dataset: ./.autotrainer/compiled/rl/evaluation.jsonl
  task_split: evaluation
  repetitions: 3
  seeds: [1701, 1702, 1703]
  holdout_unit: repository
  primary_metric: verified_task_success
  candidates: [reference_9b, base_fable, autotrainer]
  arms:
    reference_9b:
      label: Declared 9B reference
      role: reference
      parameter_class: 9b
      model:
        provider: huggingface
        id: REPLACE_WITH_REFERENCE_9B
        revision: REPLACE_WITH_40_TO_64_HEX_COMMIT
        loader: auto_text_causal_lm
        trust_remote_code: false
        dtype: bfloat16
        max_sequence_length: 2048
        quantization: project
    base_fable:
      label: Base 9B + Fable
      role: control
      parameter_class: 9b
      model: project
    autotrainer:
      label: AutoTrainer 9B
      role: candidate
      parameter_class: 9b
      model: project
      adapter:
        path: ./.autotrainer/runs/grpo
        stage: grpo
  suites:
    model_benchmark:
      kind: model_benchmark
      arms: [reference_9b, autotrainer]
      runner:
        type: command
        producer: local-model-agent
        version: REPLACE_WITH_RUNNER_VERSION
        orchestration_sha256: sha256:REPLACE_WITH_64_HEX_DIGEST
        argv: [REPLACE_WITH_MODEL_AGENT, --request, "{request}", --result, "{result}"]
    fable_ab:
      kind: fable_ab
      arms: [base_fable, autotrainer]
      runner:
        type: external
        producer: fable
        version: REPLACE_WITH_FABLE_VERSION
        orchestration_sha256: sha256:REPLACE_WITH_64_HEX_DIGEST
        result_schema: autotrainer-evaluation-result-v1
      review:
        type: manual
        blind: true
        reviewers_per_pair: 3
  fairness:
    paired_by: [task_id, repetition, seed]
    same_task_snapshot: true
    same_instruction: true
    same_tools_and_limits: true
    same_verifier: true
    same_runner_within_suite: true
    same_sampling: true
    require_seed_control: true
    immutable_models_and_adapter: true
    randomize_arm_order: true
    failures_score_zero: true
    allow_unplanned_reruns: false
  decisions:
    confidence: 0.95
    model_benchmark:
      candidate: autotrainer
      control: reference_9b
      metric: verified_task_success
      minimum_delta: 0.0
      minimum_tasks: 2
    fable_ab:
      candidate: autotrainer
      control: base_fable
      metric: blind_preference_rate
      minimum_rate: 0.5
      minimum_tasks: 2
```

Evaluation must hold out entire repository/project families. Random commit splitting can place near-identical components on both sides and overstate improvement.

`evaluation.dataset` is the canonical held-out task JSONL produced by `compile` from evaluation task-pack sources. It is consumed only by the final proof workflow. It must not be the optional `grpo.eval_dataset`, which exists for training-time validation and can influence checkpoint decisions.

The model benchmark keeps the declared 9B reference and trained candidate paired by task, repetition, and seed. The Fable A/B is a separate comparison between the project base model and that same candidate under one pinned Fable orchestration. Prompts, tools, starting states, limits, sampling, and verifiers must remain identical within each suite.

`decisions.confidence` controls the task-clustered bootstrap interval for the model benchmark. The Fable rule uses its blind preference rate and counts `minimum_tasks` by unique task ID; extra seeds or repetitions never substitute for another held-out task.

`autotrainer evaluate plan --write` freezes task, model, adapter, environment, runner, and fairness fingerprints. `evaluate run` executes command-backed suites; `evaluate export` and `evaluate ingest` exchange requests/results with external runners; `evaluate review` manages blind choices; and `evaluate report` writes separate suite decisions. Placeholder revisions, versions, digests, commands, or missing adapter bytes block planning. See the [V1 handoff plan](../V1-HANDOFF.md) for the required real-run evidence.

### Producer result envelope

`autotrainer-evaluation-result-v1` is one JSON object per trial; its machine-readable contract is [`schemas/evaluation-result.schema.json`](../../schemas/evaluation-result.schema.json). Copy every identity field from the exported request; do not reconstruct it from filenames. A successful result has this exact producer-facing shape:

```json
{
  "schema_version": 1,
  "plan_id": "<request.plan_id>",
  "trial_id": "<request.trial_id>",
  "suite_id": "<request.suite_id>",
  "arm_id": "<request.arm_id>",
  "task_id": "<request.task_id>",
  "repetition": 0,
  "seed": 1701,
  "status": "completed",
  "producer": {
    "name": "<request.runner.producer>",
    "version": "<request.runner.version>",
    "orchestration_sha256": "<request.runner.orchestration_sha256>",
    "model_revision": "<request.candidate.model.revision>",
    "adapter_sha256": "<request.candidate.adapter.sha256>",
    "seed_honored": true,
    "fallback_models_used": false
  },
  "usage": {
    "input_tokens": 1234,
    "output_tokens": 567,
    "tool_calls": 8,
    "wall_time_seconds": 42.5
  },
  "output": {
    "patch": "patch.diff",
    "transcript": "transcript.jsonl",
    "review_artifact": "site.html"
  }
}
```

`status` is `completed`, `failed`, or `timeout`. A failed or timed-out producer can use an empty `output` object, but it must still preserve the planned and producer fields. For the Fable suite, provide an auditable `review_artifact` even for a failure if the run must proceed to blind-pair export. Every declared producer value must match the frozen plan; an arm without an adapter uses JSON `null` for `adapter_sha256`. Model, adapter, or runner mismatches fail fairness and score zero. Schema-invalid envelopes—including ignored-seed or fallback-model flags—are rejected before scoring.

The `usage` and `output` objects are required but may be empty; the three `output` keys are optional file references, not inline content. Each supplied path must be relative to the result JSON, remain inside that directory after resolution, name a regular file, and be at most 10 MiB. A completed trial without `patch` scores zero. Fable blind-review export additionally requires `review_artifact` on every ingested Fable trial. Do not add score fields: closed-schema validation rejects unknown or mistyped fields, and AutoTrainer instead copies evidence under content hashes and scores the patch in the local hidden-verifier environment.

When `evaluate ingest` receives a directory, every envelope must be named `result.json` inside its trial directory or end in `.result.json`; other JSON files are treated as evidence and ignored by envelope discovery. Passing one result file explicitly supports any filename. Reserve the envelope names for envelopes, not transcripts or review artifacts.

### Blind review JSONL

After `evaluate review export`, reviewers inspect `blind-pairs.jsonl` without seeing arm identities. Validate each import line against [`schemas/blind-review-row.schema.json`](../../schemas/blind-review-row.schema.json): it contains only the exported `pair_id`, a stable nonempty reviewer ID, and a positional choice.

```jsonl
{"pair_id":"pair-0123456789abcdef01234567","reviewer_id":"reviewer-01","choice":"left"}
{"pair_id":"pair-fedcba9876543210fedcba98","reviewer_id":"reviewer-01","choice":"both_fail"}
```

Allowed choices are exactly `left`, `right`, `tie`, and `both_fail`; never write an arm or model ID. The pair must exist in the sealed export, and a reviewer may submit at most one row per pair. Every pair must have exactly `reviewers_per_pair` distinct reviewer IDs: missing or extra votes make review completeness fail so one site cannot receive more weight than another. The preference rate gives a candidate win 1 point, a tie 0.5, and a control win or `both_fail` 0; `both_fail` stays in the denominator because a site pair that failed is evidence against a candidate-better claim. Imports are immutable, so validate and complete the entire review file before import; replacing accepted reviews requires removing the evaluation plan and starting again.

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

The default deliverable is an adapter package, not a copy of the base weights. The packager resolves the one candidate shared by both evaluation decisions and normally refuses to run until both decisions are verified. A package identifies the exact base-model revision and includes a manifest of payload hashes; adapter weights without that identity are incomplete.

`autotrainer package` assembles the local artifact but does not publish or serve it. `--allow-unverified` is limited to a package marked `unverified_development_artifact`; it cannot create a winner claim.
