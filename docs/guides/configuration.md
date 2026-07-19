# Configuration reference

`autotrainer.yaml` is the source of truth for one specialist project. The GUI edits this file through the same validated services used by the CLI.

```bash
autotrainer validate --config autotrainer.yaml
```

The published editor/tooling schema is [`schemas/autotrainer.schema.json`](../../schemas/autotrainer.schema.json). CLI commands also run semantic Python validation for resolved paths, role compatibility, training/evaluation separation, adapter state, and runtime prerequisites.

## Paths and revisions

- Relative paths resolve from the directory containing `autotrainer.yaml`.
- Git and model revisions used for real work should be immutable commit hashes.
- GitHub sources added through the service are cloned and rewritten to managed local paths at pinned detached commits.
- Local repository paths stay in place and must resolve to a Git root.
- Credentials and Hugging Face tokens are process input, never YAML fields.

## Top level

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

Unknown top-level fields are rejected.

## `project`

```yaml
project:
  name: polished-frontend-9b
  seed: 42
  artifact_dir: .autotrainer
```

`artifact_dir` holds model/source receipts, compiled data, training events, evaluation evidence, packages, and the local host receipt. Managed GUI projects keep separate artifact directories while sharing the workspace model cache.

## `model`

The guarded V1 training profile is:

```yaml
model:
  provider: huggingface
  id: Qwen/Qwen3.5-9B
  revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a
  cache_dir: .autotrainer/model-cache
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

The guarded trainer verifies the supported text-only class and refuses image/processor loading. Hugging Face search may return other models, but an unverified result does not become V1-compatible by appearing in search.

Use the service rather than manually guessing revisions:

```bash
autotrainer models search "Qwen 9B"
autotrainer model use qwen3.5-9b-text --config autotrainer.yaml
autotrainer model download --config autotrainer.yaml
```

The benchmark reference is not this field. AutoTrainer fixes it separately to `empero-ai/Qwythos-9B-Claude-Mythos-5-1M` at `14a29bae5143091aeaf87ad37120de4cd57d592c` and records a separate reference download receipt.

## `sources`

Each source is a `repository`, `sft_jsonl`, or `task_pack`:

```yaml
sources:
  - id: owner-storefront
    kind: repository
    uri: .autotrainer/sources/owner-storefront
    revision: 34f8c81b3b7f5b65d8e63d82abac42b66fb60f50
    partition: train
    roles: [history, rl_seed]
    include: [src/**, tests/**]
    exclude: [node_modules/**, dist/**]
    license:
      spdx: Apache-2.0
      attribution: https://github.com/owner/storefront
```

Canonical roles are:

- `style` - reference-only code;
- `history` - accepted-change candidates for SFT review;
- `rl_seed` - repository starting states for tasks;
- `demonstrations` - explicit SFT records;
- `rl_tasks` - executable training tasks;
- `evaluation` - held-out repositories/task packs.

The GUI/normal CLI maps the repository modes `reference_only`, `accepted_changes`, `practice_tasks`, and `evaluation_holdout` to those roles. See [Data sources](data-sources.md).

## `qlora`

```yaml
qlora:
  rank: 32
  alpha: 64
  dropout: 0.0
  target_modules: all-linear
  bias: none
```

The base is loaded in 4-bit NF4 and remains frozen. This section defines the trainable PEFT adapter. Changing its architecture creates a different experiment.

## `sft`

```yaml
sft:
  enabled: true
  dataset: .autotrainer/compiled/sft/train.jsonl
  output_dir: .autotrainer/checkpoints/sft
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 0.0001
  max_length: 2048
  gradient_checkpointing: true
  assistant_only_loss: true
  completion_only_loss: true
  packing: false
  bf16: true
  tf32: true
  seed: 42
  logging_steps: 5
  save_steps: 50
  save_total_limit: 2
```

The compiler writes `dataset`; authored inputs remain declared under `sources`. Targets are text-only assistant/completion messages. The GUI graphs observed trainer events using `logging_steps`; it does not infer loss between callbacks.

## `grpo`

```yaml
grpo:
  enabled: true
  algorithm: grpo
  dataset: .autotrainer/compiled/rl/train.jsonl
  start_from: .autotrainer/checkpoints/sft
  output_dir: .autotrainer/checkpoints/grpo
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 2
  num_generations: 2
  calibration_generations: 4
  generation_batch_size: 2
  learning_rate: 0.00001
  max_steps: 100
  max_completion_length: 2048
  max_tool_calling_iterations: 8
  beta: 0.0
  loss_type: dapo
  use_vllm: false
  gradient_checkpointing: true
  bf16: true
  tf32: true
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  seed: 42
```

Before the first optimizer step, GRPO samples `calibration_generations` frozen
starting-policy rollouts per task through the same environment and verifier.
The value must be at least four and divisible by `num_generations`. Training
stops if a task produces no within-group reward variation across the sampled
groups, because that task would provide zero relative advantage at the start
of the run. The resulting evidence is written to
`starting_policy_calibration.json` beside the adapter.

`start_from` is `base` for a practice-only fresh adapter or a completed compatible PEFT adapter path. When SFT and GRPO are both enabled, it must equal `sft.output_dir`; `train auto` passes the completed SFT adapter directly into the GRPO stage.

`beta: 0.0`, two generations, and `use_vllm: false` are memory-conscious V1 choices, not universal claims about the best GRPO recipe. `grpo.eval_dataset`, if used, is training feedback and must not equal final `evaluation.dataset`.

## `environment`

```yaml
environment:
  factory: autotrainer.environments.frontend:FrontendEnvironment
  backend: docker
  image: autotrainer/frontend-runtime:0.1
  network: none
  max_tool_output_chars: 12000
  episode_timeout_seconds: 900
```

Docker or Podman runs one disposable frontend environment at a time. The policy sees bounded named tools and checks, not a host shell or hidden verifier. External networking must remain disabled.

## `evaluation`

New projects receive a built-in local model benchmark and a deferred external Fable suite:

```yaml
evaluation:
  task_pack: held-out-frontend
  dataset: .autotrainer/compiled/rl/evaluation.jsonl
  task_split: evaluation
  repetitions: 3
  seeds: [1701, 1702, 1703]
  holdout_unit: repository
  primary_metric: verified_task_success
  candidates: [reference_9b, base_fable, autotrainer]
  arms:
    reference_9b:
      label: Qwythos 9B reference
      role: reference
      parameter_class: 9b
      model:
        provider: huggingface
        id: empero-ai/Qwythos-9B-Claude-Mythos-5-1M
        revision: 14a29bae5143091aeaf87ad37120de4cd57d592c
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
        path: .autotrainer/checkpoints/grpo
        stage: grpo
  suites:
    model_benchmark:
      kind: model_benchmark
      arms: [reference_9b, autotrainer]
      runner:
        type: builtin
    fable_ab:
      kind: fable_ab
      arms: [base_fable, autotrainer]
      runner:
        type: external
        producer: fable
        version: REPLACE_WITH_FABLE_VERSION
        orchestration_sha256: sha256:0000000000000000000000000000000000000000000000000000000000000000
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
    pair_position_policy: deterministic_counterbalance
    execution_order_policy: frozen_per_suite
    per_trial_arm_randomization: false
    failures_score_zero: true
    allow_unplanned_reruns: false
  decisions:
    confidence: 0.95
    model_benchmark:
      candidate: autotrainer
      control: reference_9b
      metric: verified_task_success
      minimum_delta: 0.0
      minimum_tasks: 5
    fable_ab:
      candidate: autotrainer
      control: base_fable
      metric: blind_preference_rate
      minimum_rate: 0.5
      minimum_tasks: 5
```

The candidate adapter path must point to the stage being evaluated. A project that trains only SFT should update the path/stage to its SFT output before freezing a proof plan.

The built-in runner owns its prompt/loader protocol. Its identity includes the installed evaluator code digest and pinned dependency versions. `evaluate plan --write` also freezes tasks, model revisions, adapter bytes, environment, seeds, fairness settings, and the derived trial matrix. Plan or trial tampering fails closed.

Single-GPU execution is truthful: pair positions are counterbalanced for analysis, but trials run in a frozen grouped order so only one 9B arm is loaded at a time. The legacy `randomize_arm_order: true` fairness form remains readable for older projects; new configs use the explicit policies above.

Fable placeholders make only `fable_ab` deferred. They do not block planning or running `model_benchmark`. Replace them only when a real, pinned Fable producer exists.

### Evaluation evidence

Every planned trial has a durable `autotrainer-evaluation-result-v1` envelope validated against [`schemas/evaluation-result.schema.json`](../../schemas/evaluation-result.schema.json). The envelope repeats the frozen plan/trial identity, producer identity, model/adapter digest, seed/fallback assertions, usage, and relative evidence paths. AutoTrainer re-scores submitted patches in the trusted local verifier; producer-supplied scores are not accepted.

External result paths must remain relative to their envelope, stay inside its directory, and satisfy file limits. Failed or timed-out trials remain in decision denominators as zeroes.

Fable blind-review imports use [`schemas/blind-review-row.schema.json`](../../schemas/blind-review-row.schema.json) and contain only a frozen `pair_id`, reviewer ID, and `left`, `right`, `tie`, or `both_fail` choice. Arm identities never belong in the reviewer file.

## `package`

```yaml
package:
  type: lora_adapter
  merge_base_weights: false
```

The normal package is an adapter plus provenance, hashes, resolved recipe, licenses, and evaluation reports; it is not a copy of base weights. A winner package requires both evaluation decisions. `--allow-unverified` creates a clearly labeled development artifact, never a winner claim.

Packaging and hosting are separate. `autotrainer package` assembles files. `autotrainer host start` reads the downloaded base and a completed SFT/GRPO adapter directly from the configured output directories.
