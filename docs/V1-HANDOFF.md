# AutoTrainer V1 handoff plan

This document is the continuation plan after the first usable training and evaluation implementation. It separates what the repository can do now from what still needs real data, a Linux/CUDA machine, and pinned external runners. Do not describe an adapter as a verified V1 winner until both decisions in the evaluation summary are true.

## Current implementation

The repository now provides:

- Declarative model, source, QLoRA, SFT, GRPO, environment, evaluation, and package contracts.
- Safe local or remote Git source materialization and immutable source/model locking.
- Deterministic SFT and RL dataset compilation with train/evaluation leakage checks.
- Guarded single-GPU QLoRA SFT and GRPO continuation of the same adapter.
- Network-isolated frontend episodes with bounded tools, deadlines, browser-test gates, and hidden verification.
- Two reproducible pairwise evaluation suites:
  - Declared 9B reference versus the AutoTrainer adapter under one model-agent runner.
  - Fable plus the base 9B versus Fable plus the same AutoTrainer adapter.
- Immutable result ingestion, local patch re-verification, separate suite reports, blind-review import/export, and auditable adapter packaging.

The CLI surface is:

```text
autotrainer source materialize
autotrainer compile
autotrainer train sft
autotrainer train rl
autotrainer evaluate plan
autotrainer evaluate run
autotrainer evaluate export
autotrainer evaluate ingest
autotrainer evaluate review export
autotrainer evaluate review import
autotrainer evaluate report
autotrainer package
```

## What is not yet proven

- No full Qwen3.5-9B SFT/GRPO job has been completed on this checkout. Recipe dry-runs and code-level tests are green, but that is not a GPU training result.
- The configured local model-agent command and Fable version/orchestration digest are placeholders until the operator supplies and pins real runners.
- The repository includes one newsletter evaluation task with deterministic regressions, browser requirements, and a hidden verifier. It is an authoring fixture, not an independent repository holdout or a statistically useful benchmark; a release benchmark needs multiple external held-out project families and at least the decision's configured minimum task count.
- No real Fable outputs or blind human reviews have been ingested.
- This Windows host does not currently provide the validated WSL/CUDA, pinned Hugging Face stack, and Docker runtime required for the full run.

## Ordered continuation work

### 1. Author and compile the final held-out tasks

Add evaluation tasks from repositories and project families that contributed no code, demonstrations, mutation seeds, or RL rollouts to training. The included newsletter task demonstrates the authoring format, but its starting project is stored in the AutoTrainer repository and therefore does not satisfy repository holdout. Every production task must include a locked starting revision, browser-accessibility behavior, responsive behavior, regressions, and a hidden verifier.

Declare the compiled final proof path as `evaluation.dataset`. Do not point `grpo.eval_dataset` at it: that optional field is training-loop validation and must remain a different file so benchmark tasks cannot influence optimization or checkpoint selection.

```bash
autotrainer validate --config autotrainer.yaml
autotrainer source scan --config autotrainer.yaml
autotrainer compile --config autotrainer.yaml
```

Acceptance:

- `compile` emits the configured `evaluation.dataset` with locked evaluation task rows.
- Each held-out starting state fails its target behavior and its accepted/reference patch passes the regression, browser, accessibility, responsive, and hidden checks.
- Train and evaluation repository identities, exact commits, project families, and `groupId` values do not overlap. Different source IDs or paths into the same repository do not establish holdout.
- Any `grpo.eval_dataset` is training-only, exists independently, and resolves to a different path.

Do not require `evaluate plan` to pass yet. It fingerprints real adapter bytes and runner pins, which do not exist until the next two steps.

### 2. Run the real single-GPU training sequence

Use Linux or WSL2 with exactly one visible NVIDIA GPU and the pinned training dependencies.

```bash
autotrainer doctor --config autotrainer.yaml
autotrainer lock --config autotrainer.yaml
autotrainer compile --config autotrainer.yaml
autotrainer train sft --dry-run --config autotrainer.yaml
autotrainer train sft --config autotrainer.yaml
autotrainer train rl --dry-run --config autotrainer.yaml
autotrainer train rl --config autotrainer.yaml
```

Acceptance:

- The project model is exactly the supported V1 training model, `Qwen/Qwen3.5-9B`, at an immutable revision.
- `doctor` reports one BF16-capable GPU, the exact dependency matrix, and a usable Docker or Podman runtime.
- SFT writes a valid PEFT adapter tied to the locked base revision.
- GRPO reloads that adapter as trainable and writes a different output directory.
- Resolved recipes, dependency versions, metrics, and adapter hashes are retained.

### 3. Pin both execution runners and freeze the plan

Point `evaluation.arms.autotrainer.adapter.path` at the completed GRPO adapter, then replace every `REPLACE_WITH_*` value in `evaluation.arms` and `evaluation.suites`.

- The model benchmark command must consume `{request}`, write `{result}`, force the selected model/adapter, honor the seed, disable fallback models, and run arms sequentially on one GPU.
- The Fable runner must record its exact version and an orchestration SHA-256 covering prompts, tool policy, budgets, fallback policy, and model routing.

```bash
autotrainer evaluate plan --write --config autotrainer.yaml
```

Acceptance:

- Planning succeeds only after the held-out dataset, adapter bytes, immutable model revisions, and both runner identities are available.
- Repeating it with unchanged inputs produces the same plan ID.
- Changing a task, model revision, adapter bytes, environment image, seed, or runner digest changes the plan ID.

### 4. Execute both proof suites

```bash
autotrainer evaluate plan --write --config autotrainer.yaml
autotrainer evaluate run --suite model_benchmark --config autotrainer.yaml
autotrainer evaluate export --suite fable_ab --output ./fable-requests --config autotrainer.yaml
# Run the exported requests through the pinned Fable setup. Name directory-mode
# envelopes result.json or *.result.json.
autotrainer evaluate ingest --suite fable_ab ./fable-results --config autotrainer.yaml
autotrainer evaluate review export --suite fable_ab --output ./blind-review --config autotrainer.yaml
# Collect reviewer JSONL choices.
autotrainer evaluate review import --suite fable_ab ./reviews.jsonl --config autotrainer.yaml
autotrainer evaluate report --config autotrainer.yaml
```

Acceptance:

- Every producer result conforms to [`schemas/evaluation-result.schema.json`](../schemas/evaluation-result.schema.json); planned identities and producer fingerprints match exactly.
- Every imported review line conforms to [`schemas/blind-review-row.schema.json`](../schemas/blind-review-row.schema.json) and uses only `left`, `right`, `tie`, or `both_fail`.
- Every blind pair has exactly the configured number of distinct reviewers. Extra or missing votes fail completeness, and `both_fail` remains in the preference denominator with zero candidate credit.
- Missing or failed trials remain in denominators as zeroes.
- Producer scores are ignored; submitted patches pass local build, regression, browser, and hidden-verifier checks.
- Model and Fable suites remain separate in reports.
- Both decisions expose `observed_better` and `verified_better`; only the latter satisfies V1.

### 5. Package and release from a clean clone

```bash
autotrainer package --config autotrainer.yaml
python -m unittest discover -s services/trainer/tests -v
npm ci
npm test
```

Acceptance:

- A clean clone can install the core CLI and reproduce compilation and evaluation-plan fingerprints.
- Packaging refuses an unverified winner unless explicitly creating an `unverified_development_artifact`.
- The package manifest identifies the base model revision, adapter hash, evaluation plan, reports, source licenses, and every payload-file digest.
- Documentation and the dashboard no longer describe implemented commands as future work.

## Recorded later inputs

These inputs are deliberately deferred and do not reorder the five steps above:

- Use `empero-ai/Qwythos-9B-Claude-Mythos-5-1M` at immutable revision `14a29bae5143091aeaf87ad37120de4cd57d592c` as the declared `reference_9b` benchmark arm. Do not change the trainable project `model.id`; V1 training remains on `Qwen/Qwen3.5-9B`.
- Before pinning that reference runner, prove a text-only loading and generation path. Its published config names `Qwen3_5ForConditionalGeneration` and includes vision configuration, so the runner must avoid image processors and vision inputs, remain within the single-GPU budget, and block the arm rather than silently enabling multimodal behavior.
- Research a permissively licensed open-source CRM as a later fine-tuning target. Record its code/data licenses, privacy constraints, project-family boundaries, and suitability for demonstrations and resettable verifier-backed tasks before adding it to `sources`; do not mix it into the current held-out proof suite.

## Follow-up engineering debt

These are important but should not block collecting the first honest V1 evidence:

- Load the published JSON Schema in the installed CLI as well as maintaining semantic Python validation.
- Add a pinned Hugging Face/TRL trainer-construction smoke job; current CI does not install the multi-gigabyte CUDA training stack.
- Add a persistent/batched local model runner so a 9B model is loaded once per arm instead of once per trial.
- Turn the read-only web preview into an artifact viewer for plans, trial completeness, reports, and packages.
- Evolve that viewer into the primary local control surface for editing the same declarative configuration and starting, monitoring, or resuming jobs through the shared backend. Keep every operation reproducible from the CLI and never introduce GUI-only experiment state.

## Checkpoint discipline

Continue with small commits and push after each green checkpoint:

1. Held-out tasks and verifier tests.
2. Real SFT smoke result.
3. Real GRPO smoke result.
4. Pinned model-agent/Fable runner integration and frozen plan.
5. Evaluation evidence and package release.
