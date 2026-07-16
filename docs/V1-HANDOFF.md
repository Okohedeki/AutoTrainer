# AutoTrainer V1 handoff

This is the remaining path from the implemented local workflow to an honestly
verified V1 result. Do not call an adapter a V1 winner until both held-out
comparisons report `verified_better: true`.

## What works now

- A three-step GUI for people: choose and download the model, add work and
  optionally review accepted Git changes, then prepare and start training.
- Agent commands that call the same Python services and update the same
  `autotrainer.yaml` project record.
- Automatic source inference for GitHub repositories and supported local
  repository, JSONL, and task-pack paths.
- Immutable model and source pins, offline-only training loads, deterministic
  compilation, and train/evaluation leakage checks.
- Conditional one-GPU QLoRA training:
  - **teach:** SFT from accepted examples;
  - **practice:** GRPO from verifier-backed tasks, starting from the base QLoRA
    policy or an explicitly selected compatible adapter;
  - **both:** SFT followed by GRPO on the produced adapter.
- One local training job started and polled through the GUI, with
  `autotrainer train auto` as the agent equivalent.
- Network-isolated frontend episodes, reproducible evaluation plans, local
  result verification, blind-review import/export, and gated packaging.

The normal agent path is:

```bash
autotrainer models list
autotrainer model use qwen3.5-9b-text --config autotrainer.yaml
autotrainer model download --config autotrainer.yaml

autotrainer source add SOURCE --config autotrainer.yaml
autotrainer source list --config autotrainer.yaml

# Optional for an added Git repository.
autotrainer history list --config autotrainer.yaml
autotrainer history review CANDIDATE_ID --approve \
  --instruction "Describe the accepted change" --rights-confirmed \
  --config autotrainer.yaml

autotrainer prepare --config autotrainer.yaml
autotrainer train auto --config autotrainer.yaml
```

The GUI performs those same operations without requiring users to copy the old
`validate`, `scan`, `compile`, `plan`, and `doctor` sequence. Those commands and
the stage-specific `train sft` / `train rl` commands remain available for
diagnosis and controlled experiments.

## What is not yet proven

- No full 9B GPU training path has completed on this checkout. Code-level tests
  and recipe dry runs are not a training result.
- No candidate has beaten the declared 9B reference on a production-sized,
  independent held-out benchmark.
- No Fable base-versus-candidate outputs and blind human reviews have been
  completed and ingested.
- The bundled evaluation task is an authoring fixture in this repository, not
  an independent holdout. The model-agent command, Fable version, and
  orchestration digest are still placeholders.
- This Windows host has not yet supplied the validated WSL2/Linux CUDA,
  dependency, and container runtime needed for the real run.
- The local job record is durable across page and backend restarts. A forced
  backend stop marks an active job interrupted; optimizer checkpoint resume is
  not yet automatic, so retry starts the selected path again.

## Ordered continuation work

### 1. Supply the real learning data and held-out tasks

Use only sources the operator has the right to train on. Add one or both
learning signals:

- accepted prompt/response examples, including reviewed accepted Git changes,
  for **teach**;
- resettable executable tasks with hidden verifiers for **practice**.

Add multiple evaluation tasks from repositories and project families that
contributed no training code, examples, mutation seeds, or rollouts. Keep
`evaluation.dataset` separate from the optional training-time
`grpo.eval_dataset`.

Acceptance:

- Preparation compiles inspectable SFT rows, GRPO rows, or both under
  `.autotrainer/compiled` and selects the expected recipe.
- Raw repository code contributes no training rows without an approved example
  or executable task contract.
- Train and evaluation repository identities, commits, project families, and
  `groupId` values do not overlap.
- Each held-out accepted/reference patch passes its regression, browser,
  accessibility, responsive, and hidden-verifier checks.

### 2. Run the selected path on one GPU

Use the GUI by selecting and downloading the model, adding the final sources,
reviewing any history examples, selecting **Prepare training**, and then
**Start training**. Agents run the equivalent commands shown above.

Model download is mandatory. Preparation must reject a missing, mutable, or
mismatched local snapshot before enabling training.

Acceptance:

- The project uses the supported text-only `Qwen/Qwen3.5-9B` profile at an
  immutable revision and the recorded snapshot exists locally.
- Preparation reports one BF16-capable GPU, the pinned dependency matrix, and a
  usable Docker or Podman runtime for any practice stage.
- **teach** writes a valid PEFT adapter tied to the locked base revision.
- **practice** writes a GRPO-trained QLoRA adapter from the selected base or a
  verified compatible input adapter.
- **both** writes an SFT adapter and makes GRPO continue that exact adapter into
  a different output directory.
- Resolved recipes, dependency versions, metrics, and adapter hashes are kept.

### 3. Pin the two proof runners and freeze the plan

Point the AutoTrainer evaluation arm at the final adapter produced by the
selected path. Replace every `REPLACE_WITH_*` runner and digest value.

- The model benchmark runner must force the chosen model/adapter, consume
  `{request}`, write `{result}`, honor the seed, disable fallback models, and
  run arms sequentially on one GPU.
- The Fable runner identity must include its exact version and an orchestration
  SHA-256 covering prompts, tools, budgets, fallback policy, and model routing.

```bash
autotrainer evaluate plan --write --config autotrainer.yaml
```

Acceptance:

- Planning includes immutable model revisions, adapter bytes, held-out tasks,
  environment image, seeds, and runner identities.
- Unchanged inputs produce the same plan ID; changing any of those inputs
  changes it.

### 4. Run both held-out comparisons

```bash
autotrainer evaluate run --suite model_benchmark --config autotrainer.yaml
autotrainer evaluate export --suite fable_ab --output ./fable-requests --config autotrainer.yaml
# Run the exported requests through the pinned Fable setup.
autotrainer evaluate ingest --suite fable_ab ./fable-results --config autotrainer.yaml
autotrainer evaluate review export --suite fable_ab --output ./blind-review --config autotrainer.yaml
# Collect independent reviewer choices.
autotrainer evaluate review import --suite fable_ab ./reviews.jsonl --config autotrainer.yaml
autotrainer evaluate report --config autotrainer.yaml
```

Acceptance:

- Producer fingerprints match the frozen plan and submitted patches pass local
  build, regression, browser, and hidden-verifier checks.
- Missing and failed trials remain in denominators as zeroes.
- Blind pairs have the configured number of distinct reviewers; model and
  Fable results remain separate.
- Both suites report `verified_better: true`. `observed_better` alone is not V1
  proof.

### 5. Package from a clean clone

```bash
autotrainer package --config autotrainer.yaml
python -m unittest discover -s services/trainer/tests -v
npm ci
npm test
```

Acceptance:

- A clean clone reproduces compilation and evaluation-plan fingerprints.
- Packaging rejects an unverified winner unless the operator explicitly asks
  for an `unverified_development_artifact`.
- The manifest records the base revision, adapter hash, evaluation plan,
  reports, source licenses, recipes, and payload digests.

## Deferred inputs

- Use `empero-ai/Qwythos-9B-Claude-Mythos-5-1M` at immutable revision
  `14a29bae5143091aeaf87ad37120de4cd57d592c` as the declared `reference_9b`
  benchmark arm, not as the V1 trainable project model. Its runner must prove a
  text-only loading path and fail closed rather than enable vision inputs.
- Research a permissively licensed open-source CRM later. Check code and data
  licenses, privacy constraints, project-family boundaries, and suitability for
  demonstrations or resettable verifier-backed tasks before adding it.

## Follow-up engineering debt

- Add stage/checkpoint resume so an interrupted combined path can continue
  without repeating a completed SFT stage.
- Load the published JSON Schema in the installed CLI as well as performing the
  current semantic Python validation.
- Add a pinned Hugging Face/TRL construction smoke job; normal CI does not
  install the multi-gigabyte CUDA stack.
- Add a persistent local model runner so a 9B model is loaded once per
  evaluation arm instead of once per trial.
- Show prepared artifacts, job logs, plans, reports, and packages in the GUI
  without introducing GUI-only state.

## Checkpoint discipline

Keep commits small and push each green checkpoint:

1. Final learning data and held-out verifier tests.
2. Real training output for the selected teach, practice, or combined path.
3. Pinned model-agent and Fable runners with a frozen plan.
4. Evaluation evidence.
5. Verified package release.
