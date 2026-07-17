# AutoTrainer V1 handoff

The V1 product path is implemented:

```text
Projects -> Data -> Train -> Evaluate -> Serve
```

The remaining work is real 9B execution and evidence. Service/UI tests and dry runs are not model-quality proof.

## Implemented

- GUI-first project lifecycle with the CLI as the equivalent agent surface.
- Safe project creation/activation and one config/artifact boundary per project.
- Hugging Face search with explicit compatibility labels.
- Durable base and benchmark-reference download jobs and immutable receipts.
- GitHub source add that clones and pins before saving, plus supported local Git/JSONL/task-pack inputs.
- Explicit source intent and reviewed accepted-history examples.
- Deterministic compilation and conditional QLoRA SFT, GRPO, or SFT-to-GRPO training.
- Durable observed training telemetry and GUI graphs.
- Frozen evaluation plans, durable trial evidence, trusted local verification, and live rubric graphs.
- Built-in text-only benchmark producer comparing the pinned Qwythos reference with the candidate.
- Deferred external Fable request/ingest/blind-review/report contracts.
- Exclusive cross-project GPU coordination for Train, Evaluate, and Host.
- A separate loopback model process with `/v1/models` and non-streaming `/v1/chat/completions`.

## Not yet proven or bundled

- No model weights are included or known to be downloaded in a fresh clone.
- No complete real 9B SFT, GRPO, or combined run has been recorded for this repository.
- No candidate has beaten the reference on a production-sized independent holdout.
- No pinned local Fable runtime, Fable outputs, or completed blind review exists.
- The included evaluation fixture belongs to the AutoTrainer repository and is not an independent holdout.
- Local hosting is text-only, loopback-only, non-streaming, and handles one bounded GPU request at a time. It is not public deployment.
- Durable interruption records exist, but automatic optimizer resume is not complete for every path.

## Fixed benchmark contract

The local model benchmark reference is:

- `empero-ai/Qwythos-9B-Claude-Mythos-5-1M`
- revision `14a29bae5143091aeaf87ad37120de4cd57d592c`
- evaluation-only; never the V1 training base.

The built-in producer owns the text prompt, bounded source context, offline 4-bit loading path, and result envelope. Its frozen identity includes evaluator code and dependency versions. It loads one arm at a time on GPU 0 and re-verifies patches locally.

Fable remains an external second proof. Placeholder Fable version/digest marks only that suite deferred; it does not block the local model benchmark.

## Path to a verified V1

### 1. Supply real learning and holdout data

- Add work the operator has the right to train on.
- Review accepted changes for SFT and/or add resettable verifier-backed practice tasks.
- Add multiple evaluation task packs from repositories/project families never used for training, mutations, rollouts, or checkpoint selection.
- Inspect compiled rows and confirm train/evaluation group isolation.

Acceptance: `prepare` selects the intended teach/practice/both path and reports no training blockers.

### 2. Download both 9B snapshots

GUI: use **Use & download** for the training base and **Download reference** for Qwythos.

CLI:

```bash
autotrainer model download --config autotrainer.yaml
autotrainer model reference-download --config autotrainer.yaml
autotrainer doctor --config autotrainer.yaml
```

Acceptance: immutable local receipts match both configured revisions; CUDA, pinned packages, and Docker/Podman are ready.

### 3. Run the selected training path

```bash
autotrainer prepare --config autotrainer.yaml
autotrainer train auto --config autotrainer.yaml
```

Acceptance:

- SFT writes a valid PEFT adapter when demonstrations exist.
- GRPO writes a verifier-trained adapter when tasks exist.
- A combined path continues the exact SFT adapter and writes GRPO to a separate output.
- Resolved recipes, events, checkpoints, and adapter hashes remain durable.

### 4. Run the built-in model benchmark

Point `evaluation.arms.autotrainer.adapter` at the actual completed stage, then:

```bash
autotrainer evaluate plan --write --config autotrainer.yaml
autotrainer evaluate run --suite model_benchmark --config autotrainer.yaml
autotrainer evaluate report --config autotrainer.yaml
```

Acceptance:

- the frozen plan includes independent tasks, exact model/adapter bytes, seeds, environment, and built-in runner identity;
- every planned trial has a valid durable result or explicit zero-scored failure;
- trusted local verification supplies the displayed rubric values;
- the model decision meets its unique-task and confidence rules.

### 5. Complete the external Fable proof

First replace the placeholder Fable version and orchestration digest with a real pinned setup. Then:

```bash
autotrainer evaluate export --suite fable_ab --output ./fable-requests --config autotrainer.yaml
# Produce outputs using the separately pinned Fable runtime.
autotrainer evaluate ingest --suite fable_ab ./fable-results --config autotrainer.yaml
autotrainer evaluate review export --suite fable_ab --output ./blind-review --config autotrainer.yaml
# Collect complete blind reviewer rows.
autotrainer evaluate review import --suite fable_ab ./reviews.jsonl --config autotrainer.yaml
autotrainer evaluate report --config autotrainer.yaml
```

Acceptance: producer identity matches the frozen plan, patches pass local verification, review completeness holds, and the Fable decision meets its declared rule.

### 6. Package and use

```bash
autotrainer package --config autotrainer.yaml
autotrainer host start --adapter auto --config autotrainer.yaml
autotrainer host test "Build a focused frontend change." --config autotrainer.yaml
autotrainer host stop --config autotrainer.yaml
```

Only call the package a verified winner when both model and Fable decisions are verified. `--allow-unverified` is for a labeled development artifact.

## Checkpoint discipline

Keep real-run commits small and push each green checkpoint:

1. final learning/holdout declarations and verifier tests;
2. model download receipts and runtime report (without weights or secrets);
3. completed training metadata and adapter hashes;
4. frozen local benchmark plan and evidence;
5. external Fable evidence and final package manifest.
