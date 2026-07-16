# AutoTrainer

AutoTrainer is an Apache-2.0, local-first foundry for turning a 9B base model into a verified frontend expert on one consumer GPU.

The source of truth is `autotrainer.yaml`, not the dashboard:

```text
declare model + sources
        ↓
scan and lock evidence
        ↓
compile demonstrations + executable tasks
        ↓
4-bit QLoRA SFT
        ↓
GRPO continues the same adapter
        ↓
Declared 9B reference vs trained candidate benchmark
        ↓
Fable + base 9B vs Fable + the same trained candidate
```

## What is usable now

- A project CLI for model and source declaration, validation, scanning, compilation, locking, planning, and runtime checks.
- A loopback-only local API and GUI model setup that use the same service operations as the CLI.
- Immutable Hugging Face model download receipts and offline-only training loads.
- Deterministic repository inventories and direct SFT JSONL compilation.
- Versioned executable frontend task packs with a Docker/Podman security boundary.
- A guarded Hugging Face QLoRA SFT runner.
- A guarded TRL GRPO runner that reloads the SFT adapter as trainable instead of creating a new adapter.
- A conservative RTX 4090 / 24 GB recipe and dry runs that do not import CUDA libraries.
- Immutable, paired evaluation plans with local result verification, separate model/Fable reports, and blind-review import/export.
- Winner-gated LoRA adapter packaging with auditable file hashes and an explicitly labeled unverified-development escape hatch.

The evaluation and packaging workflow is implemented, but this checkout does not contain the evidence for a verified V1 winner. A real run still needs enough independent held-out project families, pinned model-agent and Fable runners, a completed 9B SFT/GRPO run, and blind Fable reviews. The GUI now performs model selection and download through the shared local backend; training launch and telemetry remain guarded until their backend job contract is connected. See the [V1 handoff plan](docs/V1-HANDOFF.md) for the ordered continuation work.

## Quickstart

Use Python 3.11. GPU training should run in Linux or WSL2. Core inspection works without CUDA:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./services/trainer

autotrainer init my-frontend-expert
cd my-frontend-expert
```

Or start from the complete example:

```bash
python -m pip install -e ./services/trainer
autotrainer validate --config examples/frontend-expert/autotrainer.yaml
autotrainer source scan --config examples/frontend-expert/autotrainer.yaml
autotrainer compile --config examples/frontend-expert/autotrainer.yaml
autotrainer plan --config examples/frontend-expert/autotrainer.yaml
```

The example includes a small evaluation authoring fixture, not an independent repository holdout or a statistically useful benchmark. Evaluation planning remains blocked until that source is replaced with a genuinely separate repository, the placeholder runner identities are pinned, and the candidate adapter exists; add multiple independent held-out project families before making an improvement claim.

## Declare the model

Edit the `model` section or use the CLI:

```bash
autotrainer models list
autotrainer model use qwen3.5-9b-text \
  --config autotrainer.yaml
autotrainer model status --config autotrainer.yaml
autotrainer model download --config autotrainer.yaml
```

The V1 trainable project model is `Qwen/Qwen3.5-9B`, loaded through the text-only `Qwen3_5ForCausalLM` path. AutoTrainer never loads its processor, image inputs, or vision encoder, and aborts if a different class is instantiated. A custom model can be declared for authoring, but the guarded V1 training backend currently supports only this tested profile; the separately pinned 9B benchmark reference may use a different model through its external runner.

```yaml
model:
  provider: huggingface
  id: Qwen/Qwen3.5-9B
  revision: YOUR_IMMUTABLE_HUGGING_FACE_COMMIT
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

`model download` resolves a mutable Hugging Face revision, writes the immutable commit back to YAML, downloads the complete snapshot, and records a token-free receipt. Real training is offline-only and refuses a missing or mutable model. `autotrainer lock` records that model identity with local Git revisions in `.autotrainer/autotrainer.lock.json`. A published experiment should never rely on `main`.

## Point it at repositories and data

Repositories, demonstrations, and RL tasks are separate source kinds:

```bash
autotrainer source add ../storefront \
  --name storefront \
  --kind repository \
  --roles style,history,rl_seed \
  --revision HEAD \
  --config autotrainer.yaml

autotrainer source add ./data/accepted.jsonl \
  --name accepted-trajectories \
  --kind sft_jsonl \
  --config autotrainer.yaml

autotrainer source add './tasks/train/**/*.json' \
  --name frontend-train-tasks \
  --kind task_pack \
  --config autotrainer.yaml
```

For a Git URL, declare it the same way, then run `autotrainer source materialize <source-id> --config autotrainer.yaml` to create a detached local checkout and update the source URI.

A repository alone is **not** an SFT dataset or an RL environment:

- Final source code is reference/style evidence.
- Prompt → accepted response or tool trajectory JSONL is direct SFT data.
- Git history can become SFT only after a compiler reconstructs a non-leaking instruction and accepted patch.
- GRPO requires a starting revision, task instruction, isolated tools, hidden verifier, reward, and reset mechanism.

`source scan` reports exactly which role each input can serve. `compile` writes inspectable data beneath `.autotrainer/compiled/` and never silently turns raw code into demonstrations.

## Install the training runtime

First install the CUDA build of PyTorch selected for your host at the [official PyTorch installer](https://pytorch.org/get-started/locally/). Then install the pinned reference stack:

```bash
python -m pip install -e './services/trainer[training]'
docker build -t autotrainer/frontend-runtime:0.1 -f infra/frontend-runtime/Dockerfile .
autotrainer doctor --config autotrainer.yaml
```

The reference matrix is Python 3.11, PyTorch 2.13.0, Transformers 5.13.1, TRL 1.8.0, PEFT 0.19.1, Accelerate 1.14.0, Datasets 5.0.0, bitsandbytes 0.49.2, and jmespath 1.1.0. The rollout image pins Playwright 1.61.1; task repositories should use the matching Playwright package.

## Train

Validate the resolved recipes before spending GPU time:

```bash
autotrainer compile --config autotrainer.yaml
autotrainer train sft --dry-run --config autotrainer.yaml
autotrainer train sft --config autotrainer.yaml

# This dry run succeeds only after the SFT adapter exists.
autotrainer train rl --dry-run --config autotrainer.yaml
autotrainer train rl --config autotrainer.yaml
```

The smoke profile uses 4-bit NF4, LoRA rank 32, SFT batch 1 with gradient accumulation 8, and GRPO with two generations at a 2K completion limit. `beta: 0` avoids loading a second reference model. Increase group size or context only after measuring VRAM.

## Data and environment contracts

- [Getting started](docs/guides/getting-started.md)
- [Configuration reference](docs/guides/configuration.md)
- [Data-source rules](docs/guides/data-sources.md)
- [Training and RL environment](docs/guides/training.md)
- [Architecture](docs/architecture.md)
- [RL security model](docs/rl-environment.md)
- [V1 handoff and remaining proof work](docs/V1-HANDOFF.md)
- [Project schema](schemas/autotrainer.schema.json)
- [Frontend task schema](schemas/frontend-task.schema.json)
- [External evaluation result schema](schemas/evaluation-result.schema.json)
- [Blind-review row schema](schemas/blind-review-row.schema.json)

## Development

```bash
python -m pip install -e ./services/trainer
python -m unittest discover -s services/trainer/tests -v

npm ci
npm test
```

Run the human GUI with the local backend and Vite in separate terminals:

```bash
autotrainer serve --config examples/frontend-expert/autotrainer.yaml
npm run dev
```

The GUI calls `/api/v1`; Vite forwards it to the loopback backend at `127.0.0.1:8765`. Agents use the equivalent `autotrainer model ...` commands against the same YAML and model-service code.

## License

Apache License 2.0. Training data, model checkpoints, and imported repositories retain their own licenses; AutoTrainer records source license declarations but does not grant redistribution rights to third-party material.
