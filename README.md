# AutoTrainer

AutoTrainer is an Apache-2.0, local-first tool for turning a supported 9B text model into a specialist on one consumer GPU.

The GUI is the main path for people. The CLI is the equivalent path for agents. Both use the same Python services, `autotrainer.yaml`, compiled data, job records, and evaluation evidence.

```text
Projects -> Data -> Train -> Evaluate -> Serve
```

V1 stays narrow: licensed GitHub repositories or local files, QLoRA adapters, optional verifier-backed GRPO, a frozen language-matched local benchmark, and a loopback model endpoint. It does not add cloud training, multimodal inputs, full-weight training, or distributed orchestration.

## What V1 does

- Creates and switches local projects. GUI project creation writes, validates, and activates the new project as one operation.
- Detects supported bases already present in the project or Hugging Face cache, or searches Hugging Face and downloads an exact revision. Weights are not bundled with this repository.
- Adds allowlisted GitHub repositories or supported local paths. A GitHub add requires an SPDX license, clones the repository into managed storage, and pins a detached commit before saving it.
- Makes repository intent explicit: accepted changes, practice tasks, reference only, or evaluation holdout. Raw code is never silently called training data.
- Catalogs only pull requests merged into `main` or `master`, then lets a selected local or Anthropic model inspect each patch and propose its SFT/GRPO dataset treatment.
- Treats the local dataset as a first-class workspace: people inspect proposals and patches, approve or reject candidates, see language counts, and explicitly freeze the exact inputs before training.
- Compiles reviewed demonstrations for SFT and executable verifier-backed tasks for GRPO.
- Runs the useful learning path: QLoRA SFT, QLoRA GRPO, or SFT followed by GRPO on the same adapter.
- Enforces adapter-only refinement and a user-selected hard or soft per-process VRAM budget.
- Shows the GRPO curriculum at overview, task, or rollout granularity. The GUI graphs only values returned by the trainer and trusted verifier.
- Selects a shipped Python, TypeScript/React, C#, or C++ evaluation profile from the primary frozen training language and blocks mismatched held-out code.
- Freezes and runs a built-in, text-only model benchmark against the pinned Qwythos 9B reference.
- Hosts the downloaded base or a completed adapter behind a small, loopback-only OpenAI-compatible endpoint.
- Prevents Train, Evaluate, and Serve from competing for GPU 0, including across local projects and processes.

## Start the GUI

Use Python 3.11. GPU training, evaluation, and hosting should run in Linux or WSL2 with a CUDA build of PyTorch. Basic project setup and dry inspection do not load model weights.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./services/trainer

# Terminal 1: dashboard control API, default http://127.0.0.1:8765/api/v1
autotrainer serve --config examples/frontend-expert/autotrainer.yaml

# Terminal 2: human interface, normally http://127.0.0.1:3000
npm install
npm run dev
```

The console follows one lifecycle:

1. **Projects** - create or select a specialist workspace.
2. **Data** - choose/download the base and benchmark models, then add learning and held-out sources.
3. **Train** - let AutoTrainer validate, compile, choose the learning path, check the runtime, and start training. Watch observed metrics and logs.
4. **Evaluate** - freeze the plan and watch generation, verification, and rubric results from the trusted local benchmark.
5. **Serve** - load the base or best available adapter, copy the endpoint, and send a bounded test prompt.

`autotrainer serve` and `autotrainer host` are different:

- `autotrainer serve` runs the lightweight `/api/v1` dashboard backend on port `8765`. It does not load a model.
- `autotrainer host start` launches the selected model on port `8791` by default and exposes `/v1/models` and non-streaming `/v1/chat/completions`.

Both bind to loopback only in V1.

## Agent path

Agents can perform the same work without GUI-only state:

```bash
autotrainer init ./my-specialist
cd ./my-specialist

autotrainer models search "Qwen 9B"
autotrainer models local --config autotrainer.yaml
autotrainer model use qwen3.5-9b-text --config autotrainer.yaml
autotrainer model download --config autotrainer.yaml
autotrainer model reference-download --config autotrainer.yaml

autotrainer source add Okohedeki/example-repo \
  --mode accepted_changes \
  --mode practice_tasks \
  --config autotrainer.yaml
autotrainer source add ./data/accepted.jsonl --config autotrainer.yaml
autotrainer source add ./tasks/train --config autotrainer.yaml

autotrainer dataset sync --config autotrainer.yaml
autotrainer dataset status --config autotrainer.yaml
autotrainer dataset design CANDIDATE_ID --provider local --model MODEL_ID --config autotrainer.yaml
# Or use --provider anthropic with ANTHROPIC_API_KEY and a Claude model ID.
# Approve/reject the proposal, inspect the resulting local rows, then:
autotrainer dataset freeze --config autotrainer.yaml

autotrainer prepare --config autotrainer.yaml
autotrainer curriculum --config autotrainer.yaml --json
autotrainer train auto --config autotrainer.yaml

autotrainer evaluate plan --write --config autotrainer.yaml
autotrainer evaluate run --suite model_benchmark --config autotrainer.yaml

autotrainer host start --adapter auto --config autotrainer.yaml
autotrainer host status --config autotrainer.yaml
autotrainer host test "Build a focused account settings view." --config autotrainer.yaml
autotrainer host stop --config autotrainer.yaml
```

`prepare` performs input validation, source scanning, deterministic compilation, recipe selection, snapshot checks, and local runtime checks without loading the model. `train auto` repeats that preparation before using the GPU.

Stage-specific `train sft`, `train rl`, `validate`, `compile`, `plan`, and `doctor` commands remain available for diagnosis and controlled experiments.

## Models

The guarded V1 training profile is:

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

Hugging Face search is broad, but V1 compatibility claims are not. Results outside the tested profile are labeled unverified and cannot be selected for guarded training through the GUI.

The GUI also checks the configured project cache and the standard local Hugging Face cache without using the network. A structurally complete, supported snapshot appears as **Found locally** and can be adopted without downloading it again. Agents can run `autotrainer models local --config autotrainer.yaml`, then `autotrainer model use-local <candidate-id> --config autotrainer.yaml`. The candidate ID is opaque; callers never supply a cache path.

The fixed benchmark reference is `empero-ai/Qwythos-9B-Claude-Mythos-5-1M` at revision `14a29bae5143091aeaf87ad37120de4cd57d592c`. It is downloaded separately and is never offered as the training base.

Public model downloads normally need no token. Gated models use the operator's Hugging Face authentication or `HF_TOKEN`; AutoTrainer does not write the token to YAML or receipts.

## Learning path

QLoRA is the one-GPU adapter strategy. SFT and GRPO are separate, conditional learning stages:

- accepted prompt/response examples or reviewed Git changes -> **SFT**;
- executable tasks with reset, bounded tools, and hidden verifier -> **GRPO**;
- both signals -> **SFT, then GRPO continuing the SFT adapter**.

A merged pull request supplies accepted code and its GitHub task context, but it does not choose its own learning format. The selected dataset-design LLM proposes that treatment; an operator must still inspect and approve it. GRPO additionally requires an executable reset-and-verifier contract. Training is blocked until the local dataset is explicitly frozen.

## Language-matched evaluation

The built-in model benchmark compares the pinned Qwythos 9B reference with the trained candidate on the same held-out tasks. Auto-detection uses the primary language in the frozen training dataset. Shipped profiles initially cover Python, TypeScript/React, C#, and C++, with checks and metrics informed by HumanEval, MBPP, MultiPL-E, and HumanEval-X. Benchmark names are design references; AutoTrainer ships and audits its own task/verifier execution.

The frozen plan covers task content, model revisions, adapter bytes, runtime identity, seeds, and fairness policy. The runner loads one arm at a time on the single GPU, produces a durable result envelope, then scores the patch in the trusted local verifier environment. Fable is not part of the V1 requirement. Existing projects may explicitly opt into the external compatibility workflow with `autotrainer fable pin`.

## Honest status

The V1 workflow is implemented and tested at the service and UI level, but this checkout does **not** contain:

- downloaded model weights;
- a completed real 9B SFT/GRPO run;
- a production-sized held-out 9B benchmark result;

Local hosting is text-only, loopback-only, non-streaming, and serializes one bounded generation request on the GPU. It is not a public deployment system.

Do not call an adapter a verified winner until its configured language-matched held-out benchmark meets the declared decision rule. If a project explicitly adds another suite such as Fable, that suite's decision also becomes part of its proof contract.

## Documentation

- [Getting started](docs/guides/getting-started.md)
- [Data sources](docs/guides/data-sources.md)
- [Training](docs/guides/training.md)
- [Configuration](docs/guides/configuration.md)
- [Architecture](docs/architecture.md)
- [RL environment security](docs/rl-environment.md)
- [V1 handoff and proof work](docs/V1-HANDOFF.md)

## Development

```bash
python -m pip install -e ./services/trainer
python -m pytest services/trainer/tests -q

npm install
npm test
npm run build
```

## License

Apache License 2.0. Imported repositories, datasets, models, and generated adapters retain their own licenses. AutoTrainer records source declarations; it does not grant rights to third-party material.
