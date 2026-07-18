# Getting started

AutoTrainer uses one project record, `autotrainer.yaml`. The GUI is the primary path for people and the CLI exposes the same services for agents.

```text
Projects -> Data -> Train -> Evaluate -> Serve
```

## Requirements

- Linux, or Ubuntu under WSL2 on Windows.
- Python 3.11.
- One NVIDIA GPU. The reference profile targets a 24 GiB card.
- A CUDA-enabled PyTorch build.
- Docker or Podman for verifier-backed GRPO or local benchmark tasks.
- Node.js 22 or newer for the web console and included frontend fixtures.
- Local disk space for two 9B snapshots, source clones, compiled data, checkpoints, and evidence.

Native Windows is suitable for editing and the web UI. Run CUDA and container work in Linux/WSL2.

## Install

```bash
git clone https://github.com/Okohedeki/AutoTrainer.git
cd AutoTrainer

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./services/trainer
```

Install the CUDA build of PyTorch selected for your driver, then install the pinned training extra:

```bash
python -m pip install -e './services/trainer[training]'
docker build -t autotrainer/frontend-runtime:0.1 -f infra/frontend-runtime/Dockerfile .
```

The reference package matrix is recorded in the project dependency declarations and checked by `autotrainer doctor`. Upgrade it as a tested set; Transformers, TRL, PEFT, bitsandbytes, and PyTorch interfaces change together.

## Run the console

From the repository root, use two terminals:

```bash
# Terminal 1: lightweight dashboard backend
autotrainer serve --config examples/frontend-expert/autotrainer.yaml
```

```bash
# Terminal 2: web console
npm install
npm run dev
```

Open the Vite URL, normally `http://127.0.0.1:3000`. The web app forwards `/api/v1` requests to `http://127.0.0.1:8765`.

The included example exercises authoring and service contracts. Its training and evaluation fixtures belong to the same repository, so they are not a valid independent holdout and do not prove model improvement.

## 1. Projects

Select the startup project or create a new specialist. A new GUI project is created under the dashboard's trusted `.autotrainer/projects` directory, validated, and activated in one operation. It shares the workspace model cache but owns its configuration, sources, checkpoints, evaluation evidence, and host record.

Agent equivalent:

```bash
autotrainer init ./my-specialist
cd ./my-specialist
```

The CLI can also list, create, and resolve projects inside an explicit managed root with `autotrainer projects`. Ordinary agent automation can simply pass the intended `--config` path.

## 2. Data

### Choose and download the training base

Open the model step. AutoTrainer first checks the project and standard Hugging Face caches on this machine. A complete supported snapshot appears under **On this machine**; choose **Use local** to pin that exact revision without downloading it again. This scan is bounded, local-only, and never searches arbitrary drives.

Otherwise, type a model or author in the Hugging Face search. Results are labeled **Ready for V1 training**, **Evaluation reference only**, or **Not verified for V1 training**. The search is broad; the guarded V1 trainer remains deliberately narrow.

Select `Qwen/Qwen3.5-9B`, keep its pinned revision, then choose **Use & download**. The dashboard runs the download as a durable job and reports Downloaded only after the local snapshot and receipt exist.

Agent equivalent:

```bash
autotrainer models search "Qwen 9B"
autotrainer models local --config autotrainer.yaml
autotrainer model use qwen3.5-9b-text --config autotrainer.yaml
autotrainer model download --config autotrainer.yaml
autotrainer model status --config autotrainer.yaml
```

`models local` returns opaque candidate IDs. If a compatible snapshot is found, use `autotrainer model use-local <candidate-id> --config autotrainer.yaml`; the service rechecks the snapshot and records its cache root itself.

Weights are not included in the Git repository. Public models normally need no token. For a gated model, authenticate with Hugging Face or supply `HF_TOKEN` to the process; the token is not written to project files.

### Download the benchmark reference

The model benchmark is fixed to `empero-ai/Qwythos-9B-Claude-Mythos-5-1M` at revision `14a29bae5143091aeaf87ad37120de4cd57d592c`. Use **Download reference** in the GUI or:

```bash
autotrainer model reference-download --config autotrainer.yaml
autotrainer model reference-status --config autotrainer.yaml
```

The reference cannot be used as the V1 training base.

### Add work

The input accepts a GitHub `owner/repository`, a GitHub URL, or a supported local path. For a repository, say what it contributes:

- **Accepted changes** - review small Git changes and their real instructions for SFT.
- **Practice tasks** - use the repository as a starting state for executable GRPO tasks.
- **Reference only** - retain code/style evidence without producing training rows.
- **Evaluation holdout** - isolate the repository from training for final tasks.

Accepted changes, practice tasks, and reference only may be combined. Evaluation holdout must be used alone.

```bash
autotrainer source add OWNER/REPOSITORY \
  --mode accepted_changes \
  --mode practice_tasks \
  --config autotrainer.yaml

autotrainer source add ./data/accepted.jsonl --config autotrainer.yaml
autotrainer source add ./tasks/train --config autotrainer.yaml
autotrainer source list --config autotrainer.yaml
```

A GitHub add clones into AutoTrainer-managed storage, checks out a detached pinned commit, and only then saves the source. Local Git repositories, `.jsonl` files, and task-pack directories stay where they are.

Raw repository code is not automatically a demonstration or an RL environment. The GUI shows the next required action for each source. For accepted Git history:

```bash
autotrainer history list --config autotrainer.yaml
autotrainer history review CANDIDATE_ID --approve \
  --instruction "Describe the task this accepted change solved" \
  --rights-confirmed \
  --config autotrainer.yaml
```

## 3. Train

The GUI's Train action performs preparation before launching the selected path. Agents use:

```bash
autotrainer prepare --config autotrainer.yaml
autotrainer train auto --config autotrainer.yaml
```

Preparation validates YAML and paths, scans sources, compiles inspectable trainer inputs, selects a recipe, verifies model receipts, and checks the GPU/package/container runtime without loading weights.

The compiled learning signal selects:

- demonstrations -> QLoRA SFT;
- executable tasks -> QLoRA GRPO;
- both -> SFT, then GRPO continues the produced SFT adapter.

The Training view shows observed stages, logs, loss/reward metrics, and output adapters from the durable job record. It does not synthesize progress. `train sft --dry-run`, `train rl --dry-run`, and the lower-level validation commands remain available for diagnosis.

Only one Train, Evaluate, or Host operation may own GPU 0. A second local project or agent command receives a busy error instead of loading another 9B model.

## 4. Evaluate

Add multiple task packs from repository/project families that contributed no training code, demonstrations, mutation seeds, or rollouts. Then freeze and run the local benchmark:

```bash
autotrainer evaluate plan --write --config autotrainer.yaml
autotrainer evaluate run --suite model_benchmark --config autotrainer.yaml
```

The built-in runner compares the pinned Qwythos reference and the candidate adapter one arm at a time on the same task matrix. Results are generated, stored, applied, and scored with the trusted verifier. The GUI graphs actual build, regression, task, responsive, design, and patch-quality evidence as trials complete.

Fable is a separate external suite. Its missing runtime or placeholder identity does not block the local model benchmark. When a pinned Fable setup exists:

```bash
autotrainer evaluate export --suite fable_ab --output ./fable-requests --config autotrainer.yaml
# Produce results with the separately pinned Fable setup.
autotrainer evaluate ingest --suite fable_ab ./fable-results --config autotrainer.yaml
autotrainer evaluate review export --suite fable_ab --output ./blind-review --config autotrainer.yaml
# Collect blind reviewer rows.
autotrainer evaluate review import --suite fable_ab ./reviews.jsonl --config autotrainer.yaml
autotrainer evaluate report --config autotrainer.yaml
```

AutoTrainer does not bundle or emulate Fable.

## 5. Serve

After downloading the base or producing an adapter, start the local model from the GUI or CLI:

```bash
autotrainer host start --adapter auto --config autotrainer.yaml
autotrainer host status --config autotrainer.yaml
autotrainer host test "Create an accessible settings form." --config autotrainer.yaml
autotrainer host stop --config autotrainer.yaml
```

`--adapter auto` prefers a completed GRPO adapter, then SFT, then the base. Explicit choices are `grpo`, `sft`, and `base`.

The default endpoint is `http://127.0.0.1:8791` with `/health`, `/v1/models`, and `/v1/chat/completions`. This is a small OpenAI-compatible subset: text messages, non-streaming responses, one serialized bounded request, and loopback access only.

Do not confuse it with `autotrainer serve`, which runs the dashboard control API and never loads model weights.

## Windows and WSL2

From an elevated PowerShell prompt:

```powershell
wsl --install -d Ubuntu
wsl --update
```

Install the current NVIDIA Windows driver with WSL support. Do not install a second Windows GPU driver inside the Linux distribution. In Ubuntu, verify `nvidia-smi` and `torch.cuda.is_available()`.

The checkout may remain under `/mnt/h/AutoTrainer`, but caches, containers, rollouts, and checkpoints perform better on the WSL Linux filesystem:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export AUTOTRAINER_HOME="$HOME/.local/share/autotrainer"
```

## Current limits

- Text-only 9B causal language model path; no image or multimodal inputs.
- One local GPU; no cloud, distributed, or multi-GPU training.
- GitHub repositories or supported local files only for normal V1 onboarding.
- No bundled weights and no completed real 9B training/evaluation proof in this checkout.
- No included Fable runtime, outputs, or blind reviews.
- Local model hosting is loopback-only, non-streaming, and not public deployment.
- Interrupted jobs leave durable evidence, but optimizer checkpoint resume is not automatic for every path.

See [V1 handoff](../V1-HANDOFF.md) for the remaining proof work.
