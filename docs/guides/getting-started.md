# Getting started

AutoTrainer uses one project record: `autotrainer.yaml`. The GUI is the primary
path for people; the CLI exposes the same local services for agents. Neither
surface keeps hidden training state.

The GUI has three setup steps:

1. Choose the supported 9B base and download its exact Hugging Face snapshot.
2. Add GitHub repositories or local paths. Optionally review accepted Git
   changes as examples, or add demonstration JSONL and executable task packs.
3. Select **Prepare training**. AutoTrainer validates and compiles the inputs,
   selects the useful training path, checks the machine, and enables **Start
   training** only when that exact project is ready.

QLoRA is the one-GPU parameter and memory strategy. It does not require every
project to run both learning stages. Accepted examples select **teach** (SFT),
verifier-backed tasks select **practice** (GRPO), and projects with both select
**both** (SFT followed by GRPO on the same adapter).

The repository includes an authoring example in [`examples/frontend-expert`](../../examples/frontend-expert). It contains miniature training and evaluation fixtures in separate directories for exercising the contracts, but both starting projects belong to this same Git repository and therefore do not establish repository holdout. Use multiple genuinely independent held-out project families and pin the real runners before collecting evidence.

## Requirements

- Linux, or Ubuntu under WSL2 on Windows.
- Python 3.11.
- One NVIDIA GPU. The initial recipe targets a 24 GB card, but memory use depends on model architecture and sequence length.
- A CUDA-capable PyTorch installation.
- Docker or Podman when the selected path includes verifier-backed practice or
  local evaluation.
- Node.js 22 or newer for the included frontend task fixtures.
- Enough local storage for the base model, Hugging Face cache, source snapshots, rollouts, and adapters.

Native Windows is suitable for editing and the web UI, but CUDA training and
any selected container sandbox should run in Linux. See [Windows and
WSL2](#windows-and-wsl2).

## Install from a clone

From Ubuntu or another Linux environment:

```bash
git clone https://github.com/Okohedeki/AutoTrainer.git
cd AutoTrainer

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./services/trainer
```

Install a CUDA build of PyTorch appropriate for the installed NVIDIA driver using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the pinned training extra. Installing PyTorch first prevents a generic wheel from being selected for you:

```bash
python -m pip install -e './services/trainer[training]'
```

Verify the GPU before attempting to download a model:

```bash
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
docker run --rm --network none hello-world
```

The training reference matrix checked on 2026-07-14 is:

| Package | Reference version |
|---|---:|
| `autotrainer-trainer` | `0.1.0` |
| [PyTorch](https://pypi.org/project/torch/) | `2.13.0` |
| [Transformers](https://pypi.org/project/transformers/) | `5.13.1` |
| [TRL](https://pypi.org/project/trl/) | `1.8.0` |
| [PEFT](https://pypi.org/project/peft/) | `0.19.1` |
| [Accelerate](https://pypi.org/project/accelerate/) | `1.14.0` |
| [Datasets](https://pypi.org/project/datasets/) | `5.0.0` |
| [bitsandbytes](https://pypi.org/project/bitsandbytes/) | `0.49.2` |
| [jmespath](https://pypi.org/project/jmespath/) | `1.1.0` |

The project dependency declaration remains the authority for supported ranges. Record the installed versions in every run; do not assume a newer combination is compatible merely because each package installs independently.

## Run the bundled example

Start the loopback backend from the repository root, then start the web app in a
second terminal:

```bash
autotrainer serve --config examples/frontend-expert/autotrainer.yaml
npm run dev
```

Follow the walkthrough: choose and download the model, add work, then prepare.
The example contains small authoring fixtures, not a completed 9B result. Do not
copy only `examples/frontend-expert`; its local source paths point into this
checkout. Generated state remains under its ignored `.autotrainer` directory.

Before a real run:

1. Use work you have the right to train on and review each source license.
2. Add accepted examples, executable tasks, or both. Raw repository code is
   reference material; it is not silently treated as training data.
3. Replace the evaluation fixture with multiple genuinely independent held-out
   project families before making an improvement claim.

V1 training supports the text-only `Qwen/Qwen3.5-9B` profile. A separately
declared evaluation reference runs through its own pinned external runner; it
does not expand trainer compatibility.

## The equivalent agent path

The CLI discovers `autotrainer.yaml` in the current directory. These commands
perform the same model, source, optional history, prepare, and start actions as
the GUI:

```bash
autotrainer models list
autotrainer model use qwen3.5-9b-text --config autotrainer.yaml
autotrainer model download --config autotrainer.yaml

autotrainer source add https://github.com/OWNER/REPOSITORY --config autotrainer.yaml
# Local repositories, accepted-example JSONL, and task-pack paths use the same command.
autotrainer source list --config autotrainer.yaml

# Optional: approve a small accepted Git change as a supervised example.
autotrainer history list --config autotrainer.yaml
autotrainer history review CANDIDATE_ID --approve \
  --instruction "Describe the accepted change" --rights-confirmed \
  --config autotrainer.yaml

autotrainer prepare --config autotrainer.yaml
autotrainer train auto --config autotrainer.yaml
```

`model download` is required before real training. It resolves the immutable
revision, writes it to YAML, downloads the snapshot, and records
`.autotrainer/models/current.json`. Training refuses a missing or mismatched
snapshot and loads the recorded model offline. Public models need no key. For a
gated model, authenticate with Hugging Face or pass `HF_TOKEN` to the backend;
AutoTrainer does not store the token in project files.

To create a standalone project:

```bash
autotrainer init ./my-frontend-expert
cd ./my-frontend-expert
autotrainer model use qwen3.5-9b-text --config autotrainer.yaml
autotrainer model download --config autotrainer.yaml
```

GitHub sources are cloned into AutoTrainer-managed storage and pinned when they
are added. Local sources remain in place. `source scan` is read-only with
respect to both kinds of source.

## Prepare and train

```bash
autotrainer prepare --config autotrainer.yaml
autotrainer train auto --config autotrainer.yaml
```

`prepare` performs validation, source scanning, compilation, recipe resolution,
model-snapshot checks, and local runtime checks without loading model weights.
It returns one conditional path:

- **teach:** accepted examples run QLoRA SFT only;
- **practice:** executable tasks run GRPO from a fresh QLoRA policy, or from an
  explicitly selected compatible adapter;
- **both:** SFT runs first, then GRPO continues that produced adapter.

`train auto` repeats preparation and runs only the selected stages. The
stage-specific `train sft` and `train rl` commands remain diagnostics and
controlled-experiment tools; they are not the normal human workflow. GRPO runs
rollouts through the declared network-isolated verifier environment.

See [Training](training.md) for the data formats, reward gates, and memory controls.

## Evaluate and package

The required proof has two parts:

1. Run the declared 9B reference and trained candidate through the same held-out task harness and show that the candidate earns a higher verified benchmark.
2. Run Fable with the base model and with that same trained candidate under identical orchestration, briefs, budgets, and time limits, then compare the resulting websites in a blind review.

Freeze the plan before producing any results. The command runner can execute the model suite locally; the external Fable suite uses verifier-free request export and immutable result ingestion:

```bash
autotrainer evaluate plan --write --config autotrainer.yaml
autotrainer evaluate run --suite model_benchmark --config autotrainer.yaml
autotrainer evaluate export --suite fable_ab --output ./fable-requests --config autotrainer.yaml
# Run those requests through the pinned Fable setup. Name directory-mode
# envelopes result.json or *.result.json, then ingest the results.
autotrainer evaluate ingest --suite fable_ab ./fable-results --config autotrainer.yaml
autotrainer evaluate review export --suite fable_ab --output ./blind-review --config autotrainer.yaml
# Collect reviewer JSONL choices, then import them.
autotrainer evaluate review import --suite fable_ab ./reviews.jsonl --config autotrainer.yaml
autotrainer evaluate report --config autotrainer.yaml
autotrainer package --config autotrainer.yaml
```

`benchmark` is an alias for `evaluate` and uses the same subcommands. Packaging refuses a winner unless both suite decisions are verified; `--allow-unverified` creates a clearly marked development artifact instead. The repository's placeholder runner pins, miniature task packs, and absence of real 9B/Fable results mean the included example is not yet verified. Follow the [V1 handoff plan](../V1-HANDOFF.md) to complete the proof.

## Windows and WSL2

From an elevated PowerShell prompt:

```powershell
wsl --install -d Ubuntu
wsl --update
```

Install the current NVIDIA Windows driver with WSL support. Do not install a second Windows GPU driver inside the Linux distribution. In Ubuntu, confirm `nvidia-smi` and `torch.cuda.is_available()` before proceeding.

The source checkout may remain on `H:` and appears under a path such as `/mnt/h/AutoTrainer`, but model caches, container storage, compiled datasets, rollouts, and checkpoints perform better on the WSL Linux filesystem. A practical layout is:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export AUTOTRAINER_HOME="$HOME/.local/share/autotrainer"
```

Keep user repositories wherever convenient, then reference them with Linux paths in `autotrainer.yaml`. YAML paths are resolved relative to the configuration file, not relative to the AutoTrainer source repository.

## Honest limits of the first release

- Text-only causal language models only; no image fields or multimodal training.
- One GPU, one environment at a time, and no cloud or distributed training.
- A repository is not automatically an SFT dataset or RL environment.
- Compilation does not guarantee that arbitrary history contains a useful, non-leaking instruction or verifier.
- Dependencies must be fetched before a network-disabled episode can run.
- Hidden verifier isolation depends on Docker or Podman and must not be replaced by running untrusted repository commands directly on the host.
- Evaluation orchestration and local package assembly are implemented, but AutoTrainer does not provide or host the model-agent/Fable runtimes and does not serve the packaged adapter.
- This checkout has no completed real 9B training result on any conditional
  path, production-sized held-out benchmark, Fable outputs, or blind reviews;
  both report decisions must be verified before calling an adapter a V1 winner.
