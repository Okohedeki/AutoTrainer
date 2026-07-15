# Getting started

AutoTrainer is configured from one file: `autotrainer.yaml`. That file declares the base model, the evidence and executable tasks that define the specialization, the QLoRA and GRPO recipes, and the evaluation and package contract. The command line and dashboard must both read the same file; the dashboard is not a second source of truth.

The intended local flow is:

```text
configure model and sources
        ↓
scan and lock inputs
        ↓
compile demonstrations and task packs
        ↓
plan and run machine checks
        ↓
QLoRA supervised warm start
        ↓
GRPO in executable frontend environments
        ↓
Base vs SFT vs GRPO evaluation
        ↓
Fable base-vs-winner A/B
```

The repository includes an authoring example in [`examples/frontend-expert`](../../examples/frontend-expert). It contains one resolvable training fixture and deliberately leaves evaluation blocked until you supply a different held-out repository family.

## Requirements

- Linux, or Ubuntu under WSL2 on Windows.
- Python 3.11.
- One NVIDIA GPU. The initial recipe targets a 24 GB card, but memory use depends on model architecture and sequence length.
- A CUDA-capable PyTorch installation.
- Docker or Podman for isolated frontend episodes.
- Node.js 22 or newer for the included frontend fixtures.
- Enough local storage for the base model, Hugging Face cache, source snapshots, rollouts, and adapters.

Native Windows is suitable for editing and the web UI, but the CUDA training process and container sandbox should run in Linux. See [Windows and WSL2](#windows-and-wsl2).

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

## Start from the example

Copy the example outside the AutoTrainer source checkout so experiment state stays separate from product source:

```bash
cp -R examples/frontend-expert ~/polished-frontend-9b
cd ~/polished-frontend-9b
```

Open `autotrainer.yaml` and make these changes before running training:

1. Review the example's pinned `Qwen/Qwen3.5-9B` revision and update it intentionally with `autotrainer lock` when needed.
2. Replace or add repository sources whose frontend work you have the right to use.
3. Replace the `sft_jsonl` source with your accepted, text-only demonstrations. `compile` writes the canonical file named by `sft.dataset`.
4. Replace or extend the training task-pack source. `compile` writes the canonical prompt file named by `grpo.dataset`.
5. Review every source license declaration.

`Qwen/Qwen3.5-9B` is the example model, not a hidden global default. `model.id` is the declaration that selects the base model. You may replace it with another supported text-only causal language model, but model architecture, license, chat template, and single-GPU compatibility must be verified separately.

## Configure and inspect

The CLI discovers `autotrainer.yaml` in the current directory. An explicit path is useful in scripts:

```bash
autotrainer validate --config autotrainer.yaml
autotrainer models list
autotrainer model show --config autotrainer.yaml
autotrainer source list --config autotrainer.yaml
autotrainer source scan --config autotrainer.yaml
```

To create a new project instead of copying the example:

```bash
autotrainer init ./my-frontend-expert
cd ./my-frontend-expert
autotrainer model use Qwen/Qwen3.5-9B \
  --revision REPLACE_WITH_IMMUTABLE_HUGGING_FACE_COMMIT_SHA \
  --config autotrainer.yaml
```

`source scan` is read-only with respect to source repositories. It resolves paths and revisions, applies include/exclude rules, checks file eligibility, and reports what each source can contribute. A repository containing attractive final code can be useful as style evidence without being usable as supervised or reinforcement-learning data.

## Compile and plan

```bash
autotrainer compile --config autotrainer.yaml
autotrainer plan --config autotrainer.yaml
autotrainer doctor --config autotrainer.yaml
```

Compilation locks local source revisions, validates supplied datasets and task packs, and writes canonical trainer inputs under `.autotrainer/compiled`. Run `autotrainer lock --config autotrainer.yaml` separately to resolve the Hugging Face revision. Generated state belongs under `project.artifact_dir`, which is `./.autotrainer` in the example.

`plan` checks declared sources, task links, and held-out separation. `doctor` checks Python, CUDA/GPU visibility, the pinned package matrix, and the container runtime. The stage dry runs perform the final dataset and adapter checks.

## Train

Always warm-start the adapter with supervised tuning before GRPO:

```bash
autotrainer train sft --config autotrainer.yaml
```

After SFT completes, set `grpo.sft_adapter` to the directory containing the chosen SFT adapter. It must contain PEFT adapter configuration and weights. GRPO continues training that adapter; it must not silently create a fresh adapter.

```bash
autotrainer validate --config autotrainer.yaml
autotrainer doctor --config autotrainer.yaml
autotrainer train rl --config autotrainer.yaml
```

The RL command uses the `grpo` section even though the public command is named `train rl`. AutoTrainer generates rollouts, runs them in network-isolated task environments, calculates rewards, and updates the adapter sequentially so only one training GPU is required.

See [Training](training.md) for the data formats, reward gates, and memory controls.

## Evaluate and package

The required proof has two parts:

1. Run Base, SFT, and GRPO through the same held-out task harness and show that the tuned candidate earns a higher verified benchmark.
2. Run Fable with the base model and with the winning adapter under identical orchestration, briefs, budgets, and time limits, then compare the resulting websites in a blind review.

The CLI contracts for that final milestone are:

```bash
autotrainer benchmark base --config autotrainer.yaml
autotrainer evaluate --config autotrainer.yaml
autotrainer package --config autotrainer.yaml
```

These three commands are the required next milestone and are not implemented in the current `0.1.0` trainer. Do not interpret a completed SFT or GRPO job as a verified model until held-out evaluation and candidate selection exist.

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
- Benchmark, winner selection, Fable A/B, serving, and package export remain the next milestone in `0.1.0`.
