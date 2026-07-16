# AutoTrainer architecture

AutoTrainer is a local-first foundry for producing a verified frontend expert from a 9B model on one GPU.

## Product proof

Each experiment must answer two questions:

1. Does the trained candidate beat the declared 9B reference on held-out frontend environments?
2. Does Fable using that same trained candidate produce better websites than Fable using the base 9B under identical orchestration?

## Source of truth

`autotrainer.yaml` declares the model, immutable revision, evidence, demonstrations, executable task packs, one-GPU recipes, sandbox, evaluation split, and package contract. The CLI resolves it into source locks, compiled JSONL, run recipes, checkpoints, reports, and adapter packages. The web app performs model selection, source setup, reviewed-history approval, preparation, training launch, evaluation planning, and command-backed benchmark launch through a loopback-only local API; that API and the CLI call the same Python service functions. The GUI is not a second orchestration system and does not create hidden GUI-only training or evaluation state.

## Human and agent interfaces

- Humans use the GUI to choose the base, add work, review accepted changes, prepare and run training, then freeze and watch held-out evaluation.
- Agents use `autotrainer model`, `source`, `history`, `prepare`, `train auto`, and `evaluate` for the equivalent shared-service operations.
- Both paths mutate the same YAML, call `model_service`, and receive the same cache states.
- `autotrainer serve` binds only to loopback and exposes a versioned `/api/v1` contract; it does not accept arbitrary shell commands.
- Hugging Face credentials remain process environment input. They are never returned by the API or written to YAML and receipts.

Repository code, SFT demonstrations, and RL task packs are deliberately different source kinds. Static code can establish vocabulary and style, but it is not supervised data without an instruction and accepted response, and it is not RL data without a resettable starting state and hidden verifier.

## Execution pipeline

1. Register the supported `Qwen/Qwen3.5-9B` project checkpoint and immutable revision.
2. Scan and lock repository evidence, explicit SFT JSONL, and executable task packs.
3. Compile direct demonstrations and task manifests into inspectable trainer JSONL.
4. Declare the immutable 9B reference used as the benchmark quality bar.
5. Choose the path supported by the compiled learning signal: supervised teaching, verified practice, or both.
6. Train a QLoRA adapter with SFT, GRPO in isolated environments, or SFT followed by GRPO on the same adapter.
7. Compare the declared 9B reference with the trained candidate on held-out project families.
8. Compare Fable plus the base 9B with Fable plus that same trained candidate.

## Runtime boundary

GPU work and task execution run in WSL2/Linux. The reference profile uses native Transformers generation, two GRPO generations, no vLLM, no separate reference model (`beta: 0`), and a 2K completion limit. These defaults prioritize fitting an RTX 4090 over throughput.

The environment owns repository reset, bounded tools, hidden verification, reward calculation, and cleanup. Project code executes in a network-disabled Docker/Podman container. The model never receives an arbitrary shell.

## Implemented foundation

- A model-first YAML/CLI experiment contract.
- Static repository, SFT JSONL, and task-pack inspection with deterministic locks.
- Trainer-ready compilation without treating raw repositories as demonstrations.
- A versioned frontend task manifest and bounded container environment tools.
- Deterministic reward calculation with hard build and regression gates.
- Guarded Hugging Face QLoRA and same-adapter TRL GRPO launchers.
- Immutable paired evaluation plans, shell-free command execution, external result exchange, and local patch re-verification.
- Separate model-benchmark and Fable A/B reports, deterministic blind-review pairs, and decision thresholds.
- Durable training and evaluation job records that the console polls without inventing progress between observed stage or trial boundaries.
- Winner-gated LoRA adapter packages containing provenance, reports, licenses, recipes, and payload hashes.

The CLI does not install WSL, Docker, CUDA, accept model licenses, or supply the external model-agent and Fable runtimes. Those runners must be pinned by the operator, and this checkout has not completed a full 9B training run, a statistically useful held-out benchmark, or blind Fable review. Until both evaluation decisions report `verified_better`, a trained checkpoint remains an experiment artifact rather than a verified winner. See the [V1 handoff plan](V1-HANDOFF.md) for the remaining proof work.
