# AutoTrainer architecture

AutoTrainer is a local-first foundry for producing a verified frontend expert from a 9B model on one GPU.

## Product proof

Each experiment must answer two questions:

1. Does the SFT-plus-GRPO candidate beat the declared 9B reference on held-out frontend environments?
2. Does Fable using that same trained candidate produce better websites than Fable using the base 9B under identical orchestration?

## Source of truth

`autotrainer.yaml` declares the model, immutable revision, evidence, demonstrations, executable task packs, one-GPU recipes, sandbox, evaluation split, and package contract. The CLI resolves it into source locks, compiled JSONL, run recipes, checkpoints, reports, and adapter packages. The web app is currently a read-only preview; the intended GUI will edit and run those same contracts through the shared local backend while the CLI remains an equivalent reproducibility interface. It is not a second orchestration system and must not create hidden GUI-only state.

Repository code, SFT demonstrations, and RL task packs are deliberately different source kinds. Static code can establish vocabulary and style, but it is not supervised data without an instruction and accepted response, and it is not RL data without a resettable starting state and hidden verifier.

## Execution pipeline

1. Register the supported `Qwen/Qwen3.5-9B` project checkpoint and immutable revision.
2. Scan and lock repository evidence, explicit SFT JSONL, and executable task packs.
3. Compile direct demonstrations and task manifests into inspectable trainer JSONL.
4. Declare the immutable 9B reference used as the benchmark quality bar.
5. Train a QLoRA adapter from supervised examples.
6. Reload that same adapter as trainable and continue it with GRPO in isolated environments.
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
- Winner-gated LoRA adapter packages containing provenance, reports, licenses, recipes, and payload hashes.

The CLI does not install WSL, Docker, CUDA, accept model licenses, or supply the external model-agent and Fable runtimes. Those runners must be pinned by the operator, and this checkout has not completed a full 9B training run, a statistically useful held-out benchmark, or blind Fable review. Until both evaluation decisions report `verified_better`, a trained checkpoint remains an experiment artifact rather than a verified winner. See the [V1 handoff plan](V1-HANDOFF.md) for the remaining proof work.
