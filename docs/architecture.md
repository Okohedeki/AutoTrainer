# AutoTrainer architecture

AutoTrainer is a local-first foundry for producing a verified frontend expert from a 9B model on one GPU.

## Product proof

Each experiment must answer two questions:

1. Does the RL adapter beat the base and supervised adapter on held-out frontend environments?
2. Does Fable using the winning adapter produce better websites than Fable using the base model under identical orchestration?

## Source of truth

`autotrainer.yaml` declares the model, immutable revision, evidence, demonstrations, executable task packs, one-GPU recipes, sandbox, evaluation split, and package contract. The CLI resolves it into source locks, compiled JSONL, run recipes, checkpoints, and reports. The web app is a future viewer/editor for those same artifacts; it is not a second orchestration system.

Repository code, SFT demonstrations, and RL task packs are deliberately different source kinds. Static code can establish vocabulary and style, but it is not supervised data without an instruction and accepted response, and it is not RL data without a resettable starting state and hidden verifier.

## Execution pipeline

1. Register one compatible 9B checkpoint and immutable revision.
2. Scan and lock repository evidence, explicit SFT JSONL, and executable task packs.
3. Compile direct demonstrations and task manifests into inspectable trainer JSONL.
4. Benchmark the untouched checkpoint.
5. Train a QLoRA adapter from supervised examples.
6. Reload that same adapter as trainable and continue it with GRPO in isolated environments.
7. Evaluate Base, QLoRA, and RL checkpoints on held-out project families.
8. Run the winning candidate through the Fable A/B suite.

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
- The Base → QLoRA → RL → Fable evaluation structure.

The CLI does not install WSL, Docker, CUDA, or accept model licenses for the user. Baseline agent automation, held-out evaluation, winner packaging, and the Fable A/B runner remain to be implemented. Until those exist, a trained checkpoint is an experiment artifact, not a verified winner.
