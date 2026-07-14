# AutoTrainer architecture

AutoTrainer is a local-first foundry for producing a verified frontend expert from a 9B model on one GPU.

## Product proof

Each experiment must answer two questions:

1. Does the RL adapter beat the base and supervised adapter on held-out frontend environments?
2. Does Fable using the winning adapter produce better websites than Fable using the base model under identical orchestration?

## Execution pipeline

1. Register one compatible 9B checkpoint.
2. Compile frontend examples into train and evaluation tasks.
3. Benchmark the untouched checkpoint.
4. Train a QLoRA adapter from supervised examples.
5. Continue updating that adapter with GRPO in isolated environments.
6. Evaluate Base, QLoRA, and RL checkpoints on held-out project families.
7. Run the winning candidate through the Fable A/B suite.

## Runtime boundary

The web control plane is an interface over durable experiment records. GPU work and task execution will run in WSL2/Linux. Rollout generation and gradient updates are sequential so inference and training do not have to remain resident together on the 24 GB GPU.

The environment owns repository reset, tool limits, hidden verification, reward calculation, and cleanup. Training libraries remain replaceable adapters around this contract.

## First implementation slice

The current slice establishes:

- The model-first experiment interface.
- A versioned frontend task manifest.
- Deterministic reward calculation with hard build and regression gates.
- The Base → QLoRA → RL → Fable evaluation structure.

It deliberately does not download a model, install WSL, or start a training job.
