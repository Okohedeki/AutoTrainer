# AutoTrainer

AutoTrainer is an open-source, local-first foundry for turning a 9B base model into a verified frontend expert on one consumer GPU.

> The public license will be selected before the first GitHub release. No license is implied by this development snapshot.

The first product path is:

```text
Base 9B → QLoRA warm start → reinforcement learning → held-out benchmark → Fable A/B
```

## Current state

The repository contains the first control-plane interface and the initial RL environment contracts. Model download and training are intentionally not wired yet.

## Repository layout

- `apps/web` — local React control plane.
- `services/trainer` — Python environment and training runtime.
- `schemas` — versioned cross-language task contracts.
- `examples` — redistributable example environments.
- `docs` — architecture and reward-system decisions.

## Local dashboard

```powershell
npm install
npm run dev
```

Open `http://localhost:3000`.

## Validation

```powershell
npm test
```

The test command creates a production build, verifies the rendered control plane, and exercises deterministic rollout reward calculation.

## Trainer-core tests

```powershell
py -3.11 -m unittest discover -s services/trainer/tests -v
```

The heavy CUDA training dependencies are optional and are not installed by the initial setup.

## Important files

- `apps/web/src/data.ts` — model catalog and product pipeline definitions.
- `services/trainer/src/autotrainer/environment.py` — deterministic reward calculation.
- `schemas/frontend-task.schema.json` — the versioned task contract.
- `examples/tasks/responsive-pricing/task.json` — the first example environment manifest.
- `docs/architecture.md` — system boundaries and experiment proof.
- `docs/rl-environment.md` — episode lifecycle and security rules.
