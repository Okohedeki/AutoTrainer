# Frontend RL environment v0.1

An environment is a reproducible starting workspace plus controlled tools, hidden verification, limits, and a reset mechanism. A repository alone is not an RL environment.

## Episode lifecycle

1. Materialize an isolated copy of `startingSnapshot`.
2. Mount hidden tests outside the editable workspace.
3. Disable external network access.
4. Give the policy the task instruction and controlled tools.
5. Enforce token, tool-call, command-time, and wall-clock limits.
6. Run the build and regression gates outside the editable workspace.
7. Calculate and persist each reward signal separately.
8. Destroy the workspace.

## Reward contract

Build failure or any regression produces a total reward of zero. A passing rollout receives a weighted score:

- Hidden task tests: 35%
- Regression safety: 20%
- Responsive rules: 20%
- Design rules: 15%
- Patch quality: 10%

The scalar reward is used for optimization. The individual signals are retained for auditing and reward-hacking detection.

## Security boundary

The policy must not be able to read hidden tests, change the verifier, reuse another rollout's workspace, or reach the public network. Commands execute with explicit allowlists and timeouts.
