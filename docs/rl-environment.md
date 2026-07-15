# Frontend RL environment v1.0

An environment is a reproducible starting workspace plus controlled tools, hidden verification, limits, and a reset mechanism. A repository alone is not an RL environment.

## Episode lifecycle

1. Materialize an isolated copy of the locked starting revision.
2. Keep the hidden verifier outside the editable workspace.
3. Disable external network access.
4. Give the policy the task instruction and bounded `list_files`, `read_file`, `search_code`, `apply_patch`, and named `run_check` tools.
5. Enforce token, tool-call, command-time, and wall-clock limits.
6. Run the build and regression gates outside policy control.
7. Mount the hidden verifier read-only, calculate reward, and persist each raw signal.
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

The policy must not be able to read hidden tests, change the verifier, reuse another rollout's workspace, or reach the public network. It never receives a general terminal tool. Model patches are path-checked and Git-checked before application. Only commands authored in the trusted task manifest can run, and they execute in a disposable container with no network, dropped Linux capabilities, process/memory/CPU limits, and bounded output. The verifier bundle is mounted read-only only while reward is calculated.

The reference image is built from the pinned Playwright image in `infra/frontend-runtime/Dockerfile`. The example dependency tree is baked into that image so episode setup does not require network access. Other repository lockfiles should use a derived image or an explicit, separately audited dependency-materialization step.
