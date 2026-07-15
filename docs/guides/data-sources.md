# Data sources

AutoTrainer accepts three source kinds, and they are not interchangeable:

| Kind | What it contributes | What it does not prove |
|---|---|---|
| `repository` | Code/style evidence, accepted history candidates, and starting states for tasks | That an instruction, demonstration, hidden test, or reward exists |
| `sft_jsonl` | Explicit text demonstrations for supervised QLoRA | That the response builds or earns an RL reward |
| `task_pack` | Reproducible starting states, instructions, tools, limits, and hidden verification | That the task is suitable for SFT unless a successful trajectory is also supplied |

This distinction is central to the product. Pointing AutoTrainer at repositories whose design you like is useful, but a final source tree alone does not explain which alternatives were rejected or provide a resettable reinforcement-learning environment.

## Repository sources

Canonical shape:

```yaml
sources:
  - id: preferred-storefront
    kind: repository
    uri: /work/preferred-storefront
    revision: 34f8c81b3b7f5b65d8e63d82abac42b66fb60f50
    partition: train
    roles: [style, history, rl_seed]
    include:
      - package.json
      - src/**
      - tests/**
    exclude:
      - node_modules/**
      - dist/**
      - coverage/**
    runtime:
      preset: react-vite-tailwind
      working_directory: .
      install: npm ci
      build: npm run build
      test: npm test
      browser_test: npm run test:browser
    license:
      spdx: Apache-2.0
      attribution: https://example.com/preferred-storefront
```

`uri` may be a local Git working tree or a Git URL. The V1 scanner requires Git because it must identify and lock an exact source state. Remote URLs are declared but not cloned implicitly; clone them into a reviewed local directory and point `uri` there before compilation. A plain non-Git directory is not accepted as a repository source.

For local sources, `revision: HEAD` is acceptable while authoring. Compilation records the exact commit and must also detect dirty files. Dirty input is rejected by default because a commit SHA cannot reproduce uncommitted content. For remote sources, prefer an immutable commit from the beginning.

### What scanning does

```bash
autotrainer source scan --config autotrainer.yaml
```

Scanning:

- Resolves each Git repository and revision.
- Applies include and exclude globs.
- Rejects escaping symlinks, binaries, generated output, dependencies, and files above configured safety limits.
- Reports source-role eligibility.
- Warns about secret-prone filenames and absent license metadata.
- Does not run install, build, test, browser, or repository-provided scripts.
- Does not upload source code.

The scanner can report a repository as usable for `style` but blocked for `history` or `rl_seed`. That is a useful result, not an error to hide.

### What repository roles mean

`style` means eligible final code can be indexed as reference evidence or used to construct code-domain examples. It must not be mislabeled as instruction/response SFT data.

`history` means the compiler may examine accepted commits as candidates for demonstrations. A useful candidate still needs a non-leaking instruction, a reconstructable parent state, and a valid result. A commit message that contains the exact patch is rejected as answer leakage.

`rl_seed` means the repository can provide starting states for explicit or mechanically reconstructed tasks. It does not make every commit an RL episode. A compiled task should satisfy a red/green rule: the starting state fails its target verifier and the accepted/reference state passes.

## SFT JSONL

The most direct supervised source is one JSON object per line with a nonempty conversational `messages` list:

```json
{"messages":[{"role":"system","content":"You are a focused frontend engineer."},{"role":"user","content":"Make the card grid responsive without changing desktop behavior."},{"role":"assistant","content":"I will use a one-column base rule and restore the three-column grid at the existing desktop breakpoint, then run the checks."}],"source_id":"accepted-change-42"}
```

The included minimal dataset is [`examples/frontend-expert/data/sft-messages.jsonl`](../../examples/frontend-expert/data/sft-messages.jsonl).

The loader also accepts conversational `prompt` plus `completion`. Raw free-form text, empty assistant messages, and `image` or `images` fields are rejected in the text-only V1 path.

Good SFT records come from:

- An accepted developer change paired with its real task description.
- A successful tool trajectory whose final patch passed verification.
- A failed attempt followed by an accepted correction.
- A review comment followed by the revised implementation.
- A human-selected result with provenance and redistribution permission.

Avoid examples that merely restate a target diff in the instruction. They teach copying rather than frontend problem solving.

Declare the file separately from repository evidence:

```yaml
- id: accepted-frontend-demonstrations
  kind: sft_jsonl
  uri: ./data/sft-messages.jsonl
  partition: train
  roles: [demonstrations]
  license:
    spdx: LicenseRef-Proprietary-Internal
```

`sft.dataset` points to the exact compiled file used by the trainer, normally `./.autotrainer/compiled/sft/train.jsonl`. The source entry records the authored file and its provenance; it does not implicitly treat every scanned repository document as SFT.

## Task packs

A task pack is a directory of version `1.0` task manifests plus verifier bundles. An authoring manifest links to a declared repository source instead of inventing a snapshot name:

```json
{
  "version": "1.0",
  "task": {
    "id": "responsive-pricing-train-001",
    "instruction": "Make the pricing grid readable below 768px without changing desktop behavior.",
    "sourceId": "preferred-storefront",
    "startingRevision": "locked",
    "split": "train",
    "groupId": "preferred-storefront-family"
  },
  "runtime": {
    "workingDirectory": "examples/frontend-expert/fixture-site",
    "install": "cp -a /opt/autotrainer/frontend-deps/node_modules ./node_modules",
    "build": "npm run build",
    "tests": "npm test",
    "browserTests": "npm run test:browser"
  },
  "tools": ["list_files", "read_file", "search_code", "apply_patch", "run_check"],
  "verifier": {
    "bundle": "./verifier",
    "command": "node /autotrainer-verifier/verify.mjs",
    "reportPath": ".autotrainer-verifier-report.json"
  },
  "rewards": {
    "buildGate": true,
    "regressionGate": true,
    "regressionSafety": 0.2,
    "taskTests": 0.35,
    "responsiveRules": 0.2,
    "designRules": 0.15,
    "patchQuality": 0.1
  },
  "limits": {
    "toolCalls": 40,
    "tokenBudget": 12000,
    "commandTimeoutSeconds": 120,
    "episodeTimeoutSeconds": 900,
    "networkAccess": false
  }
}
```

`sourceId` must match a `kind: repository` entry. `startingRevision: locked` is valid in a local authoring fixture; `compile` replaces it in generated state with the source revision captured by the source lock. Published task packs should use exact revisions.

`workingDirectory` is relative to the checked-out repository root. The editable workspace contains only the repository snapshot. `verifier.bundle` resolves relative to the task manifest and is mounted separately so the policy cannot inspect or modify it.

The trusted verifier command writes JSON at `reportPath` with these keys:

```json
{
  "build_passed": true,
  "regression_pass_rate": 1.0,
  "task_pass_rate": 1.0,
  "responsive_pass_rate": 1.0,
  "design_rule_pass_rate": 1.0,
  "code_quality_pass_rate": 1.0
}
```

All rates are finite values from zero through one. Build failure or a regression rate below one gates the scalar reward to zero. Individual signals remain in run artifacts for reward-hacking analysis.

The example task pack contains real local files and uses `startingRevision: locked`; it does not use a fake `project@sha` identifier.

## Compile and inspect

```bash
autotrainer compile --config autotrainer.yaml
autotrainer plan --config autotrainer.yaml
```

The current compiler produces deterministic state below `project.artifact_dir`:

```text
.autotrainer/
├── sources.lock.json
├── ingested/
│   └── <source-id>.documents.jsonl
├── compiled/
│   ├── compile-report.json
│   ├── sft/train.jsonl
│   └── rl/train.jsonl
└── plan.json
```

The current compiler locks and validates declared sources. It does not call a cloud teacher, infer product taste from screenshots, or promise to synthesize high-quality instructions and hidden tests from arbitrary commits. Those transformations must remain inspectable and reviewable when added.

## Train/evaluation isolation

Set `partition: evaluation` on held-out repository and task-pack sources. Use `groupId` to keep related variants together. The production benchmark split is by repository or project family, not by random records.

The bundled miniature training fixture demonstrates configuration and task resolution. The evaluation directory is intentionally empty so the plan remains blocked until you add a genuinely separate repository family and task pack. A real benchmark must not use projects that contributed source code, demonstrations, mutation seeds, or RL rollouts to training.

## Licensing and sensitive data

The ability to clone a public repository does not automatically grant permission to train on it or redistribute derived artifacts. Record an SPDX expression and attribution for each source, preserve upstream notices, and review model and dataset terms separately.

Never commit credentials, private source snapshots, generated rollouts containing secrets, or model weights to the AutoTrainer repository. Scanning is not a complete secret detector. Perform an independent privacy, security, and license review before training or publishing an adapter.
