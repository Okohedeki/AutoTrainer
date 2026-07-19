# Data sources

AutoTrainer accepts GitHub repositories and supported local paths. A source declaration records both **what the input is** and **what it is allowed to do**.

## Source kinds

| Kind | Accepted input | Contribution |
|---|---|---|
| `repository` | GitHub repository or local Git root | code reference, accepted-history candidates, or task starting states |
| `sft_jsonl` | local `.jsonl` file | explicit text demonstrations for SFT |
| `task_pack` | local task directory or `task.json` | resettable training/evaluation tasks with hidden verification |

These kinds are not interchangeable. A final source tree does not contain a task instruction, accepted response, reset rule, or reward by itself.

## Add a source

The GUI accepts:

- `owner/repository`;
- an HTTPS or SSH GitHub URL;
- a local Git repository root;
- a local `.jsonl` file;
- a local task-pack directory or `task.json`.

The CLI uses the same inference service:

```bash
autotrainer source add OWNER/REPOSITORY \
  --mode accepted_changes \
  --config autotrainer.yaml

autotrainer source add ../local-repository \
  --mode reference_only \
  --config autotrainer.yaml

autotrainer source add ./data/accepted.jsonl --config autotrainer.yaml
autotrainer source add ./tasks/train --config autotrainer.yaml
```

An ordinary GitHub add is transactional: AutoTrainer clones it beneath the project's managed source directory, resolves the requested revision, checks out a detached commit, validates the resulting declaration, and then saves YAML. A failed clone or validation does not leave a configured source.

Local sources remain at their existing paths. A local repository must be the root of a Git working tree and have a resolvable commit. Local paths are resolved relative to `autotrainer.yaml` unless absolute.

`source materialize` remains available for an older or hand-written YAML declaration that still points at a remote repository. Normal `source add` already materializes GitHub inputs.

## Repository modes

People choose intent in product language; YAML retains the lower-level roles used by compilation:

| Mode | YAML role | What happens next |
|---|---|---|
| `accepted_changes` | `history` | review useful Git changes and supply the instruction each answered |
| `practice_tasks` | `rl_seed` | add or construct executable tasks and verifiers from repository starting states |
| `reference_only` | `style` | keep inspectable code evidence without creating training rows |
| `evaluation_holdout` | `evaluation` | reserve the repository for held-out task packs |

`accepted_changes`, `practice_tasks`, and `reference_only` may be combined. `evaluation_holdout` must be selected alone and sets `partition: evaluation`.

```bash
autotrainer source add OWNER/REPOSITORY \
  --mode accepted_changes,practice_tasks \
  --include 'src/**' \
  --include 'tests/**' \
  --exclude 'dist/**' \
  --license Apache-2.0 \
  --config autotrainer.yaml
```

Advanced agent automation can still supply `--kind`, `--partition`, and low-level `--roles`, but it must not mix `--roles` with `--mode`.

## Repository declaration

After a GitHub repository is cloned and pinned, its canonical shape is local and immutable:

```yaml
sources:
  - id: owner-storefront
    kind: repository
    uri: .autotrainer/sources/owner-storefront
    revision: 34f8c81b3b7f5b65d8e63d82abac42b66fb60f50
    partition: train
    roles: [history, rl_seed]
    include: [src/**, tests/**]
    exclude: [node_modules/**, dist/**, coverage/**]
    license:
      spdx: Apache-2.0
      attribution: https://github.com/owner/storefront
```

For a local repository, `uri` remains the supplied local path. `revision: HEAD` is acceptable while authoring, but compilation locks the resolved commit and warns about a dirty working tree. Commit or clean local changes before collecting reproducible evidence.

## Scanning

```bash
autotrainer source scan --write --config autotrainer.yaml
```

Scanning:

- resolves the exact Git root and revision;
- applies include/exclude filters and file limits;
- rejects escaping links, binaries, dependencies, and generated output;
- reports role eligibility and license/secret warnings;
- writes deterministic inventories when `--write` is used;
- does not run repository scripts, upload code, or manufacture demonstrations.

A configured source can still be blocked for its intended role. That result should be fixed, not hidden.

## Accepted changes for SFT

`accepted_changes` makes small Git changes eligible for review. It does not approve them automatically.

```bash
autotrainer history list --config autotrainer.yaml
autotrainer history review CANDIDATE_ID --approve \
  --instruction "Make the card grid readable below 768px." \
  --rights-confirmed \
  --config autotrainer.yaml
```

The instruction must describe the real task without revealing the patch. AutoTrainer rejects stale approvals and obvious answer leakage. Only reviewed, rights-confirmed examples reach SFT compilation.

The Data screen can also create a supervised example directly when the
accepted response is already known. It stores the example in a managed JSONL
source and binds it to the selected repository's locked identity and revision.
Agents use the same contract through the CLI:

```bash
autotrainer example create \
  --source owner-storefront \
  --instruction "Make the card grid readable below 768px without changing desktop behavior." \
  --accepted-response "Updated the base grid to one column and retained the existing desktop breakpoint, then ran the project checks." \
  --rights-confirmed \
  --config autotrainer.yaml
```

The source must be a pinned training repository; an evaluation holdout cannot
supply SFT examples. `autotrainer example list` shows managed examples and
`autotrainer example remove EXAMPLE_ID` removes one. Removing the final example
also removes the empty managed JSONL declaration.

## SFT JSONL

The direct format is one text-only conversation per line:

```json
{"messages":[{"role":"system","content":"You are a focused frontend engineer."},{"role":"user","content":"Make the card grid responsive without changing desktop behavior."},{"role":"assistant","content":"I will adjust the base grid and preserve the existing desktop breakpoint, then run the checks."}],"source_id":"accepted-change-42"}
```

Conversational `prompt` plus `completion` is also accepted. Empty targets, arbitrary raw text, and image fields are rejected. Keep provenance and training rights with every authored dataset.

## Task packs

A task pack supplies the pieces a repository alone lacks:

- a task instruction and stable ID;
- a declared repository source and starting revision;
- an isolated working directory;
- bounded tools and execution limits;
- trusted build/regression checks;
- a hidden verifier and structured reward report;
- a reset mechanism for every rollout.

The Data screen has a four-step task form. Agents use the same service through
the CLI:

```bash
autotrainer task create \
  --source owner-storefront \
  --instruction "Keep the checkout action visible at narrow widths without changing desktop behavior." \
  --working-directory . \
  --build "npm run build" \
  --tests "npm test" \
  --verifier-bundle ../private-verifiers/checkout \
  --verifier-command "node /autotrainer-verifier/verify.mjs" \
  --config autotrainer.yaml
```

The repository must already be pinned and configured for `practice_tasks` or
`evaluation_holdout`. The verifier directory must already exist outside the
editable repository. AutoTrainer writes the task manifest and declares its
managed task pack, but reports the task as `declared` until
`autotrainer prepare` executes the install, build, tests, and verifier canary.
Use the Data screen's Remove action or
`autotrainer task remove TASK_ID --split train` to discard an authored task;
removing the final task also removes its empty managed pack declaration.

Minimal shape:

```json
{
  "version": "1.0",
  "task": {
    "id": "responsive-pricing-001",
    "instruction": "Make the pricing grid readable below 768px.",
    "sourceId": "owner-storefront",
    "startingRevision": "locked",
    "split": "train",
    "groupId": "storefront-family"
  },
  "runtime": {
    "workingDirectory": ".",
    "install": "",
    "build": "npm run build",
    "tests": "npm test",
    "browserTests": ""
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
    "commandTimeoutSeconds": 120,
    "episodeTimeoutSeconds": 900,
    "networkAccess": false
  }
}
```

`sourceId` must point to a declared repository. The editable workspace contains its locked snapshot; the verifier bundle stays outside that tree and is mounted read-only only for scoring.
The verifier report must contain `build_passed`, `regression_pass_rate`,
`task_pass_rate`, `responsive_pass_rate`, `design_rule_pass_rate`, and
`code_quality_pass_rate`; Prepare validates those observed values before GRPO.

## Compile and inspect

```bash
autotrainer compile --config autotrainer.yaml
autotrainer plan --write --config autotrainer.yaml
```

Compilation writes inspectable artifacts below `project.artifact_dir`:

```text
.autotrainer/
  sources.lock.json
  ingested/<source-id>.documents.jsonl
  compiled/compile-report.json
  compiled/sft/train.jsonl
  compiled/rl/train.jsonl
  compiled/rl/evaluation.jsonl
  plan.json
```

The presence of a repository never creates SFT rows. It contributes rows only through approved history examples or explicit task contracts.

## Holdout isolation

Set `evaluation_holdout` on held-out repositories and use `split: evaluation` task packs. Split by repository or project family, not random records. A real benchmark must exclude every project that contributed source code, demonstrations, task mutations, rollouts, or checkpoint feedback to training.

AutoTrainer tracks distinct evaluation `groupId` values in the Data screen and
shows progress toward the required five independent held-out groups. Reusing a
group ID adds task coverage but does not satisfy another independence group.

The bundled evaluation task is an authoring fixture inside the AutoTrainer repository. It is useful for contract tests but is not independent evidence.

## Licenses and secrets

Public access does not imply permission to train or redistribute. Record SPDX and attribution data, preserve upstream notices, and review model/dataset terms separately.

Do not commit tokens, private source snapshots, model weights, or rollouts containing secrets. Scanning provides warnings, not a complete privacy or license review.
