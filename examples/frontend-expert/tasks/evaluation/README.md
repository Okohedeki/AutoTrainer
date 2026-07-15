# Evaluation task authoring fixture

`accessible-newsletter/` evaluates the newsletter/editorial fixture in `../../evaluation-site`.
That source is a separate subtree but remains part of the AutoTrainer Git repository, so it
demonstrates the evaluation contract without satisfying repository holdout. Its public checks preserve content and desktop behavior while requiring
an accessible signup flow and a genuinely stacked mobile layout. The verifier bundle stays outside
the editable episode workspace and emits the five normalized reward rates expected by V1.

The manifest intentionally references repository source `held-out-newsletter-site` at revision
`locked`. Declare and lock that evaluation-only source before compiling the evaluation split; do
not reuse the training source ID or pricing-family group ID. For a real proof, replace it with a
genuinely independent repository identity; renaming a source that points to the same repository is
not a holdout.

The starting snapshot is deliberately unsolved: build and regression tests pass, while the browser
requirements identify the accessibility and narrow-layout work an evaluated model must complete.
