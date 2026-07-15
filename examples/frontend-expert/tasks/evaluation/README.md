# Supply a held-out task pack

This directory is intentionally empty of task manifests. Replace it with tasks backed by a separate repository or project family before evaluation.

Reusing the training fixture here would make the example look complete while invalidating the benchmark through train/evaluation leakage. `autotrainer plan` and `autotrainer doctor` should therefore report evaluation as blocked until a real held-out repository source and task pack are declared.
