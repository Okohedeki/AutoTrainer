# Contributing to AutoTrainer

AutoTrainer welcomes focused contributions that improve reproducibility, single-GPU accessibility, frontend environments, deterministic verification, or model evaluation.

## Development setup

1. Install Node 22 or newer and Python 3.11 or newer.
2. Run `python -m pip install -e ./services/trainer` for the lightweight CLI.
3. Run `npm install` from the repository root.
4. Run `npm run dev` for the local control-plane preview.
5. Run `npm test` and `python -m unittest discover -s services/trainer/tests -v` before opening a pull request.

Training dependencies are intentionally optional. Configuration, source compilation, recipe dry runs, environment contracts, and their tests must remain runnable without CUDA or a downloaded model.

## Pull requests

Keep changes small and explain their effect on reproducibility. New rewards need tests, an auditable raw signal, and a clear account of likely reward-hacking behavior. New model integrations need an exact upstream model revision and a documented single-GPU memory profile.

Do not commit model weights, proprietary repositories, secrets, generated rollouts, or datasets without redistribution rights.
