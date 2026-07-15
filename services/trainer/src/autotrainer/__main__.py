"""Allow ``python -m autotrainer`` to behave like the console command."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
