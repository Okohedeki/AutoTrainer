"""Short-lived native dependency probe used by :mod:`autotrainer.doctor`."""

from __future__ import annotations

import json

from .doctor import REFERENCE_PACKAGES, _package_check, _torch_gpu_check


def main() -> int:
    result = {
        "packages": [
            _package_check(name, version)
            for name, version in REFERENCE_PACKAGES.items()
        ],
        "gpu": _torch_gpu_check(),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
