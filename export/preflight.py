"""Image self-check run once, right after pull, before any export step.

Installer plan (docs/design/openprocessor_internal/one_line_installer_plan.md
section 3.2): a stale ``:latest`` API image once shipped without
``perception_models`` importable. This script is invoked as

    docker compose run --rm --no-deps -T yolo-api python -m export.preflight

and exits non-zero with a specific message per failing check, instead of
failing deep inside the first export step with a confusing traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path


MODEL_REPO_SEED = Path('/opt/openprocessor/model_repo_seed')


def check_core_import() -> str | None:
    try:
        import core
    except ImportError as exc:
        return f'import core (perception_models) failed: {exc}'
    return None


def check_tensorrt_import() -> str | None:
    try:
        import tensorrt
    except ImportError as exc:
        return f'import tensorrt failed: {exc}'
    return None


def check_cuda_available() -> str | None:
    try:
        import torch
    except ImportError as exc:
        return f'import torch failed: {exc}'
    if not torch.cuda.is_available():
        return 'torch.cuda.is_available() is False'
    return None


def check_model_repo_seed() -> str | None:
    if not MODEL_REPO_SEED.is_dir():
        return f'{MODEL_REPO_SEED} does not exist'
    return None


CHECKS = (
    ('core import', check_core_import),
    ('tensorrt import', check_tensorrt_import),
    ('cuda available', check_cuda_available),
    ('model repo seed', check_model_repo_seed),
)


def main() -> int:
    failures: list[str] = []
    for label, check in CHECKS:
        error = check()
        if error:
            failures.append(f'[{label}] {error}')
        else:
            print(f'OK: {label}')

    if failures:
        print('PREFLIGHT FAILED:', file=sys.stderr)
        for f in failures:
            print(f'  - {f}', file=sys.stderr)
        return 1

    print('PREFLIGHT OK: image is ready for model export')
    return 0


if __name__ == '__main__':
    sys.exit(main())
