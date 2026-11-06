"""Flower options for the `flower` service (monitoring profile).

Flower runs `celery -A src.tasks.app flower --conf=/app/scripts/flower/flowerconfig.py` and
executes this file, taking its module globals as options
(https://flower.readthedocs.io/en/latest/config.html). Flower documents `--basic-auth` and
`FLOWER_BASIC_AUTH`, which would put the credential in argv or the environment; reading the
mounted secret here keeps it in the file only (review R-30). The installer writes the file as
one `user:password` line (scripts/lib/secrets_control_plane.sh). A missing, empty or malformed
file refuses the start instead of serving Flower without authentication.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _read_basic_auth(path: Path) -> list[str]:
    try:
        value = path.read_text(encoding='utf-8').strip()
    except OSError as exc:
        sys.exit(f'flower auth: cannot read {path}: {exc.strerror}')
    user, _, password = value.partition(':')
    if not user or not password:
        sys.exit(f'flower auth: {path} must hold one user:password line')
    return [value]


basic_auth = _read_basic_auth(
    Path(os.environ.get('FLOWER_AUTH_FILE', '/run/secrets/op_flower/auth'))
)
