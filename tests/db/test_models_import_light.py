"""``src.db.models`` is importable without the web, search, GPU or queue stacks.

The trainer, export and evaluator images import ``src/db`` and carry none of those packages.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.db

REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ('fastapi', 'opensearchpy', 'torch', 'celery')

_PROBE = """
import json, sys
sys.path.insert(0, {root!r})
import src.db.models
from src.db.base import Base
print(json.dumps({{
    'loaded': sorted(m for m in sys.modules if m.split('.')[0] in {forbidden!r}),
    'tables': len(Base.metadata.tables),
}}))
"""


def test_importing_the_models_loads_none_of_the_heavy_stacks() -> None:
    out = subprocess.run(  # fixed argv, our own interpreter
        [sys.executable, '-c', _PROBE.format(root=str(REPO_ROOT), forbidden=FORBIDDEN)],
        capture_output=True,
        text=True,
        check=True,
        cwd='/',
    ).stdout
    report = json.loads(out.strip().splitlines()[-1])
    assert report['loaded'] == []
    assert report['tables'] > 0
