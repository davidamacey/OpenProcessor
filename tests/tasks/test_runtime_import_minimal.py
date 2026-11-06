"""``src.tasks.runtime`` is a leaf layer: the GPU images carry it without the API's dependencies.

A fresh interpreter imports every runtime module and none of FastAPI, the OpenSearch client or
torch (nor any ``src.services`` module) may be loaded (review R-2). Starlette is allowed: the
request-id library of the log context imports it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

_PROBE = """
import importlib, pkgutil, sys
sys.path.insert(0, '.')
import src.tasks.runtime as runtime
for info in pkgutil.iter_modules(runtime.__path__):
    importlib.import_module(f'src.tasks.runtime.{info.name}')
banned = [m for m in sys.modules if m.split('.')[0] in ('fastapi', 'opensearchpy', 'torch', 'tritonclient', 'cuml')]
services = [m for m in sys.modules if m == 'src.services' or m.startswith('src.services.')]
print(','.join(sorted(set(banned))) + '|' + ','.join(sorted(services)))
"""


def test_importing_the_runtime_loads_no_api_dependency() -> None:
    result = subprocess.run(
        [sys.executable, '-I', '-c', _PROBE],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert result.stdout.strip() == '|'
