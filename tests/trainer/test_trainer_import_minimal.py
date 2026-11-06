"""The trainer image carries the job runtime and the database layer, not the web stack.

A fresh interpreter, started the way the image starts it (``/app`` holds the trainer's flat
modules and ``src`` side by side), imports ``celery_tasks`` and none of FastAPI, the OpenSearch
client, the services or the routers may be loaded (review R-2). The image also gets the C build
of psycopg like the api image (review R-27); the Dockerfile test reads and parses the Dockerfile
(no image is built), and the interpreter check below runs inside the image.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / 'docker' / 'trainer'

_PROBE = """
import sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import celery_tasks
banned = ('fastapi', 'starlette', 'opensearchpy', 'tritonclient', 'torch', 'ultralytics')
loaded = sorted({m.split('.')[0] for m in sys.modules if m.split('.')[0] in banned})
layers = sorted(m for m in sys.modules if m.split('.')[:2] in (['src', 'services'], ['src', 'routers']))
print(','.join(loaded) + '|' + ','.join(layers))
"""


def test_importing_the_handler_module_loads_no_web_or_opensearch_code() -> None:
    result = subprocess.run(
        [sys.executable, '-I', '-c', _PROBE, str(REPO_ROOT), str(TRAINER)],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert result.stdout.strip() == '|'


def _stages() -> list[str]:
    text = (TRAINER / 'Dockerfile').read_text(encoding='utf-8')
    return re.split(r'(?m)^FROM ', text)[1:]


def test_the_dockerfile_builds_psycopg_c_and_runs_it_on_libpq5() -> None:
    builder, runtime = _stages()[0], _stages()[1]
    assert 'libpq-dev' in builder  # the libpq headers psycopg compiles against
    assert 'gcc' in builder
    assert re.search(r'\blibpq5\b', runtime)
    requirements = (TRAINER / 'requirements.txt').read_text(encoding='utf-8')
    assert re.search(r'(?m)^psycopg\[c\]==\d', requirements)
    assert 'psycopg[binary]' not in requirements


def test_the_runtime_stage_copies_the_runtime_layers_and_not_the_web_ones() -> None:
    runtime = _stages()[1]
    copied = re.findall(r'(?m)^COPY\s+(?:--\S+\s+)*(\S+)', runtime)
    for needed in ('src/tasks/runtime/', 'src/db/', 'src/config/', 'src/domain/', 'src/utils/'):
        assert needed in copied
    web = [c for c in copied if c.startswith(('src/routers', 'src/services/', 'src/app'))]
    assert web == ['src/services/training/augmentation_presets.py']  # the shared preset catalog


@pytest.mark.skipif(
    os.environ.get('OP_TRAINER_IMAGE_CHECK') != '1',
    reason='runs inside the built trainer image (set OP_TRAINER_IMAGE_CHECK=1 there)',
)
def test_inside_the_image_psycopg_is_the_c_build() -> None:
    result = subprocess.run(
        [sys.executable, '-I', '-c', 'import psycopg; print(psycopg.pq.__impl__)'],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == 'c'
