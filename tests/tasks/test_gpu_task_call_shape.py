"""The GPU images' tasks take the job id and nothing else, like every task in the API's worker app.

``tests/tasks/test_task_call_shape.py`` leaves ``gpu_train``, ``gpu_export`` and ``gpu_eval``
out because the API's worker app never imports those modules. ``task_queue.enqueue`` sends
``args=[job_id]`` to every queue, so each GPU image's task must bind exactly one positional job id
too. Each app is imported in a fresh interpreter, the way its image starts it (the trainer's flat
modules sit beside ``src``), so no flat module name leaks into this process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.config.queue import TaskQueues
from src.config.task_routes import ROUTES


REPO_ROOT = Path(__file__).resolve().parents[2]

_PROBE = """
import importlib, inspect, json, sys
sys.path[:0] = [sys.argv[1], *sys.argv[3:]]
app = importlib.import_module(sys.argv[2]).app
app.loader.import_default_modules()
out = {}
for name, task in app.tasks.items():
    if not name.startswith('op.'):
        continue
    sig = inspect.signature(task.run)

    def binds(*args):
        try:
            sig.bind(*args)
        except TypeError:
            return False
        return True

    out[name] = [binds('job-1'), binds(), binds('job-1', 'extra')]
print(json.dumps(out))
"""

# (module the image starts, extra sys.path entries, the one routed task it serves)
GPU_APPS = [
    pytest.param('celery_tasks', ['docker/trainer'], 'op.gpu_train.train', id='trainer'),
    pytest.param('export.export_tasks', [], 'op.export.model_export', id='export'),
    pytest.param('scripts.curation.bakeoff.bakeoff_tasks', [], 'op.gpu_eval.bakeoff', id='bakeoff'),
]
GPU_QUEUES = frozenset({TaskQueues.GPU_TRAIN, TaskQueues.GPU_EXPORT, TaskQueues.GPU_EVAL})


@pytest.mark.parametrize(('module', 'paths', 'expected'), GPU_APPS)
def test_a_gpu_apps_routed_task_accepts_exactly_one_job_id(
    module: str, paths: list[str], expected: str, tmp_path: Path
) -> None:
    password = tmp_path / 'queue_password'
    password.write_text('test-password\n')
    result = subprocess.run(
        [sys.executable, '-I', '-c', _PROBE, str(REPO_ROOT), module]
        + [str(REPO_ROOT / p) for p in paths],
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
        env={
            'OP_QUEUE_URL': 'redis://valkey:6379/0',
            'OP_QUEUE_PASSWORD_FILE': str(password),
        },
    )
    shapes: dict[str, list[bool]] = json.loads(result.stdout.strip().splitlines()[-1])
    gpu_tasks = {name: shape for name, shape in shapes.items() if ROUTES.get(name) in GPU_QUEUES}
    assert set(gpu_tasks) == {expected}, f'{module} registers {sorted(shapes)}'
    # binds the job id; refuses no argument; refuses an extra argument
    assert gpu_tasks[expected] == [True, False, False]
