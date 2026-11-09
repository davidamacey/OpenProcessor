"""Every Celery queue has exactly the consumer service of 02b section 7.3, and no service mounts /jobs.

Docs: https://docs.celeryq.dev/en/stable/reference/cli.html#celery-worker (``-Q``),
https://docs.docker.com/reference/compose-file/services/ (``command``, ``volumes``).
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from compose._helpers import REPO_ROOT, _services
from src.config.queue import TaskQueues


# 02b section 7.3. The arbiter queue is consumed only in the render that includes the
# GPU-arbiter overlay (RC-3), so it is checked in test_arbiter_overlay.
EXPECTED = {
    TaskQueues.JOBS: 'jobs-worker',
    TaskQueues.PERIODIC: 'periodic-worker',
    TaskQueues.OPEN_VOCAB: 'open-vocab-worker',
    TaskQueues.AUTOLABEL: 'curation-auto-label-worker',
    TaskQueues.ASSIST_LOOP: 'curation-assist-loop-worker',
    TaskQueues.CLUSTER: 'cluster-worker',
    TaskQueues.GPU_TRAIN: 'curation-trainer',
    TaskQueues.GPU_EXPORT: 'model-export-worker',
    TaskQueues.GPU_EVAL: 'curation-evaluator',
}
# Images whose Celery worker starts from the image ENTRYPOINT instead of a compose command.
ENTRYPOINT_FILES = {
    'curation-trainer': 'docker/trainer/Dockerfile',
    'curation-evaluator': 'Dockerfile',
}


def _queues(name: str, service: dict[str, Any]) -> set[str]:
    command = service.get('command') or ''
    text = command if isinstance(command, str) else ' '.join(str(c) for c in command)
    if 'celery' not in text and name in ENTRYPOINT_FILES:
        dockerfile = (REPO_ROOT / ENTRYPOINT_FILES[name]).read_text()
        text = ' '.join(
            re.findall(r'ENTRYPOINT \[(.*?)\]', dockerfile)[-1].replace('"', '').split(',')
        )
    if ' worker ' not in f' {text} ':
        return set()
    match = re.search(r'(?:-Q|--queues)[= ]\s*([\w,]+)', text)
    return set(match.group(1).split(',')) if match else set()


def consumers(services: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name, service in services.items():
        for queue in _queues(name, service):
            out.setdefault(queue, set()).add(name)
    return out


def test_each_queue_has_exactly_its_section_7_3_service() -> None:
    found = consumers(_services())
    assert {q: sorted(found.get(q, ())) for q in EXPECTED} == {
        q: [svc] for q, svc in EXPECTED.items()
    }
    assert set(found) <= set(TaskQueues.ALL)


def test_no_service_mounts_the_jobs_volume() -> None:
    mounted = [
        name
        for name, service in _services().items()
        for volume in service.get('volumes') or []
        if str(volume).split(':')[1:2] == ['/jobs'] or str(volume).startswith('openprocessor-jobs')
    ]
    assert not mounted, mounted


@pytest.mark.parametrize(
    'name', ['curation-auto-label-worker', 'curation-assist-loop-worker', 'model-export-worker']
)
def test_celery_consumers_stop_gracefully_and_remap_sigterm(name: str) -> None:
    service = _services()[name]
    env = service['environment']
    assert 'REMAP_SIGTERM=SIGQUIT' in env
    assert int(re.sub(r'\D', '', service['stop_grace_period'].split(':-')[-1])) > 30
    assert 'healthcheck' in service


def test_cluster_refresh_daemon_service_is_gone() -> None:
    assert 'curation-cluster-refresh' not in _services()


def test_the_jobs_worker_reaches_the_api_for_bulk_ingest() -> None:
    assert 'OP_API_LOOPBACK_URL=http://api:8000' in _services()['jobs-worker']['environment']
