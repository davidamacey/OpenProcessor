"""POST /models/upload and POST /models/{name}/export start a ``model_export`` job.

The routes run against the database; only the checkpoint validator, the worker probe and the
weights directory are replaced. A refused start (409) leaves the stored weights untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from fixtures.job_rows import sql

from src.routers import models_export
from src.services import model_export, model_export_jobs


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = [
    pytest.mark.db,
    pytest.mark.usefixtures('bound_async_engine', 'api_async_engine', 'job_env', 'wakes'),
]

INFO = {'num_classes': 2, 'class_names': ['car', 'bus'], 'head_kind': 'nms'}


@pytest.fixture
def weights(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    directory = tmp_path / 'pytorch_models'
    directory.mkdir()
    monkeypatch.setattr(model_export, 'PYTORCH_MODELS_DIR', directory)
    monkeypatch.setattr(models_export, 'PYTORCH_MODELS_DIR', directory)
    monkeypatch.setattr(model_export_jobs, '_consumer_listens', lambda: True)
    valid = lambda _path: (True, '', dict(INFO))  # noqa: E731
    monkeypatch.setattr(model_export, 'validate_pytorch_model', valid)
    monkeypatch.setattr(models_export, 'validate_pytorch_model', valid)
    return directory


@pytest.fixture
def client() -> TestClient:
    from src.app.factory import create_app

    return TestClient(create_app(), raise_server_exceptions=False)


def _pt(content: bytes) -> dict[str, tuple[str, bytes, str]]:
    return {'file': ('car.pt', content, 'application/octet-stream')}


def test_upload_saves_the_weights_and_starts_the_job(client, weights, pg, producer) -> None:
    resp = client.post('/models/upload', files=_pt(b'a' * 2000), data={'triton_name': 'uproute'})
    assert resp.status_code == 200, resp.text
    task_id = resp.json()['task_id']
    assert (weights / 'uproute.pt').read_bytes() == b'a' * 2000
    assert producer.sent() == [task_id]
    [(kind, queue, scope)] = sql(pg, 'SELECT kind, queue, scope FROM jobs WHERE id = %s', task_id)
    assert (kind, queue, scope) == ('model_export', 'gpu_export', 'uproute')


def test_export_route_starts_the_job_for_stored_weights(client, weights, pg, producer) -> None:
    (weights / 'exproute.pt').write_bytes(b'w' * 2000)
    resp = client.post('/models/exproute/export', json={'formats': ['onnx_end2end']})
    assert resp.status_code == 200, resp.text
    task_id = resp.json()['task_id']
    assert producer.sent() == [task_id]
    [(kind, scope)] = sql(pg, 'SELECT kind, scope FROM jobs WHERE id = %s', task_id)
    assert (kind, scope) == ('model_export', 'exproute')
    assert sql(pg, "SELECT count(*) FROM models WHERE name = 'exproute'") == [(1,)]


def test_a_refused_upload_does_not_replace_the_weights_a_running_build_reads(
    client, weights, pg, producer, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = client.post('/models/upload', files=_pt(b'a' * 2000), data={'triton_name': 'racename'})
    assert first.status_code == 200, first.text

    async def passes(_session, _name) -> None:
        """The pre-check ran before the first start committed: only the start can refuse."""

    monkeypatch.setattr(models_export, 'ensure_not_building', passes)
    second = client.post('/models/upload', files=_pt(b'b' * 2000), data={'triton_name': 'racename'})
    assert second.status_code == 409
    assert second.json()['detail']['error'] == 'job_busy'
    assert (weights / 'racename.pt').read_bytes() == b'a' * 2000
    assert sorted(p.name for p in weights.iterdir()) == ['racename.pt'], 'no staged file is left'
    assert producer.sent() == [first.json()['task_id']]
