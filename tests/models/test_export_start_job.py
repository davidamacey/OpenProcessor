"""Upload and export start a ``model_export`` job: one row, one message, one job per model name.

The kind is exclusive per Triton model name (02b section 2.1), so a second start of the same name
while one is active answers 409 ``job_busy``; a missing ``gpu_export`` consumer answers 503.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fixtures.job_rows import sql

from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.services import model_export_jobs as jobs


INFO = {'num_classes': 2, 'class_names': ['car', 'bus'], 'head_kind': 'nms'}


async def start(name: str, *, new_version: bool = True):
    return await jobs.start_export(
        triton_name=name,
        pt_file=Path(f'/app/pytorch_models/{name}.pt'),
        max_batch=16,
        formats=['trt'],
        auto_load=False,
        model_info=INFO,
        new_version=new_version,
    )


@pytest.mark.db
@pytest.mark.usefixtures('bound_async_engine', 'job_env', 'wakes', 'producer')
class TestStartInTheDatabase:
    @pytest.mark.asyncio
    async def test_one_job_one_message_one_model_version(self, pg, producer) -> None:
        result = await start('onejobonemessage')
        assert result.created
        assert producer.sent() == [str(result.job.id)]
        [(kind, queue, project, scope, group)] = sql(
            pg,
            'SELECT kind, queue, project_id, scope, exclusive_group FROM jobs WHERE id = %s',
            result.job.id,
        )
        assert (kind, queue, project, scope, group) == (
            'model_export',
            'gpu_export',
            None,
            'onejobonemessage',
            'model_export',
        )
        rows = sql(pg, "SELECT version, origin, labels FROM models WHERE name = 'onejobonemessage'")
        assert rows == [(1, 'uploaded', ['car', 'bus'])]
        assert result.job.params['formats'] == ['trt']

    @pytest.mark.asyncio
    async def test_a_second_export_of_the_same_name_while_active_is_409(self, pg, producer) -> None:
        first = await start('asecondexportoft')
        with pytest.raises(RequestRejectedError) as busy:
            await start('asecondexportoft')
        assert busy.value.status_code == 409
        assert busy.value.detail['error'] == 'job_busy'
        assert busy.value.detail['active_job_id'] == str(first.job.id)
        assert len(producer.sent()) == 1
        assert sql(pg, "SELECT count(*) FROM models WHERE name = 'asecondexportoft'") == [(1,)]
        other = await start('asecondexportoft2')
        assert other.created

    @pytest.mark.asyncio
    async def test_the_route_check_refuses_while_the_name_is_being_built(self) -> None:
        async with async_session_factory()() as session:
            await jobs.ensure_not_building(session, 'theroutecheckrefu')
        await start('theroutecheckrefu')
        async with async_session_factory()() as session:
            with pytest.raises(RequestRejectedError) as busy:
                await jobs.ensure_not_building(session, 'theroutecheckrefu')
        assert busy.value.status_code == 409

    @pytest.mark.asyncio
    async def test_a_re_export_reuses_the_version_and_an_upload_adds_one(self, pg) -> None:
        first = await start('areexportreusest')
        sql(
            pg,
            "UPDATE jobs SET status = 'cancelled', finished_at = now() WHERE id = %s",
            first.job.id,
        )
        await start('areexportreusest', new_version=False)
        assert sql(pg, "SELECT version FROM models WHERE name = 'areexportreusest'") == [(1,)]
        [(job_id,)] = sql(
            pg, "SELECT id FROM jobs WHERE status = 'queued' AND scope = 'areexportreusest'"
        )
        sql(pg, "UPDATE jobs SET status = 'cancelled', finished_at = now() WHERE id = %s", job_id)
        await start('areexportreusest', new_version=True)
        assert sql(pg, "SELECT version FROM models WHERE name = 'areexportreusest' ORDER BY 1") == [
            (1,),
            (2,),
        ]

    @pytest.mark.asyncio
    async def test_the_status_reads_the_job_row(self) -> None:
        started = await start('thestatusreadsthe')
        task = await jobs.get_export_task(str(started.job.id))
        assert task is not None
        assert (task.status.value, task.triton_name, task.num_classes) == (
            'pending',
            'thestatusreadsthe',
            2,
        )
        assert task.formats_pending == ['trt']
        assert str(started.job.id) in [t.task_id for t in await jobs.list_export_tasks()]
        assert await jobs.get_export_task('not-a-uuid') is None


@pytest.fixture
def client() -> TestClient:
    from src.app.factory import create_app

    return TestClient(create_app(), raise_server_exceptions=False)


def _pt() -> dict[str, tuple[str, bytes, str]]:
    return {'file': ('car.pt', b'x' * 2000, 'application/octet-stream')}


def test_upload_without_a_gpu_export_consumer_is_a_503_naming_the_service(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from src.services import model_export

    monkeypatch.setattr(jobs, '_consumer_listens', lambda: False)
    monkeypatch.setattr(model_export, 'PYTORCH_MODELS_DIR', tmp_path / 'pytorch_models')
    resp = client.post('/models/upload', files=_pt())
    assert resp.status_code == 503
    assert 'model-export-worker' in resp.json()['detail']
    assert '--profile export' in resp.json()['detail']
    assert not (tmp_path / 'pytorch_models').exists(), 'no weights are saved without a worker'


def test_export_without_a_gpu_export_consumer_is_a_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobs, '_consumer_listens', lambda: False)
    resp = client.post('/models/car/export', json={})
    assert resp.status_code == 503
    assert 'model-export-worker' in resp.json()['detail']


def test_a_consumer_is_one_that_listens_on_the_gpu_export_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Inspect:
        def __init__(self, replies):
            self.replies = replies

        def active_queues(self):
            return self.replies

    class Control:
        def __init__(self, replies):
            self.replies = replies

        def inspect(self, timeout):
            assert timeout <= 2
            return Inspect(self.replies)

    class App:
        def __init__(self, replies):
            self.control = Control(replies)

    for replies, expected in (
        (None, False),
        ({'w1': [{'name': 'jobs'}]}, False),
        ({'w1': [{'name': 'jobs'}], 'w2': [{'name': 'gpu_export'}]}, True),
    ):
        monkeypatch.setattr(jobs.task_queue, 'producer_app', lambda r=replies: App(r))
        assert jobs._consumer_listens() is expected
