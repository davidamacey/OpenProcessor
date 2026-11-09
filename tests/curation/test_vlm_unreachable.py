"""An unreachable VLM endpoint: ``/vlm/label_batch`` answers 503 ``vlm_unreachable``
(nothing written, no crop marked failed) and the VLM worker backs off, releases the
crops it held, reports why, and resumes once the endpoint answers again."""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from _class_registry_fake import InMemoryRegistry
from PIL import Image

from curation.query_fakes import QueryFakeOpenSearch
from scripts.curation import vlm_cli, vlm_worker
from src.config.curation_accessors import base_curation_config
from src.services.curation.worker_liveness import check_heartbeat, write_heartbeat
from src.services.labeling.vlm_client import VlmIdentity
from src.services.labeling.vlm_models import VlmClassPrediction
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


if TYPE_CHECKING:
    from pathlib import Path


ITEMS = base_curation_config().items_index


def _item(crop_id: str) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'image_path': f'/data/{crop_id}.jpg',
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        'class_source': 'item_proposal',
        'class_validated': False,
    }


async def _call_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failures: dict[str, str | None]
) -> tuple[Any, QueryFakeOpenSearch]:
    """Run the route; ``failures[crop]`` is ``'request_failed'`` or ``None`` (answered)."""
    import src.services.curation.vlm_batch as vlm_mod
    from src.services.curation.vlm_batch import run_label_batch

    buf = io.BytesIO()
    Image.new('RGB', (32, 32), (1, 2, 3)).save(buf, format='JPEG')
    for cid in failures:
        (tmp_path / 'default').mkdir(exist_ok=True)
        (tmp_path / 'default' / f'{cid}.jpg').write_bytes(buf.getvalue())
    monkeypatch.setattr(
        vlm_mod, 'get_curation_config', lambda: SimpleNamespace(crop_cache_dir=tmp_path)
    )
    reg = InMemoryRegistry()
    reg.add('widget')
    monkeypatch.setattr(vlm_mod, 'get_class_registry', lambda: reg)

    fake = QueryFakeOpenSearch({ITEMS: {cid: _item(cid) for cid in failures}})

    class _Labeler:
        identity = VlmIdentity('env@None', 'test-vlm')
        _pack = GENERIC_ITEM_PACK

        async def label_or_propose_batch(
            self, crops: list[Any], _names: list[str], **_k: Any
        ) -> list[Any]:
            return [
                VlmClassPrediction(
                    img_id=c.img_id,
                    class_name='' if failures[c.img_id] else 'widget',
                    confidence='low' if failures[c.img_id] else 'high',
                    failure=failures[c.img_id],
                )
                for c in crops
            ]

    labeler = _Labeler()
    return await run_label_batch(fake, labeler, list(failures)), fake


@pytest.mark.usefixtures('vlm_env_active')
@pytest.mark.asyncio
async def test_label_batch_is_503_vlm_unreachable_when_no_crop_got_an_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, fake = await _call_route(
        tmp_path, monkeypatch, {'c1': 'request_failed', 'c2': 'request_failed'}
    )
    assert result['vlm_unreachable'] is True
    # Nothing was written: the crops stay pending, not failed.
    assert all(
        'class_source' in d and d['class_source'] == 'item_proposal'
        for d in fake.docs(ITEMS).values()
    )


@pytest.mark.usefixtures('vlm_env_active')
@pytest.mark.asyncio
async def test_label_batch_with_some_answers_is_still_a_200(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = await _call_route(tmp_path, monkeypatch, {'c1': 'request_failed', 'c2': None})
    assert result['predicted'] == 2
    assert result['updated'] == 1


# ---- worker ---------------------------------------------------------------


class _Env:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.done = False
        self.labelled: list[list[str]] = []
        self.outage_calls = 2
        self.fetch_exclude: list[list[str]] = []
        self.sleeps: list[float] = []
        self.stall_seen: list[str | None] = []
        self.stall_fn: Any = None
        self.exc: Exception | None = None
        m = monkeypatch
        m.setattr(vlm_worker, 'curation_api_prefix', lambda: '/curation')
        m.setattr(vlm_worker, 'script_project_registry', lambda *_a, **_k: object())
        m.setattr(vlm_worker, 'make_script_opensearch', lambda *_a, **_k: self)
        m.setattr(vlm_worker, 'projects_with_active_vlm', self._projects)
        m.setattr(vlm_worker, 'fetch_pending_ids', self._fetch)
        m.setattr(vlm_worker, 'label_batch', self._label)
        m.setattr(vlm_worker, 'heartbeat_loop', self._heartbeat)
        m.setattr(vlm_worker, 'project_has_item_classes', self._has_classes)
        m.setattr(vlm_worker, 'bind_project', lambda _r: __import__('contextlib').nullcontext())
        m.setattr(vlm_worker, 'vlm_paused', self._vlm_paused)
        real_sleep = asyncio.sleep

        async def _sleep(seconds: float) -> None:
            if seconds != 30.0:
                self.sleeps.append(seconds)
                if self.stall_fn is not None:
                    self.stall_seen.append(self.stall_fn())
            await real_sleep(0)

        m.setattr(vlm_worker.asyncio, 'sleep', _sleep)

    async def close(self) -> None:
        return None

    async def _has_classes(self, _record: Any) -> bool:
        return True

    async def _projects(self, *_a: Any) -> list[Any]:
        return [SimpleNamespace(slug='alpha')]

    async def _fetch(self, _os: Any, **kw: Any) -> list[str]:
        self.fetch_exclude.append(list(kw['exclude_ids']))
        # Like OpenSearch: the crops are pending until a call labels them.
        return [] if self.done or set(kw['exclude_ids']) else ['c1', 'c2']

    async def _label(self, _c: Any, *, crop_ids: Any, **_k: Any) -> dict[str, Any]:
        self.labelled.append(list(crop_ids))
        if self.outage_calls > 0:
            self.outage_calls -= 1
            if self.exc is not None:
                raise self.exc
            return {'vlm_unreachable': True, 'message': 'name or service not known'}
        self.done = True
        return {'predicted': len(crop_ids), 'updated': len(crop_ids)}

    async def _heartbeat(self, _name: str, _status: Any, _stop: Any, stall_fn: Any = None) -> None:
        self.stall_fn = stall_fn

    async def _vlm_paused(self) -> bool:
        return False


def _args() -> Any:
    return vlm_cli.parse_args(
        [
            '--api', 'http://api.test', '--opensearch', 'http://os.test',
            '--vlm-batch-size', '2', '--concurrency', '1', '--poll-interval', '0.5',
            '--idle-stop-after', '6',
        ]
    )  # fmt: skip


@pytest.mark.asyncio
async def test_worker_backs_off_releases_and_resumes_when_the_endpoint_returns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env = _Env(monkeypatch)
    await asyncio.wait_for(vlm_worker.run(_args()), 10)
    out = capsys.readouterr().out

    # The two failed attempts were retried until the endpoint answered; the
    # final call went through, so the same crops were labelled.
    assert env.labelled[-1] == ['c1', 'c2']
    assert len(env.labelled) >= 2
    # A backoff longer than the poll interval, growing with each failure.
    backoffs = [s for s in env.sleeps if s > 0.5]
    assert backoffs[:1] == [1.0]
    # The ids were released: a later fetch picked them up again.
    assert env.fetch_exclude.count([]) >= 2
    # The reason is announced and the recovery is announced.
    assert 'VLM endpoint unreachable' in out
    assert 'VLM endpoint recovered; resuming' in out
    assert 'processed=2 updated=2 chunks=1' in out
    # The stall reason is served while down and cleared afterwards.
    assert any(s and 'VLM endpoint unreachable' in s for s in env.stall_seen)
    assert env.stall_fn() is None


@pytest.mark.asyncio
async def test_worker_backs_off_when_the_api_itself_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env = _Env(monkeypatch)
    env.exc = httpx.ConnectError('[Errno -2] Name or service not known')
    await asyncio.wait_for(vlm_worker.run(_args()), 10)
    out = capsys.readouterr().out
    assert 'curation API unreachable' in out
    assert 'API recovered; resuming' in out
    assert [s for s in env.sleeps if s > 0.5][:1] == [1.0]
    assert env.labelled[-1] == ['c1', 'c2']
    assert 'processed=2 updated=2 chunks=1' in out


def _job_api(*, statuses: list[str], result: dict | None = None, error: str | None = None):
    """A mock API: POST queues job j1, GET walks ``statuses`` then repeats the last."""
    polls = iter(statuses)
    last = {'status': statuses[-1]}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == 'POST':
            return httpx.Response(202, json={'job_id': 'j1', 'status': 'queued'})
        status = next(polls, last['status'])
        return httpx.Response(
            200, json={'id': 'j1', 'status': status, 'result': result, 'error_message': error}
        )

    return httpx.MockTransport(handler)


async def _poll(transport: httpx.MockTransport) -> dict:
    async with httpx.AsyncClient(transport=transport) as client:
        return await vlm_worker.label_batch(
            client, api='http://api.test', api_prefix='/curation', slug='alpha', crop_ids=['c1']
        )


@pytest.mark.asyncio
async def test_label_batch_client_polls_the_job_and_returns_its_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vlm_worker, '_POLL_S', 0.0)
    result = {'vlm_unreachable': True, 'message': 'no route to host'}
    got = await _poll(_job_api(statuses=['queued', 'running', 'completed'], result=result))
    assert got == result


@pytest.mark.asyncio
async def test_label_batch_client_raises_for_a_failed_job_and_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vlm_worker, '_POLL_S', 0.0)
    with pytest.raises(RuntimeError, match='boom'):
        await _poll(_job_api(statuses=['running', 'failed'], error='boom'))
    monkeypatch.setattr(vlm_worker, '_JOB_TIMEOUT_S', 0.0)
    with pytest.raises(TimeoutError):
        await _poll(_job_api(statuses=['running']))


def test_heartbeat_serves_the_stall_reason_and_stays_healthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.services.curation.worker_liveness as liveness

    monkeypatch.setattr(liveness, 'HEARTBEAT_DIR', tmp_path)
    write_heartbeat('vlm_worker', {'producer': True}, stall='VLM endpoint unreachable: x')
    healthy, reason = check_heartbeat('vlm_worker', max_age_s=120.0)
    assert healthy is True
    assert 'stalled: VLM endpoint unreachable: x' in reason
