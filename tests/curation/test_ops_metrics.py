"""Issue #94: the operator ``op_*`` metrics behind the Grafana dashboards.

Each metric is pinned (name, type, labels, gauge mode) and then driven from the
real code path that is meant to move it, using the existing fakes.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import httpx
import pytest
from _fake_project_registry import StaticProjectRegistry, default_project_record
from prometheus_client import REGISTRY
from prometheus_client.metrics import MetricWrapperBase

from curation.test_ingest_service import _jpeg_bytes, _make_service
from curation.test_vlm_labeler import _fake_client, _make_chat_response
from src.config.curation import IndexRole
from src.config.project_context import bind_project
from src.services.curation import ops_metrics as om, ops_metrics_refresh as refresh
from src.services.curation.worker_liveness import write_heartbeat
from src.services.labeling.vlm_labeler import VlmLabeler
from src.services.projects.registry import set_project_registry


REPO_ROOT = Path(__file__).resolve().parents[2]


# name -> (family type, labelnames, gauge multiprocess mode or None)
SPEC: dict[str, tuple[str, tuple[str, ...], str | None]] = {
    'op_ingest_images_total': ('counter', ('project', 'outcome'), None),
    'op_ingest_items_total': ('counter', ('project',), None),
    'op_queue_depth': ('gauge', ('queue', 'project'), 'livemostrecent'),
    'op_queue_oldest_item_age_seconds': ('gauge', ('queue',), 'livemostrecent'),
    'op_worker_last_heartbeat_timestamp_seconds': ('gauge', ('worker',), 'livemostrecent'),
    'op_worker_up': ('gauge', ('worker',), 'livemostrecent'),
    'op_embedding_state_items': ('gauge', ('project', 'state'), 'livemostrecent'),
    'op_region_segmenter_request_seconds': ('histogram', ('outcome',), None),
    'op_region_segmenter_requests_total': ('counter', ('outcome',), None),
    'op_detection_worker_items_total': ('counter', ('outcome',), None),
    'op_vlm_tokens_total': ('counter', ('direction', 'model'), None),
    'op_vlm_request_seconds': ('histogram', ('model', 'outcome'), None),
    'op_opensearch_shards': ('gauge', ('project', 'index_role'), 'livemostrecent'),
    'op_opensearch_store_bytes': ('gauge', ('project', 'index_role'), 'livemostrecent'),
}


def _objects() -> dict[str, MetricWrapperBase]:
    found: dict[str, MetricWrapperBase] = {}
    for value in vars(om).values():
        if isinstance(value, MetricWrapperBase):
            found[value._name + ('_total' if value._type == 'counter' else '')] = value
    return found


def _sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.fixture(autouse=True)
def _registry_with_default_project() -> Any:
    set_project_registry(StaticProjectRegistry([default_project_record()]))  # type: ignore[arg-type]
    yield
    set_project_registry(None)


@pytest.mark.parametrize('name', sorted(SPEC))
def test_metric_name_type_labels(name: str) -> None:
    kind, labels, mode = SPEC[name]
    metric = _objects()[name]
    assert metric._type == kind
    assert tuple(metric._labelnames) == labels
    if mode is not None:
        assert metric._multiprocess_mode == mode
    assert 'item' not in ''.join(labels)
    assert 'crop' not in ''.join(labels)


def test_every_spec_metric_is_defined_and_nothing_else_is_added() -> None:
    assert set(SPEC) == set(_objects())


# ---- ingest -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_counts_images_and_items_per_project() -> None:
    svc, _, _ = _make_service()
    with bind_project(default_project_record()):
        before_ok = _sample('op_ingest_images_total', project='default', outcome='ok')
        before_items = _sample('op_ingest_items_total', project='default')
        before_fail = _sample('op_ingest_images_total', project='default', outcome='failed')
        before_skip = _sample('op_ingest_images_total', project='default', outcome='skipped')
        ok = await svc.ingest_one(_jpeg_bytes(), '/tmp/a.jpg')
        await svc.ingest_one(b'', '/tmp/empty.jpg')
        await svc.ingest_one(_jpeg_bytes(), '/tmp/a-again.jpg')  # same bytes -> duplicate
    assert ok.status == 'success'
    assert _sample('op_ingest_images_total', project='default', outcome='ok') == before_ok + 1
    assert _sample('op_ingest_items_total', project='default') == before_items + ok.n_crops
    assert _sample('op_ingest_images_total', project='default', outcome='failed') == before_fail + 1
    assert (
        _sample('op_ingest_images_total', project='default', outcome='skipped') == before_skip + 1
    )


@pytest.mark.asyncio
async def test_batch_level_duplicates_count_as_skipped() -> None:
    svc, _, _ = _make_service()
    data = _jpeg_bytes()
    with bind_project(default_project_record()):
        before_skip = _sample('op_ingest_images_total', project='default', outcome='skipped')
        before_ok = _sample('op_ingest_images_total', project='default', outcome='ok')
        result = await svc.ingest_batch([data, data], ['/tmp/x.jpg', '/tmp/y.jpg'])
    assert (result.summary.successful, result.summary.duplicates) == (1, 1)
    assert (
        _sample('op_ingest_images_total', project='default', outcome='skipped') == before_skip + 1
    )
    assert _sample('op_ingest_images_total', project='default', outcome='ok') == before_ok + 1


def test_project_label_is_bounded_and_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    base = default_project_record()
    records = [
        base.__class__(**{**base.__dict__, 'slug': s})
        for s in ('delta', 'alpha', 'charlie', 'bravo')
    ]
    set_project_registry(StaticProjectRegistry(records))  # type: ignore[arg-type]
    monkeypatch.setenv('OP_METRICS_MAX_PROJECT_LABELS', '2')
    assert [om.project_label(s) for s in ('alpha', 'bravo', 'charlie', 'delta')] == [
        'alpha',
        'bravo',
        'other',
        'other',
    ]


# ---- VLM --------------------------------------------------------------------


def _labeler(handler: Any) -> VlmLabeler:
    client, _ = _fake_client(handler)
    return VlmLabeler(
        base_url='http://fake/v1', model='m-test', requests_per_second=1000.0, client=client
    )


def test_vlm_tokens_and_latency_from_a_real_chat_call() -> None:
    def handler(_req: httpx.Request, _n: int) -> httpx.Response:
        body = _make_chat_response('OK')
        body['usage'] = {'prompt_tokens': 11, 'completion_tokens': 3}
        return httpx.Response(200, json=body)

    before = {
        d: _sample('op_vlm_tokens_total', direction=d, model='m-test')
        for d in ('prompt', 'completion')
    }
    count = _sample('op_vlm_request_seconds_count', model='m-test', outcome='ok')
    health = asyncio.run(_labeler(handler).health())
    assert health.reachable
    assert (
        _sample('op_vlm_tokens_total', direction='prompt', model='m-test') == before['prompt'] + 11
    )
    assert (
        _sample('op_vlm_tokens_total', direction='completion', model='m-test')
        == before['completion'] + 3
    )
    assert _sample('op_vlm_request_seconds_count', model='m-test', outcome='ok') == count + 1


def test_vlm_error_is_timed_and_adds_no_tokens() -> None:
    before = _sample('op_vlm_request_seconds_count', model='m-test', outcome='error')
    tokens = _sample('op_vlm_tokens_total', direction='prompt', model='m-test')
    health = asyncio.run(_labeler(lambda *_a: httpx.Response(400, json={'error': 'bad'})).health())
    assert not health.reachable
    assert _sample('op_vlm_request_seconds_count', model='m-test', outcome='error') == before + 1
    assert _sample('op_vlm_tokens_total', direction='prompt', model='m-test') == tokens


def test_vlm_response_without_usage_adds_no_tokens() -> None:
    tokens = _sample('op_vlm_tokens_total', direction='prompt', model='m-nousage')
    om.record_vlm_request('m-nousage', 'ok', 0.1, {'choices': []})
    assert _sample('op_vlm_tokens_total', direction='prompt', model='m-nousage') == tokens


# ---- worker liveness --------------------------------------------------------


def test_heartbeat_sets_up_and_timestamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation import worker_liveness

    monkeypatch.setattr(worker_liveness, 'HEARTBEAT_DIR', tmp_path)
    t0 = time.time()
    write_heartbeat('vlm_worker', {'producer': True, 'consumers': True})
    assert _sample('op_worker_up', worker='vlm') == 1
    assert _sample('op_worker_last_heartbeat_timestamp_seconds', worker='vlm') >= t0
    write_heartbeat('vlm_worker', {'producer': True, 'consumers': False})
    assert _sample('op_worker_up', worker='vlm') == 0
    write_heartbeat('detection_worker', {'writer': True})
    write_heartbeat('auto_label_worker', {'poll': True})
    assert _sample('op_worker_up', worker='detection') == 1
    assert _sample('op_worker_up', worker='curation') == 1


def test_unknown_heartbeat_name_adds_no_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation import worker_liveness

    monkeypatch.setattr(worker_liveness, 'HEARTBEAT_DIR', tmp_path)
    write_heartbeat('some_adhoc_name', {'x': True})
    assert REGISTRY.get_sample_value('op_worker_up', {'worker': 'some_adhoc_name'}) is None


@pytest.mark.asyncio
async def test_segmenter_probe_sets_worker_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://seg.invalid:1')
    answers = iter([2, None])

    async def fake_instances(_url: str, **_kw: Any) -> int | None:
        return next(answers)

    monkeypatch.setattr('src.services.detection.segmenter_http.segmenter_instances', fake_instances)
    await refresh.refresh_segmenter_up(now=1234.0)
    assert _sample('op_worker_up', worker='segmenter') == 1
    assert _sample('op_worker_last_heartbeat_timestamp_seconds', worker='segmenter') == 1234.0
    await refresh.refresh_segmenter_up(now=2000.0)
    assert _sample('op_worker_up', worker='segmenter') == 0
    assert _sample('op_worker_last_heartbeat_timestamp_seconds', worker='segmenter') == 1234.0


# ---- snapshot gauges --------------------------------------------------------


class _FakeSnapshotClient:
    """Answers the refresher's one search per project and one cat.indices call."""

    def __init__(self, items_index: str, store_rows: list[dict[str, str]]) -> None:
        self.items_index = items_index
        self.searches: list[dict[str, Any]] = []
        self._rows = store_rows
        self.cat = self

    async def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        assert index == self.items_index
        self.searches.append(body)
        return {
            'aggregations': {
                'segment': {'doc_count': 7, 'oldest': {'value': 1_000_000.0}},
                'label': {'doc_count': 2, 'oldest': {'value': 1_900_000.0}},
                'embed': {'doc_count': 5, 'oldest': {'value': None}},
                'embedded': {'doc_count': 90},
                'failed': {'doc_count': 1},
            }
        }

    async def indices(self, **_kw: Any) -> list[dict[str, str]]:
        return self._rows


@pytest.mark.asyncio
async def test_refresh_sets_queue_embedding_and_storage_gauges() -> None:
    record = default_project_record()
    items = record.resources.indexes[IndexRole.ITEMS]
    images = record.resources.indexes[IndexRole.IMAGES]
    client = _FakeSnapshotClient(
        items,
        [
            {'index': items, 'pri': '2', 'rep': '1', 'store.size': '4096'},
            {'index': images, 'pri': '1', 'rep': '0', 'store.size': '100'},
            {'index': 'op_prj_someone_else__items', 'pri': '9', 'rep': '9', 'store.size': '9'},
        ],
    )
    await refresh.refresh_queues(client, [record], now=2000.0)
    await refresh.refresh_storage(client, [record])

    slug = record.slug
    assert _sample('op_queue_depth', queue='segment', project=slug) == 7
    assert _sample('op_queue_depth', queue='label', project=slug) == 2
    assert _sample('op_queue_depth', queue='embed', project=slug) == 5
    assert _sample('op_queue_oldest_item_age_seconds', queue='segment') == 1000.0
    assert _sample('op_queue_oldest_item_age_seconds', queue='label') == 100.0
    assert _sample('op_queue_oldest_item_age_seconds', queue='embed') == 0.0
    assert REGISTRY.get_sample_value('op_queue_depth', {'queue': 'ingest', 'project': slug}) is None
    assert _sample('op_embedding_state_items', project=slug, state='pending') == 5
    assert _sample('op_embedding_state_items', project=slug, state='embedded') == 90
    assert _sample('op_embedding_state_items', project=slug, state='failed') == 1
    assert _sample('op_opensearch_shards', project=slug, index_role='items') == 4
    assert _sample('op_opensearch_store_bytes', project=slug, index_role='items') == 4096
    assert _sample('op_opensearch_shards', project=slug, index_role='images') == 1

    body = json.dumps(client.searches[0])
    assert 'pending_detection' in body
    assert 'pending_verification' in body
    assert 'embedding_state' in body


@pytest.mark.asyncio
async def test_failed_read_keeps_the_previous_value_not_zero() -> None:
    record = default_project_record()

    class _Down:
        async def search(self, **_kw: Any) -> dict[str, Any]:
            raise ConnectionError('opensearch down')

    om.OP_QUEUE_DEPTH.labels(queue='segment', project=record.slug).set(42)
    await refresh.refresh_queues(_Down(), [record], now=1.0)
    assert _sample('op_queue_depth', queue='segment', project=record.slug) == 42


def test_refresh_throttle_lets_one_worker_per_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('PROMETHEUS_MULTIPROC_DIR', str(tmp_path))
    assert refresh._claim_turn(30.0) is True
    assert refresh._claim_turn(30.0) is False


# ---- segmenter + detection worker (real pipeline, mocked externals) ----------


@pytest.mark.usefixtures('reference_region_profile')
class TestWorkerPipelineMetrics:
    @pytest.mark.asyncio
    async def test_segmenter_hit_and_detection_ok_increment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from curation.test_region_cascade_integrity import _accept, _drive_worker, _FakeOpenSearch
        from curation.test_region_worker import _item_with
        from src.services.detection.cascade_detect.candidate import RegionCandidate

        hit = _sample('op_region_segmenter_requests_total', outcome='hit')
        hit_n = _sample('op_region_segmenter_request_seconds_count', outcome='hit')
        ok = _sample('op_detection_worker_items_total', outcome='ok')
        fake_os = _FakeOpenSearch(
            {'c1': _item_with(status='pending_detection', class_name='some-class')},
            search_delay=0.0,
            lag_searches=0,
        )
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=RegionCandidate(bbox_norm=(0.4, 0.45, 0.5, 0.5), score=0.8, source='sam3'),
            reply=_accept(),
            visible=True,
            class_group=lambda _name: 'group_a',
            profile_overrides={'secondary_shape_groups': frozenset({'group_a'})},
        )
        assert _sample('op_region_segmenter_requests_total', outcome='hit') >= hit + 1
        assert _sample('op_region_segmenter_request_seconds_count', outcome='hit') >= hit_n + 1
        assert _sample('op_detection_worker_items_total', outcome='ok') >= ok + 1

    @pytest.mark.asyncio
    async def test_segmenter_miss_increments(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from curation.test_region_cascade_integrity import _accept, _drive_worker, _FakeOpenSearch
        from curation.test_region_worker import _item_with

        miss = _sample('op_region_segmenter_requests_total', outcome='miss')
        fake_os = _FakeOpenSearch(
            {'c1': _item_with(status='pending_detection', class_name='some-class')},
            search_delay=0.0,
            lag_searches=0,
        )
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=None,
            reply=_accept(),
            visible=True,
            class_group=lambda _name: 'group_a',
            profile_overrides={'secondary_shape_groups': frozenset({'group_a'})},
        )
        assert _sample('op_region_segmenter_requests_total', outcome='miss') >= miss + 1


def test_segmenter_error_outcome_records_both_series() -> None:
    n = _sample('op_region_segmenter_requests_total', outcome='error')
    s = _sample('op_region_segmenter_request_seconds_count', outcome='error')
    om.record_segmenter_request('error', 0.2)
    assert _sample('op_region_segmenter_requests_total', outcome='error') == n + 1
    assert _sample('op_region_segmenter_request_seconds_count', outcome='error') == s + 1


# ---- worker /metrics server + multiprocess aggregation ------------------------


def test_worker_metrics_server_serves_op_series(monkeypatch: pytest.MonkeyPatch) -> None:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    om.OP_WORKER_UP.labels(worker='detection').set(1)
    assert om.start_worker_metrics_server(str(port)) == port
    body = urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=5).read().decode()
    assert 'op_worker_up{worker="detection"} 1.0' in body


@pytest.mark.parametrize('setting', ['0', 'off', '', 'not-a-port'])
def test_worker_metrics_server_can_be_disabled_or_bad(setting: str) -> None:
    assert om.start_worker_metrics_server(setting) is None


def _py(code: str, mp_dir: Path) -> str:
    env = {k: v for k, v in os.environ.items() if k != 'PROMETHEUS_MULTIPROC_DIR'}
    env.update(PROMETHEUS_MULTIPROC_DIR=str(mp_dir), PYTHONPATH=str(REPO_ROOT))
    return subprocess.run(
        [sys.executable, '-c', code],
        env=env,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_snapshot_gauge_shows_the_latest_live_value_across_processes(tmp_path: Path) -> None:
    # Two live API workers: the older one's value must not pin the scrape.
    holder = (
        'import sys\n'
        'from src.services.curation.ops_metrics import OP_QUEUE_DEPTH as G\n'
        "G.labels(queue='segment', project='p').set({v})\n"
        'print("ready", flush=True)\n'
        'sys.stdin.read()\n'  # stay alive (a live pid) until the test closes stdin
    )
    env = dict(os.environ)
    env.update(PROMETHEUS_MULTIPROC_DIR=str(tmp_path), PYTHONPATH=str(REPO_ROOT))
    procs = []
    try:
        for value in (99, 3):
            proc = subprocess.Popen(
                [sys.executable, '-c', holder.format(v=value)],
                env=env,
                cwd=REPO_ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            procs.append(proc)
            assert proc.stdout is not None
            # Imports may print unrelated warnings to stdout first.
            for line in iter(proc.stdout.readline, ''):
                if line.strip() == 'ready':
                    break
            else:
                pytest.fail('holder process exited before it was ready')
            time.sleep(0.05)
        out = _py(
            'from src.core.metrics import render_metrics\nprint(render_metrics()[0].decode())\n',
            tmp_path,
        )
    finally:
        for proc in procs:
            proc.kill()
            proc.wait()
    lines = re.findall(r'^op_queue_depth\{[^}]*\} (\S+)$', out, re.MULTILINE)
    assert lines == ['3.0']


def test_project_label_of_a_project_the_snapshot_has_not_seen_yet() -> None:
    """#124: a bound project missing from this process's registry snapshot (created
    moments ago / a failed refresh) still gets its own label, not ``other``."""
    set_project_registry(StaticProjectRegistry([default_project_record()]))  # type: ignore[arg-type]
    assert om.project_label('p1') == 'p1'


def test_project_label_cap_still_folds_projects_beyond_it(monkeypatch: pytest.MonkeyPatch) -> None:
    base = default_project_record()
    records = [base.__class__(**{**base.__dict__, 'slug': s}) for s in ('alpha', 'bravo')]
    set_project_registry(StaticProjectRegistry(records))  # type: ignore[arg-type]
    monkeypatch.setenv('OP_METRICS_MAX_PROJECT_LABELS', '2')
    assert om.project_label('zulu') == 'other'
    assert om.project_label('aaa-new') == 'aaa-new'


@pytest.mark.asyncio
async def test_tombstones_sorting_before_a_live_project_do_not_push_it_to_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#124: deleted tombstones stay in the registry snapshot; ranking them used up the
    whole label cap so every live project was filed under ``other``."""
    base = default_project_record()

    def rec(slug: str, status: str) -> Any:
        return base.__class__(**{**base.__dict__, 'slug': slug, 'status': status})

    tombstones = [rec(f'accept-{i:02d}', 'deleted') for i in range(12)]
    live = rec('v041-verify', 'active')
    set_project_registry(StaticProjectRegistry([*tombstones, live]))  # type: ignore[arg-type]
    monkeypatch.setenv('OP_METRICS_MAX_PROJECT_LABELS', '10')
    assert om.project_label('v041-verify') == 'v041-verify'

    svc, _, _ = _make_service()
    with bind_project(live):
        before = _sample('op_ingest_images_total', project='v041-verify', outcome='ok')
        ok = await svc.ingest_one(_jpeg_bytes(), '/tmp/tomb.jpg')
    assert ok.status == 'success'
    assert _sample('op_ingest_images_total', project='v041-verify', outcome='ok') == before + 1


def test_live_projects_beyond_the_cap_still_fold_to_other_with_tombstones_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = default_project_record()
    records = [
        base.__class__(**{**base.__dict__, 'slug': s, 'status': st})
        for s, st in (('aaa', 'deleted'), ('bravo', 'active'), ('charlie', 'active'))
    ]
    set_project_registry(StaticProjectRegistry(records))  # type: ignore[arg-type]
    monkeypatch.setenv('OP_METRICS_MAX_PROJECT_LABELS', '1')
    assert om.project_label('bravo') == 'bravo'
    assert om.project_label('charlie') == 'other'
