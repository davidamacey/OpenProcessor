"""The P2 worker leak test (projects_plan.md §5.1/§5.2).

Two projects, ``alpha`` and ``beta``, seeded with distinct items on one
fake OpenSearch behind the REAL project guard (a real ``AsyncOpenSearch``
client whose transport is a recording in-memory fake). The worker cores
run exactly as the long-lived processes run them: unbound at the top,
discovering both projects, binding per project/item.

Asserted for every worker:
- the guard never rejected an access (``cross_project_access_count``
  unchanged) -- a rejected access would also have raised;
- every OpenSearch call touched only indexes of the project bound at the
  time of the call;
- every write landed in its own project's items index, carrying data
  computed for that project.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import NotFoundError
from opensearchpy.serializer import JSONSerializer

from scripts.curation.worker.bulk_writer import _bulk_update
from scripts.curation.worker.fairness import FairnessScheduler, fetch_pending_multi_project
from scripts.curation.worker.state import _ItemTask, bind_task_project, bound_class_catalog
from scripts.curation.worker.verify import TaskBoxInput, _combined_class_update, verdicts_to_boxes
from src.clients.curation_opensearch import ClassRegistryFile, RegistryClassEntry
from src.config import get_region_fields
from src.config.curation import IndexRole, base_curation_config
from src.config.project_context import bind_project, try_current_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.curation.region_boxes import boxes_write_fields
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_models import VlmCombinedReply
from src.services.projects.guard import cross_project_access_count, install_project_guard


if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.unbound

SLUGS = ('alpha', 'beta')
CLASSES = {'alpha': ['car', 'truck'], 'beta': ['forklift', 'pallet', 'drum']}


def _record(tmp_path: Any, slug: str) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    resources = resources_for_new(slug, base_curation_config())
    root = tmp_path / slug
    root.mkdir(parents=True, exist_ok=True)
    registry_path = root / 'class_registry.json'
    registry_path.write_text(
        ClassRegistryFile(
            classes=[
                RegistryClassEntry(class_id=10 * i + len(slug), class_name=name)
                for i, name in enumerate(CLASSES[slug])
            ]
        ).model_dump_json()
    )
    resources = resources.__class__(
        **{
            **resources.__dict__,
            'project_state_dir': root / 'state',
            'class_registry_path': registry_path,
        }
    )
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources,
    )


def _items_index(record: ProjectRecord) -> str:
    return record.resources.indexes[IndexRole.ITEMS]


class _Snapshot:
    """What the guard reads: every project record, by slug."""

    def __init__(self, records: list[ProjectRecord]) -> None:
        self._by_slug = {r.slug: r for r in records}

    def snapshot(self) -> dict[str, ProjectRecord]:
        return dict(self._by_slug)

    def active_projects(self) -> list[ProjectRecord]:
        return list(self._by_slug.values())


class _RecordingTransport:
    """In-memory OpenSearch: per-index doc store answering a pending
    ``_search`` (docs matching ``pending``), ``_count``, the OCC ``_mget``
    and ``_bulk``. Records every call with the project bound when it
    arrived (after the guard let it through)."""

    serializer = JSONSerializer()

    def __init__(
        self,
        docs: dict[str, dict[str, dict[str, Any]]],
        pending: Callable[[dict[str, Any]], bool],
    ) -> None:
        self.docs = docs
        self.pending = pending
        self.calls: list[tuple[str | None, str, set[str]]] = []
        self.writes: list[tuple[str, str, dict[str, Any]]] = []

    async def perform_request(
        self, method: str, url: str, params: Any = None, body: Any = None, **_kw: Any
    ) -> Any:
        del params
        path, _, _query = url.partition('?')
        parts = [p for p in path.split('/') if p]
        action = parts[-1]
        url_index: str = '' if parts[0].startswith('_') else parts[0]
        bound = try_current_project()
        touched: set[str] = set()
        if action == '_search':
            touched.add(url_index)
            result = self._search(url_index, body)
        elif action == '_count':
            touched.add(url_index)
            result = {'count': len(self.docs[url_index])}
        elif action == '_mget':
            wanted: list[tuple[str, str]] = [
                (str(d.get('_index') or url_index), str(d['_id'])) for d in body['docs']
            ]
            touched |= {index for index, _ in wanted}
            result = {
                'docs': [
                    {
                        '_index': index,
                        '_id': i,
                        'found': True,
                        '_seq_no': 1,
                        '_primary_term': 1,
                        '_source': dict(self.docs[index][i]),
                    }
                    if i in self.docs[index]
                    else {'_index': index, '_id': i, 'found': False}
                    for index, i in wanted
                ]
            }
        elif action == '_bulk':
            result = self._bulk(url_index, body, touched)
        elif method == 'GET' and parts[-2:-1] == ['_doc']:
            # The project's settings document (its VLM policy): never written, so absent.
            touched.add(url_index)
            self.calls.append((bound.record.slug if bound else None, url, touched))
            raise NotFoundError(404, 'not_found', {'found': False})
        else:
            msg = f'unexpected OpenSearch call {method} {url}'
            raise AssertionError(msg)
        self.calls.append((bound.record.slug if bound else None, url, touched))
        return result

    async def close(self) -> None:
        return None

    def _search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        excluded: set[str] = set()
        for clause in (body.get('query') or {}).get('bool', {}).get('must_not', []):
            excluded |= set((clause.get('ids') or {}).get('values') or [])
        hits = [
            {'_index': index, '_id': doc_id, '_source': src}
            for doc_id, src in sorted(self.docs[index].items())
            if self.pending(src) and doc_id not in excluded
        ][: body.get('size', 10)]
        return {'hits': {'hits': hits}}

    def _bulk(self, url_index: str, body: Any, touched: set[str]) -> dict[str, Any]:
        text = body.decode() if isinstance(body, bytes) else body
        lines = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        items = []
        for action_line, doc_line in zip(lines[::2], lines[1::2], strict=True):
            meta = action_line['update']
            index = str(meta.get('_index') or url_index)
            touched.add(index)
            self.docs[index][meta['_id']].update(doc_line['doc'])
            self.writes.append((index, meta['_id'], doc_line['doc']))
            items.append({'update': {'_index': index, '_id': meta['_id'], 'status': 200}})
        return {'errors': False, 'items': items}


@pytest.fixture
def projects(tmp_path: Any) -> dict[str, ProjectRecord]:
    return {slug: _record(tmp_path, slug) for slug in SLUGS}


@pytest.fixture
def world(projects: dict[str, ProjectRecord]) -> tuple[AsyncOpenSearch, _RecordingTransport]:
    status = get_region_fields().status
    docs = {
        _items_index(rec): {
            f'{slug}-{i}': {
                'crop_id': f'{slug}-{i}',
                'image_path': f'/data/{slug}/{i}.jpg',
                'bbox_norm': [0.1, 0.1, 0.5, 0.5],
                status: 'pending_detection',
                'class_name': '',
            }
            for i in range(7)
        }
        for slug, rec in projects.items()
    }
    transport = _RecordingTransport(docs, lambda src: src.get(status) == 'pending_detection')
    client = AsyncOpenSearch(hosts=['http://fake:9200'])
    client.transport = transport
    install_project_guard(client, _Snapshot(list(projects.values())))
    return client, transport


def _index_owner(projects: dict[str, ProjectRecord]) -> dict[str, str]:
    return {name: slug for slug, rec in projects.items() for name in rec.resources.indexes.values()}


def _assert_no_cross_project_calls(
    transport: _RecordingTransport, projects: dict[str, ProjectRecord]
) -> None:
    owner = _index_owner(projects)
    assert transport.calls, 'the worker made no OpenSearch calls'
    for bound, url, touched in transport.calls:
        owners = {owner[i] for i in touched if i in owner}
        assert owners == {bound}, f'{url} touched {owners} while {bound!r} was bound'


class _CatalogVlm:
    """A VLM client double for the combined call. ``class_names`` is the
    process-wide list the worker used to load once at startup (alpha's);
    the call must ignore it and use the bound project's own catalog."""

    def __init__(self) -> None:
        self.class_names = list(CLASSES['alpha'])
        self.name_to_id = {n: i for i, n in enumerate(self.class_names)}
        self.prompts: list[tuple[str, list[str]]] = []

    async def label_combined(self, *, img_id: str, class_names: list[str], **_kw: Any) -> Any:
        self.prompts.append((img_id, list(class_names)))
        # Answer with the last class of whatever catalog it was shown.
        return VlmCombinedReply(
            img_id=img_id,
            class_id=len(class_names) - 1,
            class_confidence='high',
            region_visible=True,
            region_boxes=[VlmBoxVerdict(box=1, bbox_correct=True, confidence='high')],
        )


@pytest.mark.asyncio
async def test_detection_worker_keeps_every_read_and_write_in_its_project(
    projects: dict[str, ProjectRecord],
    world: tuple[AsyncOpenSearch, _RecordingTransport],
) -> None:
    client, transport = world
    rejected_before = cross_project_access_count()
    vlm = _CatalogVlm()

    tasks = await fetch_pending_multi_project(
        client,
        registry=_Snapshot(list(projects.values())),
        scheduler=FairnessScheduler(),
        fetch_n=8,
        queue_max=100,
    )
    assert {t.project.slug for t in tasks} == set(SLUGS)

    async def _consume(task: _ItemTask) -> None:
        # As a pipeline consumer does: bind the item's project, then run
        # the combined class+region call on it (W8: box-list path).
        bind_task_project(task)
        task.crop_jpeg = b'jpeg'
        class_names, name_to_id = bound_class_catalog()
        reply = await vlm.label_combined(
            img_id=task.crop_id,
            jpeg_bytes=task.crop_jpeg,
            class_names=class_names,
            region_bboxes_norm=[(0.3, 0.3, 0.6, 0.6)],
        )
        cand = TaskBoxInput(
            bbox_in_crop=(0.3, 0.3, 0.6, 0.6),
            bbox_in_source=(0.2, 0.2, 0.4, 0.4),
            score=0.9,
            detector='det',
            detector_version='1',
            source='det',
        )
        boxes, status, _extra = verdicts_to_boxes(
            [cand], reply.region_boxes, item_bbox_norm=task.item_bbox_norm
        )
        assert status is not None
        F = get_region_fields()
        task.update_doc = {
            F.status: status,
            **boxes_write_fields(boxes),
            **_combined_class_update(
                reply, class_names, name_to_id=name_to_id, vlm_model='vlm-model'
            ),
        }

    # Interleave the two projects' items across concurrent consumers.
    await asyncio.gather(*(_consume(t) for t in sorted(tasks, key=lambda t: t.crop_id[-1])))
    written, _ = await _bulk_update(client, tasks)
    assert written == len(tasks)

    assert cross_project_access_count() == rejected_before
    _assert_no_cross_project_calls(transport, projects)
    owner = _index_owner(projects)
    for index, doc_id, doc in transport.writes:
        slug = owner[index]
        assert doc_id.startswith(f'{slug}-'), f'{doc_id} written into {slug} index {index}'
        # Classified against its own project's catalog, never alpha's
        # process-wide list: the name and registry id are that project's.
        assert doc['class_name'] == CLASSES[slug][-1]
    for img_id, shown in vlm.prompts:
        assert shown == CLASSES[img_id.split('-')[0]], f'{img_id} saw {shown}'


@pytest.mark.asyncio
async def test_detection_writer_refuses_an_unowned_task(
    world: tuple[AsyncOpenSearch, _RecordingTransport],
) -> None:
    client, transport = world
    task = _ItemTask(
        crop_id='alpha-0',
        image_path='',
        item_bbox_norm=(0.1, 0.1, 0.5, 0.5),
        region_status='pending_detection',
        class_name='',
    )
    task.update_doc = {get_region_fields().status: 'detected'}
    with pytest.raises(ValueError, match='has no project'):
        await _bulk_update(client, [task])
    assert transport.writes == []


@pytest.mark.asyncio
async def test_guard_rejects_a_cross_project_write(
    projects: dict[str, ProjectRecord],
    world: tuple[AsyncOpenSearch, _RecordingTransport],
) -> None:
    """Control: the harness really is guarded -- beta's binding cannot
    write alpha's index."""
    from src.services.projects.guard import CrossProjectAccess

    client, transport = world
    with bind_project(projects['beta']), pytest.raises(CrossProjectAccess):
        await client.search(index=_items_index(projects['alpha']), body={'size': 1})
    assert transport.calls == []


# --- VLM worker and cluster-refresh daemon (HTTP to the API) ----------------


class _Registry(_Snapshot):
    async def ensure_fresh(self) -> None:
        return None


class _Api:
    """The curation API as the HTTP workers see it: records every request
    and, for ``label_batch``, marks the crops done in that project's store
    (as the real route's write would)."""

    def __init__(self, projects: dict[str, ProjectRecord], transport: _RecordingTransport) -> None:
        self.projects = projects
        self.transport = transport
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b'{}')
        self.requests.append((request.url.path, body))
        slug = request.url.path.split('/projects/', 1)[1].split('/', 1)[0]
        if request.url.path.endswith('/vlm/label_batch'):
            store = self.transport.docs[_items_index(self.projects[slug])]
            for cid in body['crop_ids']:
                store[cid]['vlm_done'] = True
            return httpx.Response(200, json={'predicted': len(body['crop_ids']), 'updated': 0})
        return httpx.Response(200, json={})


def _http_world(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    projects: dict[str, ProjectRecord],
) -> tuple[_RecordingTransport, _Api]:
    docs = {
        _items_index(rec): {
            f'{slug}-{i}': {'crop_id': f'{slug}-{i}', 'pe_embedding': [0.0]} for i in range(9)
        }
        for slug, rec in projects.items()
    }
    transport = _RecordingTransport(docs, lambda src: not src.get('vlm_done'))
    registry = _Registry(list(projects.values()))

    def _client(*_a: Any, **_kw: Any) -> AsyncOpenSearch:
        client = AsyncOpenSearch(hosts=['http://fake:9200'])
        client.transport = transport
        install_project_guard(client, registry)
        return client

    api = _Api(projects, transport)
    real_async_client = httpx.AsyncClient

    def _http_client(*_a: Any, **kw: Any) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(api.handler), **kw)

    monkeypatch.setattr(module, 'make_script_opensearch', _client)
    monkeypatch.setattr(module, 'script_project_registry', lambda *_a, **_kw: registry)
    monkeypatch.setattr(module.httpx, 'AsyncClient', _http_client)
    return transport, api


def _assert_requests_scoped(api: _Api, projects: dict[str, ProjectRecord]) -> None:
    assert api.requests, 'the worker never called the API'
    for path, body in api.requests:
        slug = path.split('/projects/', 1)[1].split('/', 1)[0]
        assert slug in projects, path
        for cid in body.get('crop_ids', []):
            assert cid.startswith(f'{slug}-'), f'{cid} sent to {path}'


@pytest.mark.asyncio
async def test_vlm_worker_reads_and_labels_each_project_in_its_own_scope(
    projects: dict[str, ProjectRecord], monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.curation import vlm_worker

    transport, api = _http_world(monkeypatch, vlm_worker, projects)

    async def _vlm_refresh(_client: Any) -> None:
        return None

    monkeypatch.setattr('src.services.labeling.vlm_endpoints.refresh_vlm_state', _vlm_refresh)
    monkeypatch.setattr('src.services.labeling.vlm_endpoints.vlm_configured', lambda: True)

    async def _no_heartbeat(*_a: Any, **_kw: Any) -> None:
        return None

    monkeypatch.setattr(vlm_worker, 'heartbeat_loop', _no_heartbeat)
    rejected_before = cross_project_access_count()
    args = vlm_worker.parse_args(
        ['--until-empty', '--vlm-batch-size', '4', '--concurrency', '2', '--poll-interval', '0']
    )
    assert await vlm_worker.run(args) == 0

    assert cross_project_access_count() == rejected_before
    _assert_no_cross_project_calls(transport, projects)
    _assert_requests_scoped(api, projects)
    labelled = {cid for _, body in api.requests for cid in body['crop_ids']}
    every_crop = {cid for store in transport.docs.values() for cid in store}
    assert labelled == every_crop, 'every project was drained'


@pytest.mark.asyncio
async def test_cluster_refresh_counts_and_refreshes_each_project_in_its_own_scope(
    projects: dict[str, ProjectRecord], monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.curation import cluster_refresh_daemon

    transport, api = _http_world(monkeypatch, cluster_refresh_daemon, projects)
    monkeypatch.setattr(cluster_refresh_daemon, 'write_heartbeat', lambda *_a, **_kw: None)
    monkeypatch.setattr('sys.argv', ['cluster_refresh_daemon', '--once'])
    rejected_before = cross_project_access_count()
    assert await cluster_refresh_daemon.run(cluster_refresh_daemon._parse_args()) == 0

    assert cross_project_access_count() == rejected_before
    _assert_no_cross_project_calls(transport, projects)
    _assert_requests_scoped(api, projects)
    promoted = {path.split('/projects/', 1)[1].split('/', 1)[0] for path, _ in api.requests}
    assert promoted == set(SLUGS), 'every project gets its own refresh'
