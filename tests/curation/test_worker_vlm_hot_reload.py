"""The detection worker follows a project's VLM at quiesce points (W9.3).

The worker is a separate process: its stores start cold, are pinned (a
change is applied only at a quiesce point, after the queues drained), and ONE
pinned registry store is shared by every project. These tests build that
state from what the API process wrote to OpenSearch, never from the API's own
in-memory snapshot."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from curation._fake_config_opensearch import FakeConfigOpenSearch
from curation.conftest import good_probe
from scripts.curation.worker.runtime import (
    RuntimeHolder,
    current_want,
    maybe_hot_reload,
    quiesce_and_swap,
)
from scripts.curation.worker.state import _ItemTask
from src.config.curation import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.config_store import get_config_store, get_global_config_store
from src.services.config_store.global_store import reset_global_config_store
from src.services.config_store.index import activate
from src.services.config_store.store import reset_config_stores
from src.services.config_store.vlm_endpoints import record_probe, save_endpoint
from src.services.detection import profile_registry
from src.services.labeling.vlm_client import VlmIdentity
from src.services.labeling.vlm_endpoint_body import VlmEndpointBody
from src.services.labeling.vlm_endpoints import VlmEndpointUnavailableError, active_vlm_endpoint
from src.services.labeling.vlm_prompts import resolve_prompt_pack


pytestmark = pytest.mark.usefixtures('reference_region_profile', 'vlm_off_env')


@pytest.fixture
def vlm_off_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_VLM_URL', raising=False)


def _record(slug: str) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base_curation_config()),
    )


def _body(model: str = 'm') -> VlmEndpointBody:
    return VlmEndpointBody(base_url='http://vlm:8000/v1', model=model)


async def _endpoint(client: Any, name: str, *, root: str, model: str = 'm') -> None:
    body = _body(model)
    await save_endpoint(client, name=name, body=body, expected_revision=None)
    await record_probe(client, name=name, revision=1, body=body, record=good_probe(root=root))


async def _activate_in(
    client: Any,
    record: ProjectRecord,
    name: str | None,
    revision: int | None,
    expected: dict[str, Any] | None,
) -> None:
    """What ANOTHER process (the API) does: write the activation doc."""
    with bind_project(record):
        await activate(
            client,
            get_config_store().index,
            axis='vlm',
            name=name,
            revision=revision,
            expected_active=expected,
            doc_fields={'acked_refs': {}, 'external_ack_at': None},
        )


def _go_cold() -> None:
    """A fresh worker process: nothing this process loaded so far exists."""
    reset_config_stores()
    reset_global_config_store()


class Worker:
    """One worker's collaborators, mirroring ``runner.run``."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.holder = RuntimeHolder()
        self.registry = get_global_config_store(mode='pinned')
        self.stores: dict[str, Any] = {}
        self.built: list[tuple[str, str]] = []  # (endpoint ref, model) per build
        self.detector = MagicMock()
        self.profile = profile_registry.get_active_region_profile()
        self.pack = resolve_prompt_pack()

    def _build_vlm(self, endpoint: Any, _pack: Any) -> Any:
        self.built.append((endpoint.ref, endpoint.model_id))
        vlm = MagicMock()
        vlm.aclose = AsyncMock()
        vlm.identity = VlmIdentity(endpoint.ref, endpoint.model_id)
        return vlm

    async def cycle(self, record: ProjectRecord) -> Any:
        with bind_project(record):
            store = self.stores.setdefault(record.slug, get_config_store(mode='pinned'))
            return await maybe_hot_reload(
                store=store,
                registry=self.registry,
                opensearch=self.client,
                holder=self.holder,
                slug=record.slug,
                pool=MagicMock(),
                args=MagicMock(segmenter_url='', vlm_url=''),
                queues=[],
                get_active_profile=lambda: self.profile,
                get_active_pack=lambda: self.pack,
                get_active_vlm=active_vlm_endpoint,
                region_detector_cls=MagicMock(),
                ocr_recognizer_cls=MagicMock(),
                segmenter_cls=MagicMock(),
                build_vlm=self._build_vlm,
            )


# ---- a cold worker resolves the VLM from what the API wrote -------------------


@pytest.mark.asyncio
async def test_a_cold_worker_runs_the_activated_endpoint_at_its_pinned_revision() -> None:
    client = FakeConfigOpenSearch()
    alpha = _record('alpha')
    await _endpoint(client, 'one', root='org/model-one')
    await _activate_in(client, alpha, 'one', 1, None)
    _go_cold()

    worker = Worker(client)
    runtime = await worker.cycle(alpha)

    assert runtime is not None
    assert runtime.vlm is not None
    assert worker.built == [('one@1', 'org/model-one')]  # the probe's root, not the alias
    assert runtime.vlm_identity == VlmIdentity('one@1', 'org/model-one')
    assert runtime.vlm_model == 'org/model-one'


@pytest.mark.asyncio
async def test_a_project_that_never_activated_one_runs_the_env_builtin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm:8000/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'local-vlm')
    client = FakeConfigOpenSearch()
    _go_cold()
    worker = Worker(client)
    runtime = await worker.cycle(_record('alpha'))
    assert runtime is not None
    assert runtime.vlm is not None
    assert worker.built[0][0].startswith('env@')
    assert runtime.vlm_identity.model == 'local-vlm'


@pytest.mark.asyncio
async def test_an_explicit_off_and_no_configuration_run_without_a_vlm() -> None:
    client = FakeConfigOpenSearch()
    alpha = _record('alpha')
    await _endpoint(client, 'one', root='org/one')
    _go_cold()
    worker = Worker(client)
    runtime = await worker.cycle(alpha)  # nothing activated, no OP_VLM_URL
    assert runtime is not None
    assert runtime.vlm is None
    assert runtime.vlm_identity is None
    assert runtime.vlm_available is False

    await _activate_in(client, alpha, 'one', 1, None)
    runtime = await worker.cycle(alpha)
    assert runtime is not None
    assert runtime.vlm is not None
    await _activate_in(client, alpha, None, None, {'name': 'one', 'revision': 1})
    runtime = await worker.cycle(alpha)
    assert runtime is not None
    assert runtime.vlm is None


# ---- swapping ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_switching_endpoints_swaps_once_and_closes_the_old_labeler() -> None:
    client = FakeConfigOpenSearch()
    alpha = _record('alpha')
    await _endpoint(client, 'one', root='org/one')
    await _endpoint(client, 'two', root='org/two')
    await _activate_in(client, alpha, 'one', 1, None)
    _go_cold()
    worker = Worker(client)
    first = await worker.cycle(alpha)
    assert first is not None
    old_vlm = first.vlm

    for _ in range(3):  # no change: no rebuild
        await worker.cycle(alpha)
    assert len(worker.built) == 1

    await _activate_in(client, alpha, 'two', 1, {'name': 'one', 'revision': 1})
    for _ in range(3):
        swapped = await worker.cycle(alpha)
    assert worker.built == [('one@1', 'org/one'), ('two@1', 'org/two')]
    assert swapped is not first
    assert swapped.vlm_identity == VlmIdentity('two@1', 'org/two')
    old_vlm.aclose.assert_awaited_once()
    swapped.vlm.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_saving_a_new_revision_of_the_active_endpoint_changes_nothing() -> None:
    client = FakeConfigOpenSearch()
    alpha = _record('alpha')
    await _endpoint(client, 'one', root='org/one')
    await _activate_in(client, alpha, 'one', 1, None)
    _go_cold()
    worker = Worker(client)
    await worker.cycle(alpha)
    await save_endpoint(client, name='one', body=_body('edited'), expected_revision=1)
    for _ in range(3):
        runtime = await worker.cycle(alpha)
    assert worker.built == [('one@1', 'org/one')]
    assert runtime.vlm_identity == VlmIdentity('one@1', 'org/one')


@pytest.mark.asyncio
async def test_a_rebuild_after_an_edit_still_uses_the_revision_that_was_activated() -> None:
    """Something else forces a rebuild (a probe of the EDITED body lands):
    the labeler is built from the pinned revision 1, never from the newer
    revision 2, and revision 2's probe is not attributed to revision 1."""
    client = FakeConfigOpenSearch()
    alpha = _record('alpha')
    await _endpoint(client, 'one', root='org/one')
    await _activate_in(client, alpha, 'one', 1, None)
    _go_cold()
    worker = Worker(client)
    await worker.cycle(alpha)
    edited = _body('edited')
    await save_endpoint(client, name='one', body=edited, expected_revision=1)
    await record_probe(
        client,
        name='one',
        revision=2,
        body=edited,
        record=good_probe(root='org/edited', probed_at='2026-09-28T16:00:00+00:00'),
    )
    for _ in range(2):
        runtime = await worker.cycle(alpha)
    assert worker.built[-1] == ('one@1', 'org/one')
    assert runtime.vlm_endpoint.body.model == 'm'


@pytest.mark.asyncio
async def test_a_reprobe_that_changes_the_model_root_rebuilds_the_labeler() -> None:
    client = FakeConfigOpenSearch()
    alpha = _record('alpha')
    await _endpoint(client, 'one', root='org/before')
    await _activate_in(client, alpha, 'one', 1, None)
    _go_cold()
    worker = Worker(client)
    await worker.cycle(alpha)
    await record_probe(
        client,
        name='one',
        revision=1,
        body=_body(),
        record=good_probe(root='org/after', probed_at='2026-09-28T13:00:00+00:00'),
    )
    for _ in range(2):
        runtime = await worker.cycle(alpha)
    assert worker.built == [('one@1', 'org/before'), ('one@1', 'org/after')]
    assert runtime.vlm_model == 'org/after'


@pytest.mark.asyncio
async def test_an_unresolvable_pinned_revision_fails_closed_and_keeps_the_old_runtime() -> None:
    client = FakeConfigOpenSearch()
    alpha = _record('alpha')
    await _endpoint(client, 'one', root='org/one')
    await _endpoint(client, 'two', root='org/two')
    await _activate_in(client, alpha, 'one', 1, None)
    _go_cold()
    worker = Worker(client)
    first = await worker.cycle(alpha)
    await _activate_in(client, alpha, 'two', 1, {'name': 'one', 'revision': 1})
    # the immutable copy the activation points at is gone
    from src.services.config_store.index import config_doc_id

    del client._docs[get_global_config_store().index][config_doc_id('vlm_endpoint', 'two', 1)]
    _go_cold_registry_cache()

    with pytest.raises(VlmEndpointUnavailableError):
        await worker.cycle(alpha)
    assert worker.holder.get('alpha') is first  # never a silent fall back to "no VLM"
    assert worker.built == [('one@1', 'org/one')]


def _go_cold_registry_cache() -> None:
    from src.services.config_store.vlm_snapshot import reset_vlm_revision_cache

    reset_vlm_revision_cache()


# ---- one pinned registry, many projects ---------------------------------------


@pytest.mark.asyncio
async def test_one_projects_activation_does_not_swap_another_projects_runtime() -> None:
    client = FakeConfigOpenSearch()
    alpha, beta = _record('alpha'), _record('beta')
    await _endpoint(client, 'one', root='org/one')
    await _endpoint(client, 'two', root='org/two')
    await _activate_in(client, alpha, 'one', 1, None)
    await _activate_in(client, beta, 'one', 1, None)
    _go_cold()
    worker = Worker(client)
    await worker.cycle(alpha)
    await worker.cycle(beta)
    assert len(worker.built) == 2

    await _activate_in(client, alpha, 'two', 1, {'name': 'one', 'revision': 1})
    await worker.cycle(alpha)
    await worker.cycle(beta)
    assert worker.built == [('one@1', 'org/one')] * 2 + [('two@1', 'org/two')]
    alpha_rt, beta_rt = worker.holder.get('alpha'), worker.holder.get('beta')
    assert alpha_rt is not None
    assert beta_rt is not None
    assert alpha_rt.vlm_identity == VlmIdentity('two@1', 'org/two')
    assert beta_rt.vlm_identity == VlmIdentity('one@1', 'org/one')


@pytest.mark.asyncio
async def test_a_reprobe_of_a_shared_endpoint_reaches_every_project_running_it() -> None:
    client = FakeConfigOpenSearch()
    alpha, beta = _record('alpha'), _record('beta')
    await _endpoint(client, 'shared', root='org/v1')
    await _activate_in(client, alpha, 'shared', 1, None)
    await _activate_in(client, beta, 'shared', 1, None)
    _go_cold()
    worker = Worker(client)
    await worker.cycle(alpha)
    await worker.cycle(beta)
    await record_probe(
        client,
        name='shared',
        revision=1,
        body=_body(),
        record=good_probe(root='org/v2', probed_at='2026-09-28T14:00:00+00:00'),
    )
    await worker.cycle(alpha)
    await worker.cycle(beta)
    alpha_rt, beta_rt = worker.holder.get('alpha'), worker.holder.get('beta')
    assert alpha_rt is not None
    assert beta_rt is not None
    assert alpha_rt.vlm_model == 'org/v2'
    assert beta_rt.vlm_model == 'org/v2'


@pytest.mark.asyncio
async def test_the_want_carries_the_vlm_ref_and_the_probe_marker() -> None:
    client = FakeConfigOpenSearch()
    alpha = _record('alpha')
    await _endpoint(client, 'one', root='org/one')
    await _activate_in(client, alpha, 'one', 1, None)
    _go_cold()
    worker = Worker(client)
    await worker.cycle(alpha)
    with bind_project(alpha):
        store = worker.stores['alpha']
        _profile, _pack, vlm_ref, marker = current_want(store, worker.registry)
    assert vlm_ref == ('one', 1)
    assert marker is not None


@pytest.mark.asyncio
async def test_the_swap_pins_the_registry_together_with_the_project_store() -> None:
    """If only the project store were pinned, the worker would resolve the
    new activation against the OLD registry snapshot (a stale probe)."""
    client = FakeConfigOpenSearch()
    alpha = _record('alpha')
    await _endpoint(client, 'one', root='org/before')
    await _activate_in(client, alpha, 'one', 1, None)
    _go_cold()
    registry = get_global_config_store(mode='pinned')
    with bind_project(alpha):
        store = get_config_store(mode='pinned')
        await store.refresh(client)
        await registry.refresh(client)
        store.pin_active()
        registry.pin_active()
        await record_probe(
            client,
            name='one',
            revision=1,
            body=_body(),
            record=good_probe(root='org/after', probed_at='2026-09-28T15:00:00+00:00'),
        )
        await store.refresh(client)
        await registry.refresh(client)
        assert registry.pending_snapshot is not None
        seen: list[Any] = []

        def capture_active_vlm() -> Any:
            seen.append(active_vlm_endpoint())
            return seen[-1]

        holder = RuntimeHolder()
        await quiesce_and_swap(
            queues=[],
            holder=holder,
            slug='alpha',
            store=store,
            registry=registry,
            pool=MagicMock(),
            args=MagicMock(segmenter_url='', vlm_url=''),
            want=current_want(store, registry),
            get_active_profile=profile_registry.get_active_region_profile,
            get_active_pack=resolve_prompt_pack,
            get_active_vlm=capture_active_vlm,
            region_detector_cls=MagicMock(),
            ocr_recognizer_cls=MagicMock(),
            segmenter_cls=MagicMock(),
            build_vlm=lambda _endpoint, _pack: MagicMock(),
        )
    assert seen[0].model_id == 'org/after'
    assert registry.pending_snapshot is None


# ---- per-task identity -----------------------------------------------------------


def _task() -> _ItemTask:
    return _ItemTask(
        project=_record('alpha'),
        crop_id='c1',
        image_path='/dev/null',
        item_bbox_norm=(0.1, 0.1, 0.5, 0.5),
        region_status='pending_detection',
        class_name='',
    )


@pytest.mark.asyncio
async def test_a_write_carries_the_identity_of_the_call_not_of_the_runtime_at_write_time() -> None:
    """An item called the old endpoint, then a swap happened before its
    write flushed: the write must still name the endpoint that answered."""
    import scripts.curation.region_worker_main as worker_main
    from curation.occ_fakes import make_bulk_response, make_bulk_update_item, make_mget_response
    from src.config import get_region_fields

    F = get_region_fields()
    task = _task()
    task.update_doc = {F.status: 'detected', F.count: 1}
    task.mark_vlm_called(VlmIdentity('old@3', 'org/old-model'))

    async def _mget(*, body: dict[str, Any]) -> dict[str, Any]:
        return make_mget_response({d['_id']: {F.status: 'pending_detection'} for d in body['docs']})

    async def _bulk(*, body: list[dict[str, Any]], **_kw: Any) -> dict[str, Any]:
        return make_bulk_response(
            [make_bulk_update_item(a['update']['_id'], status=200) for a in body[0::2]]
        )

    opensearch = MagicMock()
    opensearch.mget = AsyncMock(side_effect=_mget)
    opensearch.bulk = AsyncMock(side_effect=_bulk)
    written, _ = await worker_main._bulk_update(opensearch, [task])
    assert written == 1
    assert opensearch.bulk.await_args is not None
    doc = opensearch.bulk.await_args.kwargs['body'][1]['doc']
    assert doc['vlm_endpoint'] == 'old@3'
    assert doc['vlm_model'] == 'org/old-model'


@pytest.mark.asyncio
async def test_an_item_that_never_called_the_vlm_is_stamped_with_no_endpoint() -> None:
    import scripts.curation.region_worker_main as worker_main
    from curation.occ_fakes import make_bulk_response, make_bulk_update_item, make_mget_response
    from src.config import get_region_fields

    F = get_region_fields()
    task = _task()
    task.update_doc = {F.status: 'detected', F.count: 1}

    async def _mget(*, body: dict[str, Any]) -> dict[str, Any]:
        return make_mget_response({d['_id']: {F.status: 'pending_detection'} for d in body['docs']})

    async def _bulk(*, body: list[dict[str, Any]], **_kw: Any) -> dict[str, Any]:
        return make_bulk_response(
            [make_bulk_update_item(a['update']['_id'], status=200) for a in body[0::2]]
        )

    opensearch = MagicMock()
    opensearch.mget = AsyncMock(side_effect=_mget)
    opensearch.bulk = AsyncMock(side_effect=_bulk)
    await worker_main._bulk_update(opensearch, [task])
    assert opensearch.bulk.await_args is not None
    doc = opensearch.bulk.await_args.kwargs['body'][1]['doc']
    assert 'vlm_endpoint' not in doc
    assert 'vlm_model' not in doc


@pytest.mark.asyncio
async def test_probing_a_newer_revision_does_not_change_the_running_revisions_marker() -> None:
    """The probe of ``one@2`` is not the probe of the ``one@1`` this project
    runs, so it must not make the worker rebuild (W9 review M3)."""
    client = FakeConfigOpenSearch()
    alpha = _record('alpha')
    await _endpoint(client, 'one', root='org/one')
    await _activate_in(client, alpha, 'one', 1, None)
    _go_cold()
    worker = Worker(client)
    await worker.cycle(alpha)
    with bind_project(alpha):
        before = current_want(worker.stores['alpha'], worker.registry)
    edited = _body('edited')
    await save_endpoint(client, name='one', body=edited, expected_revision=1)
    await record_probe(
        client,
        name='one',
        revision=2,
        body=edited,
        record=good_probe(root='org/edited', probed_at='2026-09-28T16:00:00+00:00'),
    )
    for _ in range(2):
        await worker.cycle(alpha)
    with bind_project(alpha):
        after = current_want(worker.stores['alpha'], worker.registry)
    assert after == before
    assert len(worker.built) == 1
