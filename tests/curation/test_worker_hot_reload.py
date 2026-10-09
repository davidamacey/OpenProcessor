"""W2 sec 4.5: the detection worker's hot-reloadable runtime --
``scripts/curation/worker/runtime.py`` (``build_runtime``,
``RuntimeHolder``, ``quiesce_and_swap``), and the write-path stamping
(``RegionFields.profile``/``profile_revision``, ``vlm_prompt_pack``) in
``scripts/curation/worker/bulk_writer.py``."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import scripts.curation.region_worker_main as worker
from curation._fake_config_opensearch import FakeConfigOpenSearch, TwoProjectOpenSearch
from curation.occ_fakes import make_bulk_response, make_bulk_update_item, make_mget_response
from scripts.curation.worker.state import _ItemTask
from src.config import get_region_fields
from src.config.curation import base_curation_config
from src.config.project_context import bind_project, current_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.config_store import get_global_config_store
from src.services.config_store.index import activate, save_config
from src.services.config_store.store import get_config_store, reset_config_stores
from src.services.detection import profile_registry


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


pytestmark = pytest.mark.usefixtures('reference_region_profile')


def _record(slug: str, revision: int = 1) -> ProjectRecord:
    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=revision,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base_curation_config()),
    )


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    reset_config_stores()
    yield
    reset_config_stores()


# =============================================================================
# RuntimeHolder isolation: "alpha activation does not swap beta"
# =============================================================================


@pytest.mark.asyncio
async def test_alpha_activation_does_not_swap_beta() -> None:
    """Activating a profile in project alpha's config store must not
    make ``config_wants_swap`` report ``True`` for beta's store -- each
    project's activation is tracked in its own, independently-refreshed
    ``ConfigStore`` (keyed by slug), so a change in one is invisible to
    the other."""
    from scripts.curation.worker.runtime import current_want

    client = FakeConfigOpenSearch()
    alpha = _record('alpha')
    beta = _record('beta')
    registry = get_global_config_store(mode='pinned')

    with bind_project(alpha):
        alpha_store = get_config_store(mode='pinned')
        await alpha_store.refresh(client)
    with bind_project(beta):
        beta_store = get_config_store(mode='pinned')
        await beta_store.refresh(client)
    await registry.refresh(client)

    baseline_alpha = current_want(alpha_store, registry)
    baseline_beta = current_want(beta_store, registry)
    assert baseline_alpha[:2] == baseline_beta[:2] == (None, None)

    with bind_project(alpha):
        idx = alpha_store.index
        doc = await save_config(
            client, idx, kind='region_profile', name='wheel', body={}, expected_revision=None
        )
        await activate(
            client,
            idx,
            axis='detection_profile',
            name='wheel',
            revision=doc['revision'],
            expected_active=None,
        )
        await alpha_store.refresh(client)
        alpha_store.pin_active()

    # Alpha's own store now wants a swap...
    assert current_want(alpha_store, registry) != baseline_alpha
    # ...but beta's store, never refreshed against alpha's index, is untouched.
    await beta_store.refresh(client)
    assert current_want(beta_store, registry) == baseline_beta


# =============================================================================
# build_runtime / quiesce_and_swap
# =============================================================================


def _fake_args() -> Any:
    ns = MagicMock()
    ns.segmenter_url = ''
    return ns


def _fake_ctors() -> dict[str, Any]:
    """Stand-ins for the four heavy-IO constructors ``build_runtime``
    now takes as parameters (never imports fresh) -- mirrors how
    ``runner.py`` passes its own module-level names, which is exactly
    what lets a real monkeypatch on those names reach a rebuild."""
    return {
        'region_detector_cls': MagicMock(),
        'ocr_recognizer_cls': MagicMock(),
        'segmenter_cls': MagicMock(),
        'build_vlm': MagicMock(),
    }


def _swap_kw() -> dict[str, Any]:
    """The extra collaborators a swap needs: the pinned deployment-wide
    registry store and the active-endpoint resolver (none active here)."""
    return {
        'registry': get_global_config_store(mode='pinned'),
        'get_active_vlm': lambda: None,
        **_fake_ctors(),
    }


@pytest.mark.asyncio
async def test_build_runtime_returns_refs() -> None:
    from scripts.curation.worker.runtime import build_runtime

    profile = profile_registry.get_active_region_profile()
    assert profile is not None
    from src.services.labeling.vlm_prompts import resolve_prompt_pack

    pack = resolve_prompt_pack()

    rt = await build_runtime(
        pool=MagicMock(),
        profile=profile,
        pack=pack,
        args=_fake_args(),
        profile_revision=3,
        pack_revision=None,
        vlm_endpoint=None,
        **_fake_ctors(),
    )
    assert rt.profile_ref == (profile.name, 3)
    assert rt.pack_ref == (pack.name, None)
    assert rt.detector is not None
    assert rt.vlm is None  # no active endpoint


@pytest.mark.asyncio
async def test_build_runtime_uses_the_passed_in_constructors_not_fresh_imports() -> None:
    """The monkeypatch-target fix: build_runtime must call the exact
    classes it was handed, never import its own copies of
    RegionDetector/PaddleOcrTextRecognizer/SegmenterClient/build_vlm_labeler --
    otherwise a test (or runner.py's producer loop rebuilding through a
    module-level name a test patched) silently keeps talking to the
    real, unpatched class."""
    from scripts.curation.worker.runtime import build_runtime

    profile = profile_registry.get_active_region_profile()
    assert profile is not None
    from src.services.labeling.vlm_prompts import resolve_prompt_pack

    pack = resolve_prompt_pack()
    ctors = _fake_ctors()
    args = _fake_args()
    endpoint = MagicMock()
    pool = MagicMock()

    rt = await build_runtime(
        pool=pool, profile=profile, pack=pack, args=args, vlm_endpoint=endpoint, **ctors
    )

    ctors['region_detector_cls'].assert_called_once_with(pool, profile)
    ctors['ocr_recognizer_cls'].assert_called_once_with(pool, profile)
    ctors['segmenter_cls'].assert_called_once()
    ctors['build_vlm'].assert_called_once_with(endpoint, pack)
    assert rt.detector is ctors['region_detector_cls'].return_value
    assert rt.vlm is ctors['build_vlm'].return_value
    assert rt.vlm_endpoint is endpoint


@pytest.mark.asyncio
async def test_quiesce_and_swap_drains_queues_before_building() -> None:
    from scripts.curation.worker.runtime import RuntimeHolder, current_want, quiesce_and_swap

    profile = profile_registry.get_active_region_profile()
    assert profile is not None
    from src.services.labeling.vlm_prompts import resolve_prompt_pack

    pack = resolve_prompt_pack()

    q: asyncio.Queue[Any] = asyncio.Queue()
    await q.put('item')
    holder = RuntimeHolder()
    drained_before_build = False

    async def _drain_soon() -> None:
        nonlocal drained_before_build
        await asyncio.sleep(0.01)
        drained_before_build = True
        q.task_done()

    project = _record('alpha')
    with bind_project(project):
        store = get_config_store(mode='pinned')

        asyncio.get_event_loop().create_task(_drain_soon())
        rt = await quiesce_and_swap(
            queues=[q],
            holder=holder,
            slug='alpha',
            store=store,
            pool=MagicMock(),
            args=_fake_args(),
            want=current_want(store, get_global_config_store(mode='pinned')),
            get_active_profile=lambda: profile,
            get_active_pack=lambda: pack,
            **_swap_kw(),
        )
    assert drained_before_build
    assert holder.get('alpha') is rt
    assert holder.get('beta') is None


@pytest.mark.asyncio
async def test_quiesce_and_swap_pins_before_resolving_active_profile() -> None:
    """B2: pin_active() must run strictly between the drain and the
    build, so `get_active_profile`/`get_active_pack` (which read
    `store.current`) already see the newly-pinned snapshot -- not the
    stale one from before this cycle's refresh."""
    from scripts.curation.worker.runtime import RuntimeHolder, current_want, quiesce_and_swap

    client = FakeConfigOpenSearch()
    project = _record('alpha')
    profile = profile_registry.get_active_region_profile()
    assert profile is not None

    with bind_project(project):
        store = get_config_store(mode='pinned')
        idx = store.index
        doc = await save_config(
            client, idx, kind='region_profile', name=profile.name, body={}, expected_revision=None
        )
        await activate(
            client,
            idx,
            axis='detection_profile',
            name=profile.name,
            revision=doc['revision'],
            expected_active=None,
        )
        await store.refresh(client)
        assert store.pending_snapshot is not None
        assert store.current.active_profile is None  # not pinned yet

        seen_active_profile_at_call_time: list[Any] = []

        def _get_active_profile() -> Any:
            seen_active_profile_at_call_time.append(store.current.active_profile)
            return profile

        holder = RuntimeHolder()
        await quiesce_and_swap(
            queues=[],
            holder=holder,
            slug='alpha',
            store=store,
            pool=MagicMock(),
            args=_fake_args(),
            want=current_want(store, get_global_config_store(mode='pinned')),
            get_active_profile=_get_active_profile,
            get_active_pack=lambda: profile_registry_pack(),
            **_swap_kw(),
        )
        # By the time get_active_profile() ran, the store had already
        # been pinned to the new activation.
        assert seen_active_profile_at_call_time == [(profile.name, doc['revision'])]
        assert store.pending_snapshot is None


def profile_registry_pack() -> Any:
    from src.services.labeling.vlm_prompts import resolve_prompt_pack

    return resolve_prompt_pack()


# =============================================================================
# maybe_hot_reload: the producer-loop swap check -- pointer 2
# =============================================================================


@pytest.mark.asyncio
async def test_maybe_hot_reload_never_swaps_when_activation_is_unchanged() -> None:
    """No activation change across repeated cycles must never rebuild --
    the swap decision compares the store's served
    ``(profile, pack)`` ``AxisRef`` pair, never object identity or the
    runtime's own always-populated ``profile_ref``/``pack_ref``."""
    from scripts.curation.worker.runtime import RuntimeHolder, maybe_hot_reload

    client = FakeConfigOpenSearch()
    project = _record('steady-project')
    profile = profile_registry.get_active_region_profile()
    assert profile is not None
    from src.services.labeling.vlm_prompts import resolve_prompt_pack

    pack = resolve_prompt_pack()
    ctors = _swap_kw()

    with bind_project(project):
        store = get_config_store(mode='pinned')
        holder = RuntimeHolder()

        for _ in range(3):
            await maybe_hot_reload(
                store=store,
                opensearch=client,
                holder=holder,
                slug='steady-project',
                pool=MagicMock(),
                args=_fake_args(),
                queues=[],
                get_active_profile=lambda: profile,
                get_active_pack=lambda: pack,
                **ctors,
            )

        # First cycle builds once (nothing synced yet, store shows
        # nothing active either -> the "off/off" baseline still counts
        # as a change from "never synced"); every later cycle, with the
        # store unchanged, must not call any constructor again.
        assert ctors['region_detector_cls'].call_count == 1


@pytest.mark.asyncio
async def test_maybe_hot_reload_swaps_exactly_once_on_a_real_pinned_activation() -> None:
    """B2 (reviewer probe #3, reproduced): pinned store, baseline sync,
    then save+activate a profile, then 3 `maybe_hot_reload` cycles. Must
    build exactly once (on the cycle right after the activation), not
    zero times -- a pinned store's `refresh()` only stages
    `pending_snapshot`; the pre-fix code compared `store.current`
    (unmoved) against the last-synced refs and never saw the change."""
    from scripts.curation.worker.runtime import RuntimeHolder, maybe_hot_reload

    client = FakeConfigOpenSearch()
    project = _record('pinned-project')
    profile = profile_registry.get_active_region_profile()
    assert profile is not None
    from src.services.labeling.vlm_prompts import resolve_prompt_pack

    pack = resolve_prompt_pack()
    ctors = _swap_kw()

    with bind_project(project):
        store = get_config_store(mode='pinned')
        holder = RuntimeHolder()

        # Baseline cycle: nothing activated yet.
        await maybe_hot_reload(
            store=store,
            opensearch=client,
            holder=holder,
            slug='pinned-project',
            pool=MagicMock(),
            args=_fake_args(),
            queues=[],
            get_active_profile=lambda: profile,
            get_active_pack=lambda: pack,
            **ctors,
        )
        assert ctors['region_detector_cls'].call_count == 1  # env default, first-ever build

        idx = store.index
        doc = await save_config(
            client, idx, kind='region_profile', name='wheel', body={}, expected_revision=None
        )
        await activate(
            client,
            idx,
            axis='detection_profile',
            name='wheel',
            revision=doc['revision'],
            expected_active=None,
        )

        for _ in range(3):
            await maybe_hot_reload(
                store=store,
                opensearch=client,
                holder=holder,
                slug='pinned-project',
                pool=MagicMock(),
                args=_fake_args(),
                queues=[],
                get_active_profile=lambda: profile,
                get_active_pack=lambda: pack,
                **ctors,
            )

        # Exactly one more build across all 3 post-activation cycles.
        assert ctors['region_detector_cls'].call_count == 2
        assert store.current.active_profile == ('wheel', doc['revision'])


@pytest.mark.asyncio
async def test_build_uses_the_activated_pack_not_the_env_default() -> None:
    """B4: a pack activation must rebuild with THAT pack (via
    `active_prompt_pack`), never the env/file default (`resolve_prompt_pack`).
    The reviewer's probe: activate a stored pack named differently from
    the env default, swap, and check the built runtime's pack name and
    ref -- not just that *a* pack got attached."""
    from scripts.curation.worker.runtime import RuntimeHolder, maybe_hot_reload
    from src.services.labeling.vlm_prompts import active_prompt_pack, resolve_prompt_pack

    client = FakeConfigOpenSearch()
    project = _record('pack-project')
    profile = profile_registry.get_active_region_profile()
    assert profile is not None
    env_pack = resolve_prompt_pack()
    ctors = _swap_kw()

    with bind_project(project):
        store = get_config_store(mode='pinned')
        idx = store.index
        pack_body = {**env_pack.to_dict(), 'name': 'mypack'}
        doc = await save_config(
            client,
            idx,
            kind='prompt_pack',
            name='mypack',
            body=pack_body,
            expected_revision=None,
        )
        await activate(
            client,
            idx,
            axis='prompt_pack',
            name='mypack',
            revision=doc['revision'],
            expected_active=None,
        )

        holder = RuntimeHolder()
        new_rt = await maybe_hot_reload(
            store=store,
            opensearch=client,
            holder=holder,
            slug='pack-project',
            pool=MagicMock(),
            args=_fake_args(),
            queues=[],
            get_active_profile=lambda: profile,
            get_active_pack=active_prompt_pack,
            **ctors,
        )

    assert new_rt is not None
    assert new_rt.pack.name == 'mypack'
    assert new_rt.pack_ref == ('mypack', doc['revision'])


# =============================================================================
# Write-path stamping
# =============================================================================


def _make_task(*, crop_id: str, status: str | None = 'pending') -> _ItemTask:
    return _ItemTask(
        project=current_project().record,
        crop_id=crop_id,
        image_path='/dev/null/never-read',
        item_bbox_norm=(0.1, 0.1, 0.5, 0.5),
        region_status=status,
        class_name='audi',
    )


@pytest.mark.asyncio
async def test_bulk_write_stamps_region_profile_and_pack() -> None:
    F = get_region_fields()
    a = _make_task(crop_id='a')
    a.update_doc = {F.status: 'detected', F.max_score: 0.9}
    # Minor 5 (W2 review): the pack stamp is per-TASK, gated on whether a
    # VLM call actually contributed to this task's write this pass.
    a.vlm_called = True

    async def _fake_mget(*, body: dict[str, Any]) -> dict[str, Any]:
        found = {d['_id']: {F.status: 'pending'} for d in body['docs']}
        return make_mget_response(found)

    async def _fake_bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        items = [
            make_bulk_update_item(action['update']['_id'], status=200) for action in body[0::2]
        ]
        return make_bulk_response(items)

    opensearch = MagicMock()
    opensearch.mget = AsyncMock(side_effect=_fake_mget)
    opensearch.bulk = AsyncMock(side_effect=_fake_bulk)

    n_written, _ = await worker._bulk_update(opensearch, [a])
    assert n_written == 1

    assert opensearch.bulk.await_args is not None
    bulk_body = opensearch.bulk.await_args.kwargs['body']
    written_doc = bulk_body[1]['doc']
    active_profile = profile_registry.get_active_region_profile()
    assert active_profile is not None
    assert written_doc[F.profile] == active_profile.name
    # env-registered (never activated through the store) -- no revision.
    assert written_doc[F.profile_revision] is None
    assert written_doc['vlm_prompt_pack']


@pytest.mark.asyncio
async def test_bulk_write_does_not_stamp_pack_when_no_vlm_call_happened() -> None:
    """Minor 5 (W2 review): a write whose task never actually involved a
    VLM call (no VLM configured, or a write path that skipped it, e.g.
    the high-confidence secondary-segmenter auto-skip) must not carry
    ``vlm_prompt_pack`` -- that would claim a VLM ran when it didn't."""
    F = get_region_fields()
    a = _make_task(crop_id='a')
    a.update_doc = {F.status: 'detected', F.max_score: 0.9}
    assert a.vlm_called is False  # the default

    async def _fake_mget(*, body: dict[str, Any]) -> dict[str, Any]:
        found = {d['_id']: {F.status: 'pending'} for d in body['docs']}
        return make_mget_response(found)

    async def _fake_bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        items = [
            make_bulk_update_item(action['update']['_id'], status=200) for action in body[0::2]
        ]
        return make_bulk_response(items)

    opensearch = MagicMock()
    opensearch.mget = AsyncMock(side_effect=_fake_mget)
    opensearch.bulk = AsyncMock(side_effect=_fake_bulk)

    n_written, _ = await worker._bulk_update(opensearch, [a])
    assert n_written == 1

    assert opensearch.bulk.await_args is not None
    bulk_body = opensearch.bulk.await_args.kwargs['body']
    written_doc = bulk_body[1]['doc']
    assert 'vlm_prompt_pack' not in written_doc


@pytest.mark.asyncio
async def test_bulk_write_stamps_store_activated_profile_revision() -> None:
    """A profile activated through the store (with a revision) stamps
    that revision, not ``None`` -- distinguishing a stored activation
    from the env/file default."""
    F = get_region_fields()
    client = FakeConfigOpenSearch()
    project = _record('wheel-project')

    active_profile = profile_registry.get_active_region_profile()
    assert active_profile is not None

    with bind_project(project):
        store = get_config_store(mode='pinned')
        idx = store.index
        doc = await save_config(
            client,
            idx,
            kind='region_profile',
            name=active_profile.name,
            body={},
            expected_revision=None,
        )
        await activate(
            client,
            idx,
            axis='detection_profile',
            name=doc['name'],
            revision=doc['revision'],
            expected_active=None,
        )
        await store.refresh(client)
        store.pin_active()

        a = _make_task(crop_id='a')
        a.update_doc = {F.status: 'detected', F.max_score: 0.9}

        async def _fake_mget(*, body: dict[str, Any]) -> dict[str, Any]:
            found = {d['_id']: {F.status: 'pending'} for d in body['docs']}
            return make_mget_response(found)

        async def _fake_bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
            items = [
                make_bulk_update_item(action['update']['_id'], status=200) for action in body[0::2]
            ]
            return make_bulk_response(items)

        opensearch = MagicMock()
        opensearch.mget = AsyncMock(side_effect=_fake_mget)
        opensearch.bulk = AsyncMock(side_effect=_fake_bulk)

        n_written, _ = await worker._bulk_update(opensearch, [a])
        assert n_written == 1
        assert opensearch.bulk.await_args is not None
        bulk_body = opensearch.bulk.await_args.kwargs['body']
        written_doc = bulk_body[1]['doc']
        assert written_doc[F.profile_revision] == doc['revision']


# =============================================================================
# B1: through the real worker, not RuntimeHolder units -- two projects,
# alpha activation swaps alpha only
# =============================================================================


async def _eventually(check: Callable[[], bool], *, timeout: float = 15.0) -> None:
    """Poll until ``check()`` holds, or ``timeout`` passes.

    The real ``worker.run()`` builds its runtimes on the event loop, and fixed sleeps
    made this test fail whenever the host was busy. On timeout this returns and the
    caller's own assertion reports the state it actually saw.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not check() and loop.time() < deadline:
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_two_project_worker_alpha_activation_swaps_alpha_only() -> None:
    """B1, through the real worker (not a RuntimeHolder unit test):
    ``worker.run()`` with two active projects (alpha, beta) from a real
    ``ProjectRegistry`` build. Activating a profile in alpha's config
    store mid-run must rebuild alpha's runtime (its own detector
    construction count increases) while beta's stays untouched."""
    import scripts.curation.region_worker_main as worker
    from src.config.curation import base_curation_config
    from src.config.projects import resources_for_new
    from src.services.projects.registry import record_to_doc

    client = TwoProjectOpenSearch()
    client.close = AsyncMock()

    def _project_doc(slug: str) -> dict[str, Any]:
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        record = ProjectRecord(
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
        return record_to_doc(record)

    projects_index = 'op_projects'
    client._docs.setdefault(projects_index, {})
    client._docs[projects_index]['project:alpha'] = {
        '_source': _project_doc('alpha'),
        '_seq_no': 0,
    }
    client._docs[projects_index]['project:beta'] = {
        '_source': _project_doc('beta'),
        '_seq_no': 0,
    }

    detector_calls: dict[str, int] = {'alpha': 0, 'beta': 0}
    segmenter_calls: dict[str, int] = {'alpha': 0, 'beta': 0}

    def _counting_detector(pool: Any, profile: Any) -> Any:
        from src.config.project_context import current_project

        detector_calls[current_project().record.slug] += 1
        return MagicMock()

    def _counting_segmenter(*_a: Any, **_k: Any) -> Any:
        from src.config.project_context import try_current_project

        bound = try_current_project()
        if bound is not None:
            segmenter_calls[bound.record.slug] += 1
        m = MagicMock()
        m.aclose = AsyncMock()
        m.enabled = False
        return m

    pool = MagicMock()
    pool.initialize = AsyncMock()
    pool.close = AsyncMock()
    pool.is_model_ready = AsyncMock(return_value=False)
    monkeypatch_targets: list[tuple[Any, str, Any]] = []

    def _patch(obj: Any, name: str, value: Any) -> None:
        monkeypatch_targets.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    _patch(worker, 'AsyncTritonPool', MagicMock(return_value=pool))
    _patch(worker, 'make_script_opensearch', MagicMock(return_value=client))
    import scripts.curation.worker.runner as runner_mod

    _patch(runner_mod, 'RegionDetector', _counting_detector)
    _patch(worker, 'SegmenterClient', _counting_segmenter)

    def _noop_signal_handler(*_a: object, **_k: object) -> None:
        return None

    import asyncio as _asyncio

    _patch(_asyncio.get_event_loop().__class__, 'add_signal_handler', _noop_signal_handler)

    try:
        args = worker.parse_args(
            [
                '--opensearch=http://os.local:9200',
                '--triton=triton:8001',
                '--segmenter-url=',
                '--continuous',
                '--poll-interval=0.01',
            ]
        )
        run_task = asyncio.create_task(worker.run(args))
        try:
            # Both projects get their first runtime built.
            await _eventually(lambda: detector_calls == {'alpha': 1, 'beta': 1})
            assert detector_calls == {'alpha': 1, 'beta': 1}

            # M1: runtime:detection_worker:<host> was written for both
            # projects on their first sync (never suppressed, never
            # deferred past 60s the first time).
            from src.config.curation import IndexRole
            from src.services.config_store.index import get_runtime_docs

            alpha_idx = resources_for_new('alpha', base_curation_config()).indexes[
                IndexRole.CONFIGS
            ]
            alpha_runtime_docs = await get_runtime_docs(
                client, alpha_idx, process='detection_worker'
            )
            assert len(alpha_runtime_docs) == 1
            assert alpha_runtime_docs[0]['project'] == 'alpha'

            # Activate a profile in ALPHA's store only.
            profile = profile_registry.get_active_region_profile()
            assert profile is not None
            with bind_project(_record('alpha')):
                store = get_config_store(mode='pinned')
                idx = store.index
                doc = await save_config(
                    client,
                    idx,
                    kind='region_profile',
                    name=profile.name,
                    body={},
                    expected_revision=None,
                )
                await activate(
                    client,
                    idx,
                    axis='detection_profile',
                    name=profile.name,
                    revision=doc['revision'],
                    expected_active=None,
                )

            await _eventually(lambda: detector_calls['alpha'] >= 2)
            # A short settle so a spurious extra rebuild would still be caught.
            await asyncio.sleep(0.1)
            assert detector_calls['alpha'] == 2
            assert detector_calls['beta'] == 1
        finally:
            run_task.cancel()
            with __import__('contextlib').suppress(_asyncio.CancelledError):
                await run_task
    finally:
        for obj, name, old in reversed(monkeypatch_targets):
            setattr(obj, name, old)
