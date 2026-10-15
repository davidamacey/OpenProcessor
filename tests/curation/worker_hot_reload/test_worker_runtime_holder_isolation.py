"""RuntimeHolder isolation, and the real two-project worker swapping alpha only."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from curation._fake_config_opensearch import FakeConfigOpenSearch, TwoProjectOpenSearch
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord
from src.services.config_store import get_global_config_store
from src.services.config_store.index import activate, save_config
from src.services.config_store.store import get_config_store
from src.services.detection import profile_registry


if TYPE_CHECKING:
    from collections.abc import Callable
from curation.worker_hot_reload._helpers import _record


pytestmark = pytest.mark.usefixtures('reference_region_profile')


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
# B1: through the real worker, not RuntimeHolder units -- two projects,
# alpha activation swaps alpha only
# =============================================================================


async def _eventually(check: Callable[[], bool], *, timeout: float = 15.0) -> None:
    """Poll until ``check()`` holds, or ``timeout`` passes.

    The real ``run()`` builds its runtimes on the event loop, and fixed sleeps
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
    ``run()`` with two active projects (alpha, beta) from a real
    ``ProjectRegistry`` build. Activating a profile in alpha's config
    store mid-run must rebuild alpha's runtime (its own detector
    construction count increases) while beta's stays untouched."""
    from src.config.curation_accessors import base_curation_config
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

    import scripts.curation.worker.runner as runner_mod
    from scripts.curation.worker.__main__ import parse_args

    _patch(runner_mod, 'AsyncTritonPool', MagicMock(return_value=pool))
    _patch(runner_mod, 'make_script_opensearch', MagicMock(return_value=client))

    _patch(runner_mod, 'RegionDetector', _counting_detector)
    _patch(runner_mod, 'SegmenterClient', _counting_segmenter)

    def _noop_signal_handler(*_a: object, **_k: object) -> None:
        return None

    import asyncio as _asyncio

    _patch(_asyncio.get_event_loop().__class__, 'add_signal_handler', _noop_signal_handler)

    try:
        args = parse_args(
            [
                '--opensearch=http://os.local:9200',
                '--triton=triton:8001',
                '--segmenter-url=',
                '--continuous',
                '--poll-interval=0.01',
            ]
        )
        run_task = asyncio.create_task(runner_mod.run(args))
        try:
            # Both projects get their first runtime built.
            await _eventually(lambda: detector_calls == {'alpha': 1, 'beta': 1})
            assert detector_calls == {'alpha': 1, 'beta': 1}

            # M1: runtime:detection_worker:<host> was written for both
            # projects on their first sync (never suppressed, never
            # deferred past 60s the first time).
            from src.config.curation_roles import IndexRole
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
