"""Sentinel pause and SIGTERM handling of the region worker loop."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import scripts.curation.worker.state as worker_state
from scripts.curation.worker import runner as runner_mod
from scripts.curation.worker.__main__ import parse_args


if TYPE_CHECKING:
    from pathlib import Path

# The cascade needs an active region profile; the default is none.
pytestmark = pytest.mark.usefixtures('reference_region_profile')


# =============================================================================
# Sentinel pause
# =============================================================================


class TestSentinel:
    @pytest.mark.asyncio
    async def test_wait_returns_immediately_when_sentinel_absent(self, tmp_path: Path) -> None:
        sentinel = tmp_path / 'pause.sentinel'  # does not exist
        # Should not block.
        await asyncio.wait_for(
            worker_state._wait_for_sentinel_clear(sentinel, sleep_s=0.01), timeout=0.5
        )

    @pytest.mark.asyncio
    async def test_wait_blocks_then_unblocks(self, tmp_path: Path) -> None:
        sentinel = tmp_path / 'pause.sentinel'
        sentinel.write_text('paused')

        async def _delete_after(delay: float) -> None:
            await asyncio.sleep(delay)
            sentinel.unlink()

        # Kick off the deletion concurrently with the wait.
        deleter = asyncio.create_task(_delete_after(0.05))
        await asyncio.wait_for(
            worker_state._wait_for_sentinel_clear(sentinel, sleep_s=0.01), timeout=1.0
        )
        await deleter
        assert not sentinel.exists()


# =============================================================================
# SIGTERM handling
# =============================================================================


class TestSignalHandling:
    @pytest.mark.asyncio
    async def test_run_exits_cleanly_on_stop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A faked ``run``-style loop respects the stop event and shuts down
        all owned resources (httpx clients, OpenSearch, Triton pool)."""

        # Patch the heavy I/O constructors so ``run()`` is pure async logic.
        pool = MagicMock()
        pool.initialize = AsyncMock()
        pool.close = AsyncMock()
        monkeypatch.setattr(runner_mod, 'AsyncTritonPool', MagicMock(return_value=pool))

        # B1: the producer loop now iterates the real ProjectRegistry's
        # active_projects() every cycle, so the registry needs a real
        # 'default' doc to see -- previously the single global runtime
        # was built unconditionally from the env profile, regardless of
        # what the registry (a separate, only-used-for-fetch concern)
        # contained.
        from src.config.curation import base_curation_config
        from src.config.projects import resources_for_new
        from src.services.projects.registry import (
            REVISION_DOC_ID,
            ProjectRecord,
            projects_index,
            record_to_doc,
        )

        default_record = ProjectRecord(
            slug='default',
            display_name='Default',
            description='',
            status='active',
            revision=1,
            created_at='',
            updated_at='',
            origin=None,
            resources=resources_for_new('default', base_curation_config()),
        )

        async def _search(*, index: str, body: dict[str, Any]) -> dict[str, Any]:
            if index == projects_index():
                return {'hits': {'hits': [{'_source': record_to_doc(default_record)}]}}
            return {'hits': {'hits': []}}

        async def _get(*, index: str, id: str) -> dict[str, Any]:  # noqa: A002
            if index == projects_index() and id == REVISION_DOC_ID:
                return {'found': True, '_source': {'revision': 1}}
            return {'found': False}

        os_client = MagicMock()
        os_client.search = AsyncMock(side_effect=_search)
        os_client.get = AsyncMock(side_effect=_get)
        os_client.bulk = AsyncMock()
        os_client.close = AsyncMock()
        monkeypatch.setattr(runner_mod, 'make_script_opensearch', MagicMock(return_value=os_client))

        segmenter = MagicMock()
        segmenter.aclose = AsyncMock()
        monkeypatch.setattr(runner_mod, 'SegmenterClient', MagicMock(return_value=segmenter))

        vlm = MagicMock()
        vlm.aclose = AsyncMock()
        monkeypatch.setattr(runner_mod, 'build_vlm_labeler', MagicMock(return_value=vlm))

        # Patch signal-handler installation away — adding signal handlers
        # in a non-main asyncio loop would raise here.
        def _noop_signal_handler(*_args: object, **_kwargs: object) -> None:
            return None

        monkeypatch.setattr(
            asyncio.get_event_loop().__class__,
            'add_signal_handler',
            _noop_signal_handler,
            raising=False,
        )

        # Bind the metrics HTTP server to an ephemeral port so the test
        # doesn't collide with a worker container already holding the
        # default 4609 on this host.
        monkeypatch.setenv('OP_REGION_WORKER_METRICS_PORT', '0')

        sentinel = tmp_path / 'pause.sentinel'  # absent
        monkeypatch.setenv('OP_VLM_URL', 'http://vlm.local:8000')
        args = parse_args(
            [
                '--opensearch=http://os.local:9200',
                '--triton=triton:8001',
                '--segmenter-url=http://segmenter.local:8000',
                f'--pause-sentinel={sentinel}',
                '--max-iterations=1',
            ]
        )
        # No --continuous, so empty queue should make run() exit immediately
        # after one fetch.
        rc = await runner_mod.run(args)
        assert rc == 0
        # Cleanup happened.
        os_client.close.assert_awaited()
        pool.close.assert_awaited()
        segmenter.aclose.assert_awaited()
        vlm.aclose.assert_awaited()
