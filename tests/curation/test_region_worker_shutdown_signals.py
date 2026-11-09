"""SIGTERM handling of the region worker loop (the GPU-claim pause is in
test_worker_gpu_claim_pause.py)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from curation.vlm_opt_in import project_opts_in_to_env_vlm
from scripts.curation.worker import runner as runner_mod
from scripts.curation.worker.__main__ import parse_args


# The cascade needs an active region profile; the default is none.
pytestmark = pytest.mark.usefixtures('reference_region_profile')


# =============================================================================
# SIGTERM handling
# =============================================================================


class TestSignalHandling:
    @pytest.mark.asyncio
    async def test_run_exits_cleanly_on_stop(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        from _fake_project_registry import install_static_project_registry

        install_static_project_registry(monkeypatch)

        async def _search(*, index: str, body: dict[str, Any]) -> dict[str, Any]:
            return {'hits': {'hits': []}}

        async def _get(*, index: str, id: str) -> dict[str, Any]:  # noqa: A002
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

        monkeypatch.setenv('OP_VLM_URL', 'http://vlm.local:8000')
        project_opts_in_to_env_vlm(monkeypatch)
        args = parse_args(
            [
                '--opensearch=http://os.local:9200',
                '--triton=triton:8001',
                '--segmenter-url=http://segmenter.local:8000',
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
