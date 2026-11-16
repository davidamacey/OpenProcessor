"""Activating an ingest profile asks Triton whether the detector is ready. A readiness call that
could not be delivered (reset or aged-out channel, timeout) is "Triton could not be asked", never
"the model is not loaded": the first would let the person force through, the second blocks them
with a wrong message."""

from __future__ import annotations

from typing import Any

import pytest
from tritonclient.utils import InferenceServerException

from curation._ingest_profile_world import Pool
from src.clients import frame_engine
from src.clients.model_adapters import FUSED_OUTPUT
from src.services.config_store.ingest_profile_validation import _servable_issues


class _Flaky(Pool):
    def __init__(self, failures: int, answer: bool = True) -> None:
        super().__init__()
        self.failures = failures
        self.answer = answer
        self.channels: list[int] = []

    async def probe_model_ready(
        self, name: str, *, channel: int = 0, timeout: float | None = None
    ) -> bool:
        self.channels.append(channel)
        if len(self.channels) <= self.failures:
            raise InferenceServerException('Too many pings', status='StatusCode.UNAVAILABLE')
        return self.answer

    async def is_model_ready(self, name: str) -> bool:
        # The real pool's contract: any failed call reads as "not ready".
        try:
            return await self.probe_model_ready(name)
        except InferenceServerException:
            return False


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_s: float) -> None:
        return None

    monkeypatch.setattr(frame_engine, 'RETRY_SLEEP', instant)


async def _issues(pool: Any) -> list[str]:
    found = await _servable_issues(pool, 'birds', (FUSED_OUTPUT,), 'primary.model', fused=True)
    return [i.code for i in found]


@pytest.mark.asyncio
async def test_an_undelivered_probe_is_asked_again_on_another_channel() -> None:
    pool = _Flaky(failures=1)
    assert await _issues(pool) == []
    assert pool.channels == [0, 1]


@pytest.mark.asyncio
async def test_a_probe_never_delivered_is_triton_unreachable_not_not_ready() -> None:
    pool = _Flaky(failures=99)
    found = await _servable_issues(pool, 'birds', (FUSED_OUTPUT,), 'primary.model', fused=True)
    assert [i.code for i in found] == ['triton_unreachable']
    assert 'Too many pings' in found[0].message
    assert len(pool.channels) == frame_engine.PROBE_ATTEMPTS


@pytest.mark.asyncio
async def test_an_answer_of_not_ready_is_reported_and_not_retried() -> None:
    pool = _Flaky(failures=0, answer=False)
    assert await _issues(pool) == ['detector_model_not_ready']
    assert pool.channels == [0]


@pytest.mark.asyncio
async def test_a_ready_model_without_the_outputs_is_unusable() -> None:
    pool = _Flaky(failures=0)
    pool.outputs['birds'] = ['other']
    assert await _issues(pool) == ['detector_model_unusable']
