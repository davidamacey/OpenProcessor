"""A model that is READY is unloaded, awaited and only then loaded (Triton reload crash)."""

from __future__ import annotations

import httpx
import pytest

from src.services.models.triton_control import ModelReloadError, load_model_safely


class _Fake:
    """Records calls; ``ready_polls_after_unload`` ready answers follow an unload."""

    def __init__(self, ready: bool, polls_after_unload: int = 1, stuck: bool = False) -> None:
        self.ready = ready
        self.left = polls_after_unload
        self.stuck = stuck
        self.calls: list[str] = []

    async def get(self, url: str) -> httpx.Response:
        self.calls.append('ready')
        if self.ready and self.calls.count('unload') and not self.stuck:
            if self.left <= 0:
                self.ready = False
            self.left -= 1
        return httpx.Response(200 if self.ready else 400)

    async def post(self, url: str) -> httpx.Response:
        op = url.rsplit('/', 1)[1]
        self.calls.append(op)
        return httpx.Response(200)


def _ops(fake: _Fake) -> list[str]:
    return [c for c in fake.calls if c != 'ready']


@pytest.mark.asyncio
async def test_ready_model_is_unloaded_awaited_then_loaded() -> None:
    fake = _Fake(ready=True, polls_after_unload=2)
    await load_model_safely(fake, 'http://t', 'm', poll_interval=0)  # type: ignore[arg-type]
    assert _ops(fake) == ['unload', 'load']
    assert fake.calls.index('unload') < fake.calls.index('load')
    assert fake.calls[-2] == 'ready'  # the last look before the load saw it gone


@pytest.mark.asyncio
async def test_unloaded_model_is_just_loaded() -> None:
    fake = _Fake(ready=False)
    await load_model_safely(fake, 'http://t', 'm')  # type: ignore[arg-type]
    assert _ops(fake) == ['load']


@pytest.mark.asyncio
async def test_a_model_that_never_unloads_is_not_loaded() -> None:
    fake = _Fake(ready=True, stuck=True)
    with pytest.raises(ModelReloadError, match='still ready'):
        await load_model_safely(
            fake,  # type: ignore[arg-type]
            'http://t',
            'm',
            unload_timeout=0.05,
            poll_interval=0.01,
        )
    assert 'load' not in fake.calls


@pytest.mark.asyncio
async def test_an_unreadable_state_refuses_the_load() -> None:
    class _Down(_Fake):
        async def get(self, url: str) -> httpx.Response:
            return httpx.Response(503)

    fake = _Down(ready=False)
    with pytest.raises(ModelReloadError):
        await load_model_safely(fake, 'http://t', 'm')  # type: ignore[arg-type]
    assert fake.calls == []
