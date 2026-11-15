"""Fixtures of the native tests: the release binary and a factory that starts either engine."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from .engines import Engine, PythonEngine, RustEngine, Settings, build_binary


if TYPE_CHECKING:
    from pathlib import Path


StartEngine = Callable[..., Awaitable[Engine]]


@pytest.fixture(scope='session')
def binary() -> Path:
    return build_binary()


@pytest_asyncio.fixture
async def start(binary: Path, tmp_path: Path) -> AsyncIterator[StartEngine]:
    """``await start('rust' | 'python', max_in_flight=1, ...)``: an engine on free ports, stopped
    (by the exact process it started) when the test ends."""
    started: list[Engine] = []

    async def make(kind: str, extra_env: dict[str, str] | None = None, **settings: float) -> Engine:
        config = Settings(**settings)  # type: ignore[arg-type]
        if kind == 'rust':
            rust = RustEngine(binary, tmp_path, config, extra_env)
            started.append(rust)
            await rust.wait_started()
            return rust
        python = PythonEngine(config)
        started.append(python)
        await python.start()
        return python

    yield make
    for engine in started:
        await engine.stop()


@pytest.fixture(params=['rust', 'python'])
def kind(request: pytest.FixtureRequest) -> str:
    """Every case that takes it runs against the native binary and the Python engine's front."""
    return str(request.param)
