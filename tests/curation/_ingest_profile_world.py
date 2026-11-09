"""Shared fixtures for the ingest-profile tests: a fake Triton pool and profile bodies."""

from __future__ import annotations

from typing import Any


FUSED = ['output0']


class Pool:
    def __init__(self) -> None:
        self.ready: dict[str, bool] = {}
        self.outputs: dict[str, list[str]] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.default_ready = True

    async def is_model_ready(self, name: str) -> bool:
        return self.ready.get(name, self.default_ready)

    async def get_model_output_names(self, name: str) -> list[str]:
        return self.outputs.get(name, FUSED)

    async def get_model_output_shape(self, name: str, _output: str) -> tuple[int, ...]:
        return self.shapes.get(name, (-1, 300, 6))


def body(
    model: str = 'birds', primary: dict[str, Any] | None = None, **extra: Any
) -> dict[str, Any]:
    return {'primary': {'model': model, **(primary or {})}, **extra}
