"""Shared fixtures for the ingest-profile tests: an in-memory OpenSearch that serves
both the config store and the settings document (the ingest policy), and a fake Triton
pool."""

from __future__ import annotations

from typing import Any

from curation._fake_config_opensearch import FakeConfigOpenSearch


END2END = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']


class WorldOpenSearch(FakeConfigOpenSearch):
    async def index(self, index: str, id: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: A002
        kw.pop('op_type', None)
        return await super().index(index, id, body, **kw)


class Pool:
    def __init__(self) -> None:
        self.ready: dict[str, bool] = {}
        self.outputs: dict[str, list[str]] = {}
        self.default_ready = True

    async def is_model_ready(self, name: str) -> bool:
        return self.ready.get(name, self.default_ready)

    async def get_model_output_names(self, name: str) -> list[str]:
        return self.outputs.get(name, END2END)


def body(
    model: str = 'birds', primary: dict[str, Any] | None = None, **extra: Any
) -> dict[str, Any]:
    return {'primary': {'model': model, **(primary or {})}, **extra}
