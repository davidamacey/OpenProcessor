"""The query-evaluating fake behind the real curation app, for label-operation tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.clients.curation_opensearch.registry import ClassRegistry
from src.config import get_curation_config


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def items_index() -> str:
    return get_curation_config().items_index


def proposal(crop_id: str, **extra: Any) -> dict[str, Any]:
    """An unlabeled detector proposal sitting in a residual candidate cluster."""
    return {
        'crop_id': crop_id,
        'proposal_name': 'thing',
        'confidence': 0.4,
        'class_source': 'detector_proposal',
        'class_detector': 'detector',
        'class_validated': False,
        'cluster_id': 10_003,
        'created_at': '2026-02-01T00:00:00+00:00',
        **extra,
    }


class InstrumentedFake(QueryFakeOpenSearch):
    """The query fake plus what the undo tests assert on: concurrency, bulk sizes,
    single-doc updates, and ids whose conditional writes always conflict."""

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.in_flight = 0
        self.max_in_flight = 0
        self.bulk_sizes: list[int] = []
        self.update_calls = 0
        self.conflict_ids: set[str] = set()
        self.fail_mget_on_call: int | None = None

    async def _tracked(self, coro: Any) -> Any:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0)
            return await coro
        finally:
            self.in_flight -= 1

    async def mget(self, **kw: Any) -> Any:
        if self.fail_mget_on_call is not None and self.mget_calls + 1 == self.fail_mget_on_call:
            raise RuntimeError('simulated crash')
        return await self._tracked(super().mget(**kw))

    async def bulk(self, *, body: list[dict[str, Any]], **kw: Any) -> Any:
        self.bulk_sizes.append(len(body) // 2)
        if self.conflict_ids:
            for action in body[0::2]:
                meta = action.get('update')
                if meta is not None and meta['_id'] in self.conflict_ids:
                    meta['if_seq_no'] = -1
        return await self._tracked(super().bulk(body=body, **kw))

    async def update(self, **kw: Any) -> Any:
        self.update_calls += 1
        return await super().update(**kw)


@dataclass
class OpsEnv:
    client: TestClient
    fake: InstrumentedFake
    registry: ClassRegistry
    class_ids: dict[str, int]
    prefix: str

    def docs(self) -> dict[str, dict[str, Any]]:
        return self.fake.docs(items_index())

    def label(self, class_name: str, **target: Any) -> dict[str, Any]:
        body = {'class_id': self.class_ids[class_name], **target}
        resp = self.client.put(f'{self.prefix}/crops/batch_label', json=body)
        assert resp.status_code == 200, resp.text
        return resp.json()

    def move(self, cluster_id: int, **target: Any) -> dict[str, Any]:
        resp = self.client.post(
            f'{self.prefix}/crops/move', json={'cluster_id': cluster_id, **target}
        )
        assert resp.status_code == 200, resp.text
        return resp.json()

    def undo(self, op_id: str, *, dry_run: bool | None = False) -> Any:
        body = {} if dry_run is None else {'dry_run': dry_run}
        return self.client.post(f'{self.prefix}/label_ops/{op_id}/undo', json=body)

    def list_ops(self, **params: Any) -> Any:
        return self.client.get(f'{self.prefix}/label_ops', params=params)

    def state(self) -> Path:
        return get_curation_config().project_state_dir / 'label_ops'


def build_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, docs: dict[str, dict[str, Any]] | None = None
) -> OpsEnv:
    from _curation_app import SCOPED, mount_curation_routers

    import src.config.curation_accessors as accessors
    from src.routers.curation import _common, _raw_opensearch_dep, crops_write, router
    from src.services.curation import index_bootstrap

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setattr(accessors, '_default_curation_config', None)
    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', {'default'})
    registry = ClassRegistry(path=tmp_path / 'class_registry.json')
    class_ids = {name: registry.add_class(name) for name in ('widget', 'gadget', 'gizmo')}
    monkeypatch.setattr(crops_write, 'get_class_registry', lambda: registry)
    fake = InstrumentedFake({items_index(): docs or {}})
    app = mount_curation_routers(FastAPI(), router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    app.dependency_overrides[_common._registry_dep] = lambda: registry
    return OpsEnv(TestClient(app, raise_server_exceptions=False), fake, registry, class_ids, SCOPED)
