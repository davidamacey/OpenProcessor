"""The query-evaluating fake behind the real curation app, for label-operation tests."""

from __future__ import annotations

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


@dataclass
class OpsEnv:
    client: TestClient
    fake: QueryFakeOpenSearch
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
    fake = QueryFakeOpenSearch({items_index(): docs or {}})
    app = mount_curation_routers(FastAPI(), router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    app.dependency_overrides[_common._registry_dep] = lambda: registry
    return OpsEnv(TestClient(app), fake, registry, class_ids, SCOPED)
