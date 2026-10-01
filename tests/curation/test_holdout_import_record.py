"""W10.9: an import freezes its test split under its own record; the curated
freeze (``current.json``) is never replaced, and an imported holdout counts
as an existing holdout for the curated freeze."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.curation.holdout import (
    build_cohort_query,
    compute_holdout_sha,
    mark_freeze_record_undone,
    persist_freeze_record,
)


if TYPE_CHECKING:
    from pathlib import Path


def _persist(state_dir: Path, ids: list[str], **kwargs: Any) -> Path:
    return persist_freeze_record(
        crop_ids=ids,
        sha=compute_holdout_sha(ids),
        cohort_spec={'kind': 'x'},
        per_class_counts={'0': len(ids)},
        state_dir=state_dir,
        **kwargs,
    )


def test_an_import_freeze_writes_under_imports_and_leaves_current_json_alone(
    tmp_path: Path,
) -> None:
    curated = _persist(tmp_path, ['a', 'b'])
    current = tmp_path / 'current.json'
    before = current.read_text()

    snapshot = _persist(tmp_path, ['c', 'd', 'e'], kind='import')

    assert snapshot.parent == tmp_path / 'imports'
    # Only timestamped snapshots live there: no second "current" pointer.
    assert [p.name for p in snapshot.parent.iterdir()] == [snapshot.name]
    assert curated.parent == tmp_path
    assert current.read_text() == before
    record = json.loads(snapshot.read_text())
    assert record['kind'] == 'import'
    assert record['n_frozen'] == 3
    assert json.loads(current.read_text())['kind'] == 'curated'


def test_an_import_freeze_with_no_prior_curated_freeze_creates_no_current_json(
    tmp_path: Path,
) -> None:
    _persist(tmp_path, ['a'], kind='import')
    assert not (tmp_path / 'current.json').exists()


def test_the_default_kind_keeps_todays_behaviour(tmp_path: Path) -> None:
    snapshot = _persist(tmp_path, ['a'])
    assert snapshot.parent == tmp_path
    assert json.loads((tmp_path / 'current.json').read_text()) == json.loads(snapshot.read_text())


def test_mark_undone_stamps_the_import_record_only(tmp_path: Path) -> None:
    snapshot = _persist(tmp_path, ['a', 'b'], kind='import')
    mark_freeze_record_undone(snapshot)
    record = json.loads(snapshot.read_text())
    assert record['undone_at']
    assert record['crop_ids'] == ['a', 'b']
    curated = _persist(tmp_path, ['z'])
    with pytest.raises(ValueError, match='not an import freeze'):
        mark_freeze_record_undone(curated)
    with pytest.raises(FileNotFoundError):
        mark_freeze_record_undone(tmp_path / 'missing.json')


def test_the_curated_cohort_never_samples_imported_labels() -> None:
    filters = build_cohort_query()['bool']['filter']
    assert {'term': {'class_source': 'human'}} in filters


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from _curation_app import mount_curation_routers

    from src.config.curation import CurationConfig
    from src.core.dependencies import get_opensearch
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr(
        'src.services.curation.holdout.get_curation_config',
        lambda: CurationConfig(state_dir=tmp_path),
    )
    fake = AsyncMock()
    fake.indices = AsyncMock()
    fake.indices.exists = AsyncMock(return_value=True)
    fake.indices.create = AsyncMock(return_value={'acknowledged': True})
    fake.indices.refresh = AsyncMock(return_value={'_shards': {}})
    fake.indices.put_mapping = AsyncMock(return_value={'acknowledged': True})
    fake.search = AsyncMock(return_value={'hits': {'hits': [], 'total': {'value': 0}}})
    fake.bulk = AsyncMock(return_value={'errors': False, 'items': []})
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[get_opensearch] = lambda: fake
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as client:
        client.fake_os = fake  # type: ignore[attr-defined]
        yield client


def test_the_curated_freeze_refuses_while_an_imported_holdout_exists(app_client: Any) -> None:
    app_client.fake_os.count = AsyncMock(return_value={'count': 41})
    url = '/curation/projects/default/test_holdout/freeze'
    refused = app_client.post(url, json={'percent': 20})
    assert refused.status_code == 409
    # Any flagged item counts, whatever wrote the flag.
    query = app_client.fake_os.count.call_args.kwargs['body']['query']
    assert query == {'term': {'test_holdout': True}}
    # force=true gets past the guard (this cohort is empty, so 422, not 409).
    forced = app_client.post(url + '?force=true', json={'percent': 20})
    assert forced.status_code == 422
