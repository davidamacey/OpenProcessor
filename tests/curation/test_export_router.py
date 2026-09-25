"""Router-level tests for the curation export endpoints.

The reference test this was ported from exercised the reference's
Bucket-B domain-specific export service's version_tag/seed passthrough
bug fix directly. That service is never ported (plan §1), so this file instead
exercises the generic port's equivalent: ``POST /curation/export/yolo``
passes ``version_tag``/``seed`` through to
:class:`~src.services.curation.export.GenericYoloExportService`, plus
the registry-artifact-serving endpoints (ported close to verbatim — pure
filesystem serving, no Bucket-B dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers.curation._common import ExportYoloRequest
from src.routers.curation.export import export_yolo as export_yolo_handler
from src.services.curation.export import ExportResult, GenericYoloExportService, SplitCounts


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_export_yolo_handler_passes_version_tag_and_seed_to_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def _fake_export_dataset(self: GenericYoloExportService, **kwargs: Any) -> ExportResult:
        captured.update(kwargs)
        return ExportResult(
            export_dir='/tmp/fake-export',
            version_tag=kwargs.get('version_tag', ''),
            manifest_path='/tmp/fake-export/manifest.json',
            data_yaml_path='/tmp/fake-export/data.yaml',
            dataset_sha='deadbeef',
            split_counts=SplitCounts(),
            image_count=0,
            class_count=0,
            classes_with_objects=0,
            started_at='2026-01-01T00:00:00+00:00',
            finished_at='2026-01-01T00:00:01+00:00',
            current_symlink='/tmp/fake-export/current',
        )

    monkeypatch.setattr(
        'src.services.curation.export.GenericYoloExportService.export_dataset',
        _fake_export_dataset,
    )

    payload = ExportYoloRequest(version_tag='audit-smoke-v1', seed=1234, max_images=5)
    response = await export_yolo_handler(payload, MagicMock(), MagicMock())

    assert captured.get('version_tag') == 'audit-smoke-v1'
    assert captured.get('seed') == 1234
    assert response['version_tag'] == 'audit-smoke-v1'


# =============================================================================
# Registry artifact download endpoint
# =============================================================================


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = AsyncMock()
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    return TestClient(app)


def test_export_registry_route_is_mounted(app_client: TestClient) -> None:
    route_paths = {route.path for route in app_client.app.routes}
    assert '/curation/export/registry/{artifact}' in route_paths

    # Point resolution at a directory with nothing in it — the whitelist
    # check + directory resolution both run, we just want to prove the URL
    # actually reaches our handler (any non-405 status is fine here; the
    # exact 404 behavior for a missing current export is covered below).
    for artifact in ('class_registry.json', 'data.yaml', 'manifest.json'):
        response = app_client.get(f'/curation/export/registry/{artifact}')
        assert response.status_code != 405


@pytest.fixture
def fake_export_dir(tmp_path: Path) -> Path:
    export_dir = tmp_path / '20260912T000000Z_v7-test'
    export_dir.mkdir()
    (export_dir / 'class_registry.json').write_text('{"classes": ["car", "truck"]}\n')
    (export_dir / 'data.yaml').write_text('nc: 2\nnames: [car, truck]\n')
    (export_dir / 'manifest.json').write_text('{"dataset_sha": "deadbeef"}\n')
    (export_dir / 'label_stats.json').write_text('{"car": 10}\n')
    return export_dir


@pytest.mark.parametrize(
    ('artifact', 'content_type'),
    [
        ('class_registry.json', 'application/json'),
        ('data.yaml', 'application/x-yaml'),
        ('manifest.json', 'application/json'),
        ('label_stats.json', 'application/json'),
    ],
)
def test_registry_artifact_happy_path(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_export_dir: Path,
    artifact: str,
    content_type: str,
) -> None:
    monkeypatch.setattr(
        'src.routers.curation.export._resolve_current_export_dir',
        lambda: fake_export_dir,
    )

    response = app_client.get(f'/curation/export/registry/{artifact}')

    assert response.status_code == 200
    assert response.content == (fake_export_dir / artifact).read_bytes()
    assert response.headers['content-type'].startswith(content_type)
    assert f'filename="{artifact}"' in response.headers['content-disposition']
    assert response.headers['content-disposition'].startswith('attachment')
    assert response.headers['cache-control'] == 'no-store'


def test_registry_artifact_unknown_name_short_circuits_before_filesystem(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> Path:
        msg = 'directory resolver must not be called for a non-whitelisted artifact'
        raise AssertionError(msg)

    monkeypatch.setattr(
        'src.routers.curation.export._resolve_current_export_dir',
        _boom,
    )

    response = app_client.get('/curation/export/registry/nonexistent.yaml')

    assert response.status_code == 404


@pytest.mark.parametrize(
    'artifact',
    [
        '../../../../etc/passwd',
        '..%2F..%2F..%2F..%2Fetc%2Fpasswd',
        '..%252F..%252Fetc%252Fpasswd',
    ],
)
def test_registry_artifact_path_traversal_rejected(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    def _boom() -> Path:
        msg = 'directory resolver must not be called for a path-traversal attempt'
        raise AssertionError(msg)

    monkeypatch.setattr(
        'src.routers.curation.export._resolve_current_export_dir',
        _boom,
    )

    response = app_client.get(f'/curation/export/registry/{artifact}')

    assert response.status_code == 404


def test_registry_artifact_no_current_export_returns_actionable_404(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> Path:
        raise FileNotFoundError

    monkeypatch.setattr(
        'src.routers.curation.export._resolve_current_export_dir',
        _raise,
    )

    response = app_client.get('/curation/export/registry/manifest.json')

    assert response.status_code == 404
    assert 'export' in response.json()['detail'].lower()
    assert 'POST /curation/export/yolo' in response.json()['detail']


def test_registry_artifact_missing_file_in_valid_export_dir(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / '20260912T000000Z_v7-partial'
    export_dir.mkdir()
    (export_dir / 'manifest.json').write_text('{}\n')
    # class_registry.json deliberately absent.

    monkeypatch.setattr(
        'src.routers.curation.export._resolve_current_export_dir',
        lambda: export_dir,
    )

    response = app_client.get('/curation/export/registry/class_registry.json')

    assert response.status_code == 404
    assert 'class_registry.json' in response.json()['detail']


# =============================================================================
# Single-class / class-subset export endpoints (plan §4.1 G2)
# =============================================================================


def test_single_class_routes_are_mounted(app_client: TestClient) -> None:
    route_paths = {route.path for route in app_client.app.routes}
    assert '/curation/export/single_class' in route_paths
    assert '/curation/export/single_class/status' in route_paths


@pytest.mark.asyncio
async def test_single_class_handler_builds_the_profile_from_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The target vocabulary comes from the REQUEST, not from anything
    hardcoded in the service — the whole point of G2's genericization."""
    from src.routers.curation._common import ExportSingleClassRequest
    from src.routers.curation.export_single_class import (
        export_single_class as export_single_class_handler,
    )
    from src.services.curation.export_single_class import (
        SingleClassExportProfile,
        SingleClassExportResult,
        SingleClassExportService,
        SingleClassSplitCounts,
    )

    captured: dict[str, Any] = {}

    def _fake_init(
        self: SingleClassExportService,
        opensearch: Any,
        *,
        profile: SingleClassExportProfile,
        **_kwargs: Any,
    ) -> None:
        captured['profile'] = profile

    async def _fake_export(
        self: SingleClassExportService,
        **kwargs: Any,
    ) -> SingleClassExportResult:
        captured.update(kwargs)
        return SingleClassExportResult(
            export_dir='/tmp/fake-single-class',
            version_tag=kwargs.get('version_tag', ''),
            manifest_path='/tmp/fake-single-class/manifest.json',
            data_yaml_path='/tmp/fake-single-class/data.yaml',
            dataset_sha='abc123',
            frozen_test_sha='def456',
            split_counts=SingleClassSplitCounts(train=8, val=1, test=1),
            image_count=10,
            class_count=2,
            positive_images=10,
            background_images=0,
            started_at='2026-01-01T00:00:00+00:00',
            finished_at='2026-01-01T00:00:05+00:00',
            current_symlink='/tmp/exports/plates/current',
        )

    monkeypatch.setattr(SingleClassExportService, '__init__', _fake_init)
    monkeypatch.setattr(SingleClassExportService, 'export', _fake_export)

    payload = ExportSingleClassRequest(
        version_tag='subset-v1',
        class_ids=[7, 3],
        profile_name='regions',
        box_source='region',
        region_class_name='region',
        seed=99,
        max_positive_images=250,
        img_max_side=640,
    )
    response = await export_single_class_handler(payload, MagicMock())

    profile = captured['profile']
    assert profile.class_ids == (7, 3)
    assert profile.name == 'regions'
    assert profile.box_source == 'region'
    assert profile.region_class_name == 'region'
    assert captured['seed'] == 99
    assert captured['max_positive_images'] == 250
    assert captured['img_max_side'] == 640
    assert response['dataset_sha'] == 'abc123'
    assert response['frozen_test_sha'] == 'def456'
    assert response['positives_zero_warning'] is False


@pytest.mark.asyncio
async def test_single_class_handler_maps_bad_config_to_422_not_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty class_ids is a caller mistake, not a server fault."""
    from fastapi import HTTPException

    from src.routers.curation._common import ExportSingleClassRequest
    from src.routers.curation.export_single_class import (
        export_single_class as export_single_class_handler,
    )
    from src.services.curation.export_single_class import SingleClassExportService

    async def _raise(self: SingleClassExportService, **_kwargs: Any) -> None:
        msg = "box_source='item' requires a non-empty profile.class_ids"
        raise ValueError(msg)

    monkeypatch.setattr(SingleClassExportService, 'export', _raise)

    with pytest.raises(HTTPException) as excinfo:
        await export_single_class_handler(ExportSingleClassRequest(), MagicMock())

    assert excinfo.value.status_code == 422
    assert 'class_ids' in str(excinfo.value.detail)


def test_single_class_status_is_idle_before_any_export(app_client: TestClient) -> None:
    response = app_client.get('/curation/export/single_class/status?profile_name=never-run')

    assert response.status_code == 200
    assert response.json() == {
        'status': 'idle',
        'last_run': None,
        'profile_name': 'never-run',
    }


def test_single_class_status_reports_the_manifest(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import json

    export_dir = tmp_path / '20260921T000000Z'
    export_dir.mkdir()
    (export_dir / 'manifest.json').write_text(
        json.dumps(
            {
                'dataset_kind': 'single_class',
                'dataset_sha': 'cafe1234',
                'frozen_test_sha': 'beef5678',
                'finished_at': '2026-09-21T00:00:05+00:00',
                'class_count': 1,
                'class_names': ['region'],
                'image_count': 42,
                'positive_images': 40,
                'background_images': 2,
                'false_positive_background_images': 2,
                'positives_zero_warning': False,
                'split_counts': {'train': 34, 'val': 4, 'test': 4},
            }
        )
    )
    monkeypatch.setattr(
        'src.routers.curation.export_single_class._resolve_current_dir',
        lambda _profile_name: export_dir,
    )

    body = app_client.get('/curation/export/single_class/status?profile_name=regions').json()

    assert body['status'] == 'success'
    assert body['profile_name'] == 'regions'
    assert body['dataset_sha'] == 'cafe1234'
    assert body['frozen_test_sha'] == 'beef5678'
    assert body['class_names'] == ['region']
    assert body['split_counts'] == {'train': 34, 'val': 4, 'test': 4}


def test_single_class_status_unknown_when_manifest_unreadable(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / '20260921T000001Z'
    export_dir.mkdir()
    (export_dir / 'manifest.json').write_text('{ not json')
    monkeypatch.setattr(
        'src.routers.curation.export_single_class._resolve_current_dir',
        lambda _profile_name: export_dir,
    )

    body = app_client.get('/curation/export/single_class/status').json()

    assert body['status'] == 'unknown'
    assert body['export_dir'] == str(export_dir)
