"""Preflight export selection: the default current export, an explicit directory and the val-split block."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest


if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient


# =============================================================================
# dataset_export_dir defaults to the current export
# =============================================================================


def test_preflight_defaults_to_the_current_export_when_omitted(
    app_client: TestClient,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    project_export_root: Path,
) -> None:
    current = project_export_root / '20260925T000000Z'
    current.mkdir(parents=True)
    monkeypatch.setattr('src.services.curation.export.resolve_current_export_dir', lambda: current)

    r = app_client.post(
        '/curation/projects/default/train/preflight',
        json={'model_size': 'm', 'profile': 'medium'},
    )

    assert r.status_code == 200, r.text
    out = r.json()
    check = next(c for c in out['checks'] if c['name'] == 'dataset_export_dir')
    assert check['severity'] == 'ok'
    assert str(current) in check['message']


def test_preflight_reports_a_clear_block_when_no_export_exists_at_all(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise() -> Any:
        msg = 'no current export symlink found'
        raise FileNotFoundError(msg)

    monkeypatch.setattr('src.services.curation.export.resolve_current_export_dir', _raise)

    r = app_client.post(
        '/curation/projects/default/train/preflight',
        json={'model_size': 'm', 'profile': 'medium'},
    )

    assert r.status_code == 200, r.text
    out = r.json()
    assert out['blocked'] is True
    check = next(c for c in out['checks'] if c['name'] == 'dataset_export_dir')
    assert check['severity'] == 'block'
    assert 'export/yolo' in check['message']


def test_preflight_still_honors_an_explicit_dataset_export_dir(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch, project_export_root: Path
) -> None:
    """An explicit dataset_export_dir must win over the current-export
    default -- resolve_current_export_dir must not even be consulted."""
    explicit = project_export_root / 'explicit_export'
    explicit.mkdir()

    def _boom() -> Any:
        msg = 'must not be called when dataset_export_dir is explicit'
        raise AssertionError(msg)

    monkeypatch.setattr('src.services.curation.export.resolve_current_export_dir', _boom)

    r = app_client.post(
        '/curation/projects/default/train/preflight',
        json={'dataset_export_dir': str(explicit), 'model_size': 'm', 'profile': 'medium'},
    )

    assert r.status_code == 200, r.text
    names = [c['name'] for c in r.json()['checks']]
    assert 'dataset_export_dir' not in names


def _empty_val_report(empty: list[str]) -> Any:
    from src.services.training.preflight_checks import PreflightCheck, PreflightReport

    return PreflightReport(
        blocked=True,
        checks=[
            PreflightCheck(
                name='export_splits_nonempty',
                severity='block',
                message=f'export has 0 images in {empty}',
                detail={'empty_splits': empty},
            )
        ],
    )


@pytest.mark.parametrize('route', ['start', 'start_campaign'])
def test_force_does_not_bypass_empty_val_split(
    app_client: TestClient, project_export_root: Path, route: str
) -> None:
    export = str(project_export_root / 'x')
    body: dict[str, Any] = (
        {'dataset_export_dir': export, 'profile': 'medium'}
        if route == 'start'
        else {'dataset_export_dir': export, 'runs': [{'profile': 'nano', 'model_size': 'n'}]}
    )
    with patch(
        'src.routers.curation_train.start.run_preflight',
        new=AsyncMock(return_value=_empty_val_report(['val'])),
    ):
        r = app_client.post(f'/curation/projects/default/train/{route}?force=true', json=body)
    assert r.status_code == 422, r.text
    assert r.json()['detail']['error'] == 'empty_val_split'


def test_force_still_bypasses_an_empty_train_only_block(
    app_client: TestClient, project_export_root: Path
) -> None:
    """Only the val split is a hard failure; other blocks stay forceable."""
    body = {'dataset_export_dir': str(project_export_root / 'x'), 'profile': 'medium'}
    with patch(
        'src.routers.curation_train.start.run_preflight',
        new=AsyncMock(return_value=_empty_val_report(['train'])),
    ):
        r = app_client.post('/curation/projects/default/train/start?force=true', json=body)
    assert r.status_code == 201, r.text
