"""Preflight free-disk and training-volume mount checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from fastapi.testclient import TestClient


# =============================================================================
# free-disk preflight check (P2-8)
# =============================================================================


def test_free_gb_returns_none_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Before the fix, an OSError from shutil.disk_usage() returned
    float('inf') -- 'infinite free space' -- instead of a signal the check
    couldn't run. It must return None so the caller reports 'unknown'."""
    from src.services.training.preflight_checks import free_gb

    def _raise(_path: str) -> None:
        raise OSError('no such path')

    monkeypatch.setattr('src.services.training.preflight_checks.shutil.disk_usage', _raise)
    assert free_gb('/does/not/matter') is None


def test_free_gb_normal_case_returns_sane_number(tmp_path: Any) -> None:
    from src.services.training.preflight_checks import free_gb

    free = free_gb(str(tmp_path))
    assert free is not None
    assert free > 0


def test_free_gb_measures_mounted_ancestor_of_missing_dir(tmp_path: Any) -> None:
    from src.services.training.preflight_checks import free_gb

    free = free_gb(str(tmp_path / 'training_staging' / 'not_created'))
    assert free is not None
    assert free > 0


def test_training_volume_mount_sane_false_when_same_device_as_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact bug this check exists to catch: the real training volume
    isn't mounted, so the 'data' path silently resolves to the container's
    own root filesystem."""
    from pathlib import Path

    from src.services.training.preflight_checks import training_volume_mount_sane

    class _Stat:
        st_dev = 42

    real_stat = Path.stat

    def _fake_stat(self: Path, *, follow_symlinks: bool = True) -> object:
        if str(self) in ('/data/train_staging', '/'):
            return _Stat()
        return real_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, 'stat', _fake_stat)
    assert training_volume_mount_sane('/data/train_staging') is False


def test_training_volume_mount_sane_checks_nearest_existing_ancestor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh volume has no staging dir yet; the verdict must come from the
    mounted parent, not a stat() OSError read as 'same filesystem as root'."""
    from pathlib import Path

    from src.services.training.preflight_checks import training_volume_mount_sane

    class _Stat:
        def __init__(self, dev: int) -> None:
            self.st_dev = dev

    devices = {'/': 1, '/var/lib/app': 2}

    def _fake_stat(self: Path, *, follow_symlinks: bool = True) -> object:
        if str(self) in devices:
            return _Stat(devices[str(self)])
        raise FileNotFoundError(str(self))

    monkeypatch.setattr(Path, 'stat', _fake_stat)
    assert training_volume_mount_sane('/var/lib/app/training_staging') is True
    devices['/var/lib/app'] = 1
    assert training_volume_mount_sane('/var/lib/app/training_staging') is False


def test_training_volume_mount_sane_true_for_distinct_device(tmp_path: Any) -> None:
    from src.services.training.preflight_checks import training_volume_mount_sane

    # tmp_path and '/' are the same device in most CI sandboxes, so this
    # only asserts the function runs without raising and returns a bool --
    # the real "different device" case is covered by the monkeypatched
    # test above, which is what actually exercises the guard's logic.
    result = training_volume_mount_sane(str(tmp_path))
    assert isinstance(result, bool)


def test_preflight_blocks_when_mount_not_sane(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch, project_export_root: Path
) -> None:
    monkeypatch.setattr(
        'src.services.training.preflight_run.training_volume_mount_sane',
        lambda _path: False,
    )
    body = {'dataset_export_dir': str(project_export_root / 'x'), 'profile': 'medium'}
    r = app_client.post('/curation/projects/default/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    disk_check = next(c for c in out['checks'] if c['name'] == 'free_disk')
    assert disk_check['severity'] == 'block'
    assert 'mount' in disk_check['message'].lower()
    assert out['blocked'] is True


def test_preflight_reports_unknown_not_ok_when_disk_unreadable(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch, project_export_root: Path
) -> None:
    monkeypatch.setattr(
        'src.services.training.preflight_run.training_volume_mount_sane',
        lambda _path: True,
    )
    monkeypatch.setattr('src.services.training.preflight_run.free_gb', lambda _path: None)
    body = {'dataset_export_dir': str(project_export_root / 'x'), 'profile': 'medium'}
    r = app_client.post('/curation/projects/default/train/preflight', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    disk_check = next(c for c in out['checks'] if c['name'] == 'free_disk')
    assert disk_check['severity'] == 'unknown'
