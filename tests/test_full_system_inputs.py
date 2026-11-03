"""``tests/test_full_system.py`` must fail loudly when its input images are missing.

A download that fails (for example into a root-owned ``test_images``) used to print a
warning, skip the checks that need the image and still report success.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def system_test(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        '_full_system_under_inputs_test', Path(__file__).parent / 'test_full_system.py'
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'TEST_IMAGE', tmp_path / 'ro' / 'a.jpg')
    monkeypatch.setattr(module, 'TEST_IMAGE_2', tmp_path / 'ro' / 'b.jpg')

    def _refuse(_url: str, _path: Path) -> None:
        raise PermissionError('read-only directory')

    monkeypatch.setattr('scripts.datasets._common.download', _refuse)
    return module


def test_a_failed_download_raises_naming_the_missing_images(system_test: Any) -> None:
    with pytest.raises(RuntimeError, match=r'a\.jpg.*b\.jpg'):
        system_test.download_test_images()


def test_the_run_stops_with_failure_before_any_check(
    system_test: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _must_not_run() -> bool:
        msg = 'a check ran with the test images missing'
        raise AssertionError(msg)

    monkeypatch.setattr(system_test, 'clear_opensearch_data', _must_not_run)
    monkeypatch.setattr(system_test, 'test_health_checks', _must_not_run)

    assert system_test.run_all_tests() is False
    assert 'test images are missing' in capsys.readouterr().out


def test_an_existing_image_needs_no_download(
    system_test: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    for path in (system_test.TEST_IMAGE, system_test.TEST_IMAGE_2):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'x')
    system_test.download_test_images()


def test_a_mounted_ingest_directory_that_is_missing_fails(
    system_test: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(system_test, 'INGEST_DIR', tmp_path / 'absent')
    monkeypatch.setenv('OP_TEST_INGEST_DIR_MOUNTED', '1')
    assert system_test.test_ingest_directory() is False
