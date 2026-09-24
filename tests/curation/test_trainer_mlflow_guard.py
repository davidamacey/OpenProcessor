"""ML-1 defense in depth: ``_guard_ultralytics_mlflow_artifact_root``.

The primary fix is serving MLflow artifacts through the tracking
server's HTTP proxy (``docker-compose.yml``'s ``curation-mlflow``
command: ``--serve-artifacts --default-artifact-root=mlflow-artifacts:/``).
This test covers the trainer-side guard that disables Ultralytics' own
(un-try/excepted) MLflow integration when an experiment's
``artifact_location`` is still a local path the process cannot write --
the exact condition that made every training run fail at
``on_train_end`` before the fix.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock


if TYPE_CHECKING:
    import pytest


TRAINER_DIR = Path(__file__).resolve().parents[2] / 'docker' / 'trainer'
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

import trainer  # noqa: E402


def test_guard_disables_ultralytics_mlflow_when_artifact_root_is_unwritable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Local artifact_location under a directory the process can't create -> disabled."""
    unwritable_parent = tmp_path / 'no_perm'
    unwritable_parent.mkdir()
    unwritable_parent.chmod(0o500)  # read+execute only, no write
    bad_location = str(unwritable_parent / 'sub' / '0')

    fake_experiment = MagicMock(artifact_location=bad_location)
    fake_mlflow = MagicMock()
    fake_mlflow.get_experiment_by_name.return_value = fake_experiment

    fake_settings = MagicMock()
    fake_settings.get.return_value = True

    monkeypatch.setitem(sys.modules, 'mlflow', fake_mlflow)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils', MagicMock(SETTINGS=fake_settings))

    try:
        trainer._guard_ultralytics_mlflow_artifact_root('openprocessor')
    finally:
        unwritable_parent.chmod(0o700)

    fake_settings.update.assert_called_once_with({'mlflow': False})


def test_guard_is_a_noop_for_proxied_artifact_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """``mlflow-artifacts:/...`` (the fixed default) must not be touched."""
    fake_experiment = MagicMock(artifact_location='mlflow-artifacts:/0')
    fake_mlflow = MagicMock()
    fake_mlflow.get_experiment_by_name.return_value = fake_experiment

    fake_settings = MagicMock()
    fake_settings.get.return_value = True

    monkeypatch.setitem(sys.modules, 'mlflow', fake_mlflow)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils', MagicMock(SETTINGS=fake_settings))

    trainer._guard_ultralytics_mlflow_artifact_root('openprocessor')

    fake_settings.update.assert_not_called()


def test_guard_is_a_noop_when_ultralytics_mlflow_integration_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = MagicMock()
    fake_settings = MagicMock()
    fake_settings.get.return_value = False  # Ultralytics' own integration is off

    monkeypatch.setitem(sys.modules, 'mlflow', fake_mlflow)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils', MagicMock(SETTINGS=fake_settings))

    trainer._guard_ultralytics_mlflow_artifact_root('openprocessor')

    fake_mlflow.get_experiment_by_name.assert_not_called()
    fake_settings.update.assert_not_called()
