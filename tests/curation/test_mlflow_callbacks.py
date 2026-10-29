"""Tests for ``docker/trainer/mlflow_callbacks.py``'s lineage tagging.

MLflow tags (``on_train_start``) and registry metadata (``on_train_end``)
must be byte-identical to the run manifest's ``lineage``/``code_versions``
block -- both this module and ``job_manifest.write_manifest`` read the same
``job_manifest.build_lineage(spec)`` dict, so a drift here would mean the
tracked run and its manifest disagree about which dataset/build produced it.

Exercised with a fake ``mlflow`` module (no tracking server needed) and the
same ``sys.path`` trick the ``trainer_protocol`` conftest uses to import the
trainer's flat module layout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest


TRAINER_DIR = Path(__file__).resolve().parents[2] / 'docker' / 'trainer'
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

import mlflow_callbacks  # noqa: E402


class _FakeTrainerObj:
    """Minimal stand-in for Ultralytics' ``BaseTrainer`` the callbacks read."""

    def __init__(self, save_dir: Path) -> None:
        self.save_dir = str(save_dir)
        self.epoch = 0


class _FakeModel:
    def __init__(self) -> None:
        self.callbacks: dict[str, list[Any]] = {}

    def add_callback(self, event: str, fn: Any) -> None:
        self.callbacks.setdefault(event, []).append(fn)


class _FakeMlflow:
    """Just enough of the ``mlflow`` module surface for these two hooks."""

    def __init__(self) -> None:
        self.tags: dict[str, str] | None = None
        self.params: dict[str, Any] = {}

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags = tags

    def log_param(self, key: str, value: Any) -> None:
        self.params[key] = value

    def log_dict(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def log_artifact(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def active_run(self) -> None:
        return None  # no tracking server -- on_train_end stops before the registry push


@pytest.fixture
def lineage() -> dict[str, Any]:
    return {
        'dataset_sha': 'dsha123',
        'frozen_test_sha': 'ftsha123',
        'test_label_sha': 'tlsha123',
        'dataset_version_tag': 'v3',
        'api_sha': 'apisha123',
        'trainer_sha': 'trainersha123',
        'trainer_image_id': 'sha256:trainerimg',
    }


def _register(model: _FakeModel, tmp_path: Path, lineage: dict[str, Any]) -> None:
    data_yaml_path = tmp_path / 'data.yaml'
    data_yaml_path.write_text('names:\n  0: a\n')
    mlflow_callbacks.register_callbacks(
        model,
        run_name='run-1',
        profile='probe',
        seed=42,
        lineage=lineage,
        data_cfg={'names': {0: 'a'}},
        data_yaml_path=data_yaml_path,
    )


def test_on_train_start_tags_are_byte_identical_to_lineage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lineage: dict[str, Any]
) -> None:
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr(mlflow_callbacks, '_safe_mlflow', lambda: fake_mlflow)

    model = _FakeModel()
    _register(model, tmp_path, lineage)

    save_dir = tmp_path / 'run'
    save_dir.mkdir()
    for fn in model.callbacks['on_train_start']:
        fn(_FakeTrainerObj(save_dir))

    assert fake_mlflow.tags == {
        'git_sha': lineage['trainer_sha'],
        'dataset_sha': lineage['dataset_sha'],
        'dataset_version': lineage['dataset_version_tag'],
        'frozen_test_sha': lineage['frozen_test_sha'],
        'test_label_sha': lineage['test_label_sha'],
        'api_sha': lineage['api_sha'],
        'docker_digest_trainer': lineage['trainer_image_id'],
        'profile': 'probe',
        'seed': '42',
        'run_name': 'run-1',
    }


def test_on_train_end_registry_metadata_is_byte_identical_to_lineage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lineage: dict[str, Any]
) -> None:
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr(mlflow_callbacks, '_safe_mlflow', lambda: fake_mlflow)

    model = _FakeModel()
    _register(model, tmp_path, lineage)

    save_dir = tmp_path / 'run'
    (save_dir / 'weights').mkdir(parents=True)
    (save_dir / 'weights' / 'best.pt').write_bytes(b'stub-checkpoint')

    for fn in model.callbacks['on_train_end']:
        fn(_FakeTrainerObj(save_dir))

    metadata = json.loads((save_dir / 'run_metadata.json').read_text())
    assert metadata['dataset_sha'] == lineage['dataset_sha']
    assert metadata['dataset_version'] == lineage['dataset_version_tag']
    assert metadata['frozen_test_sha'] == lineage['frozen_test_sha']
    assert metadata['test_label_sha'] == lineage['test_label_sha']
    assert metadata['api_sha'] == lineage['api_sha']
    assert metadata['git_sha'] == lineage['trainer_sha']
    assert metadata['docker_digest_trainer'] == lineage['trainer_image_id']


class _FakeRegistryClient:
    """Records registry calls; any stage API raises like a removed MLflow API."""

    instances: list[_FakeRegistryClient] = []

    def __init__(self) -> None:
        self.aliases: list[tuple[str, str, Any]] = []
        self.versions: list[dict[str, Any]] = []
        _FakeRegistryClient.instances.append(self)

    def get_registered_model(self, _name: str) -> None:
        return None

    def create_model_version(self, **kwargs: Any) -> Any:
        self.versions.append(kwargs)

        class _Version:
            version = '7'

        return _Version()

    def set_registered_model_alias(self, name: str, alias: str, version: Any) -> None:
        self.aliases.append((name, alias, version))

    def transition_model_version_stage(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError('registry stages are deprecated; use an alias (#154)')


def test_register_model_sets_the_staging_alias_and_never_a_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lineage: dict[str, Any]
) -> None:
    import types

    fake_tracking = types.ModuleType('mlflow.tracking')
    fake_tracking.MlflowClient = _FakeRegistryClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'mlflow', types.ModuleType('mlflow'))
    monkeypatch.setitem(sys.modules, 'mlflow.tracking', fake_tracking)
    _FakeRegistryClient.instances.clear()
    model = _FakeModel()
    _register(model, tmp_path, lineage)
    callbacks = model.callbacks['on_train_end']
    # the callback object owning _register_model is the bound method's instance
    owner = callbacks[0].__self__

    owner._register_model('run-abc', {'dataset_sha': 'x'})

    (client,) = _FakeRegistryClient.instances
    assert client.aliases == [(owner.registry_name, 'staging', '7')]
