"""Uploaded checkpoints are read in the restricted (weights-only) mode."""

from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip('torch')
pytest.importorskip('ultralytics')

from src.services.model_export import validate_pytorch_model  # noqa: E402  # after importorskip


def _write_marker(path: str) -> None:
    Path(path).write_text('ran')


class _NotWeights:
    """Harmless stand-in for a non-weights object: unpickling it would call a function."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (_write_marker, (str(self.marker),))


def test_checkpoint_with_a_non_weights_object_is_refused(tmp_path: Path) -> None:
    marker = tmp_path / 'marker.txt'
    ckpt = tmp_path / 'bad.pt'
    torch.save({'model': _NotWeights(marker), 'pad': torch.zeros(2000)}, ckpt)

    ok, msg, info = validate_pytorch_model(ckpt)

    assert not ok, msg
    assert info == {}
    assert not marker.exists()


def test_legitimate_yolo_checkpoint_still_validates(tmp_path: Path) -> None:
    from ultralytics import YOLO

    ckpt = tmp_path / 'tiny.pt'
    YOLO('yolo11n.yaml').save(str(ckpt))

    ok, msg, info = validate_pytorch_model(ckpt)

    assert ok, msg
    assert info['task'] == 'detect'
    assert info['num_classes'] == 80
    assert info['end2end'] is False


def test_upload_is_refused_when_restricted_loading_is_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # torch_safe_load silently drops safe_only on a torch that cannot restrict
    # unpickling; validation must refuse rather than load without restriction.
    from ultralytics.nn import tasks

    monkeypatch.setattr(tasks._SafeLoad, 'SUPPORTED', False)
    marker = tmp_path / 'marker.txt'
    ckpt = tmp_path / 'bad.pt'
    torch.save({'model': _NotWeights(marker), 'pad': torch.zeros(2000)}, ckpt)

    ok, msg, _ = validate_pytorch_model(ckpt)

    assert not ok
    assert 'Restricted model loading is unavailable' in msg
    assert not marker.exists()
