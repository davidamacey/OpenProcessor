"""Fixtures shared by the primary and secondary class-by-name tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from curation.test_class_resolution_by_name import _Entry, _Registry
from src.config import DetectionProfile
from src.services.curation.ingest_detect import WholeImageDetector


if TYPE_CHECKING:
    from pathlib import Path

    import pytest

# model output index -> name; registry id 3 carries the name of model index 2
MODEL_NAMES = ['alpha', 'bravo', 'delta']
GAPPED = (_Entry(0, 'alpha'), _Entry(1, 'bravo'), _Entry(3, 'delta'))
IN_ORDER = (_Entry(0, 'alpha'), _Entry(1, 'bravo'), _Entry(2, 'delta'))


def write_model_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """The model's labels file, inside the (patched) model repository."""
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    path = tmp_path / 'labels.txt'
    path.write_text('\n'.join(MODEL_NAMES) + '\n')
    return str(path)


def primary_detector(labels: str, registry: _Registry) -> WholeImageDetector:
    profile = DetectionProfile(
        name='item', detector_model='prim', assigns_class=True, labels_path=labels
    )
    return WholeImageDetector(triton_pool=None, registry=registry, profile=profile)
