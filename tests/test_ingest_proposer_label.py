"""Found live: with OP_INGEST_PRIMARY_LABELS_PATH unset every proposal was
named by its bare class id ('2'), so a region profile's ``parent_classes:
['car']`` matched nothing and the wheel example silently produced no regions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.config.detection_profile import DetectionProfile
from src.config.ingest_profiles import proposer_label
from src.utils.class_names import clear_class_name_cache


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    clear_class_name_cache()
    yield tmp_path
    clear_class_name_cache()


def test_an_unconfigured_labels_path_uses_the_models_own_labels(models_dir: Path) -> None:
    (models_dir / 'detector').mkdir()
    (models_dir / 'detector' / 'labels.txt').write_text('person\nbicycle\ncar\n')

    profile = DetectionProfile(name='item', detector_model='detector')

    assert proposer_label(profile, 2) == 'car'


def test_a_model_without_labels_falls_back_to_the_bare_id(models_dir: Path) -> None:
    profile = DetectionProfile(name='item', detector_model='no_such_model')

    assert proposer_label(profile, 2) == '2'


def test_a_configured_labels_path_wins(models_dir: Path, tmp_path: Path) -> None:
    (models_dir / 'detector').mkdir()
    (models_dir / 'detector' / 'labels.txt').write_text('a\nb\nc\n')
    mine = tmp_path / 'mine.txt'
    mine.write_text('x\ny\nz\n')

    profile = DetectionProfile(name='item', detector_model='detector', labels_path=str(mine))

    assert proposer_label(profile, 2) == 'z'
