"""Every shipped example region profile must gate the stage by class NAME.

An empty ``parent_classes`` means every item gets the region stage (GH #48:
the segmenter ran on people and food crops of a public COCO set)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.curation.region_scope import in_parent_classes
from src.services.detection.profile_registry import region_profile_from_file


EXAMPLES = sorted(
    (Path(__file__).resolve().parents[1] / 'examples' / 'region_profiles').glob('*.json')
)
VEHICLES = ['car', 'truck', 'bus', 'motorcycle', 'Car', 'TRUCK']
NOT_VEHICLES = ['person', 'pizza', 'chair', 'dog', 'traffic light']


def _selects(path: Path, name: str) -> bool:
    profile = region_profile_from_file(str(path))
    return in_parent_classes(profile.parent_classes, class_name=name, proposal_name=None)


def _example(name: str) -> Path:
    return next(p for p in EXAMPLES if p.stem == name)


@pytest.mark.parametrize('path', EXAMPLES, ids=lambda p: p.stem)
def test_every_example_profile_scopes_the_stage_to_named_classes(path: Path) -> None:
    assert region_profile_from_file(str(path)).parent_classes


@pytest.mark.parametrize('name', VEHICLES)
def test_license_plate_selects_vehicle_items(name: str) -> None:
    assert _selects(_example('license_plate'), name)


@pytest.mark.parametrize('name', NOT_VEHICLES)
@pytest.mark.parametrize('example', ['license_plate', 'vehicle_wheel'])
def test_examples_reject_non_vehicle_items(example: str, name: str) -> None:
    assert not _selects(_example(example), name)


def test_vehicle_wheel_selects_cars() -> None:
    assert _selects(_example('vehicle_wheel'), 'car')
