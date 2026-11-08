"""P2 §5.5 (owner D1): src.services.training.model_classes.

Class identity invariant: a model's class NAME crosses a project
boundary, never its raw dense id. These tests cover the three plan
cases directly (exact match, case-insensitive match, no match) plus the
class-source ordering (the live promotion's labels win over labels.txt).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.clients.curation_opensearch import ClassRegistryFile, RegistryClassEntry
from src.services.training.model_classes import (
    ModelClass,
    is_model_shared,
    model_class_mapping,
    model_classes,
    model_owner_project,
)


if TYPE_CHECKING:
    from pathlib import Path

    from _fake_model_registry import FakeModelRegistry

pytestmark = pytest.mark.asyncio


def _registry(*entries: tuple[int, str]) -> ClassRegistryFile:
    return ClassRegistryFile(
        classes=[RegistryClassEntry(class_id=cid, class_name=name) for cid, name in entries]
    )


async def test_model_classes_prefers_the_promotion_labels_over_labels_txt(
    tmp_path: Path, monkeypatch, model_registry_fake: FakeModelRegistry
) -> None:
    """§5.5 #1/#2: the live promotion's labels (model order) win outright."""
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    model_dir = tmp_path / 'cars__det_v3'
    model_dir.mkdir()
    model_registry_fake.promote(
        'cars__det_v3', owner='cars', labels=['truck', 'car'], shared_with=['wheels']
    )
    (model_dir / 'labels.txt').write_text('wrong\nnames\n')

    assert await model_classes('cars__det_v3') == [
        ModelClass(0, 'truck'),
        ModelClass(1, 'car'),
    ]
    assert await is_model_shared('cars__det_v3') is True
    assert await model_owner_project('cars__det_v3') == 'cars'


async def test_model_classes_falls_back_to_labels_txt_for_a_base_model(
    tmp_path: Path, monkeypatch
) -> None:
    """A base/core-pipeline model has no promotion at all."""
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    model_dir = tmp_path / 'yolov11_small_trt_end2end'
    model_dir.mkdir()
    (model_dir / 'labels.txt').write_text('person\nbicycle\ncar\n')

    from src.utils.class_names import clear_class_name_cache

    clear_class_name_cache()
    assert await model_classes('yolov11_small_trt_end2end') == [
        ModelClass(0, 'person'),
        ModelClass(1, 'bicycle'),
        ModelClass(2, 'car'),
    ]
    assert await is_model_shared('yolov11_small_trt_end2end') is False
    assert await model_owner_project('yolov11_small_trt_end2end') is None


async def test_model_class_mapping_exact_case_insensitive_and_none(
    tmp_path: Path, monkeypatch, model_registry_fake: FakeModelRegistry
) -> None:
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    model_registry_fake.promote(
        'cars__det_v3', owner='cars', labels=['car', 'Truck', 'van'], shared_with=['wheels']
    )
    registry = _registry((2, 'car'), (7, 'truck'), (9, 'bus'))

    mapping = await model_class_mapping('cars__det_v3', registry, project='wheels')

    by_id = {e.model_id: e for e in mapping.entries}
    assert by_id[0].class_id == 2
    assert by_id[0].match == 'exact'
    assert by_id[1].class_id == 7
    assert by_id[1].match == 'case_insensitive'
    assert by_id[2].class_id is None
    assert by_id[2].match == 'none'
    assert mapping.unmapped == ('van',)
    assert mapping.not_covered == ('bus',)
    assert mapping.model_project == 'cars'
    assert mapping.project == 'wheels'


async def test_a_renamed_class_is_unmapped_even_in_its_own_project(
    tmp_path: Path, monkeypatch, model_registry_fake: FakeModelRegistry
) -> None:
    """§5.5 #4: this runs for every model, own-project included -- a
    class renamed since training shows up as unmapped there too."""
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    model_registry_fake.promote('cars__det_v3', owner='cars', labels=['sedan'])
    registry = _registry((0, 'sedan_renamed'))

    mapping = await model_class_mapping('cars__det_v3', registry, project='cars')
    assert mapping.unmapped == ('sedan',)
    assert mapping.not_covered == ('sedan_renamed',)
