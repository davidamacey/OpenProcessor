"""P2 §5.5 (owner D1): src.services.training.model_classes.

Class identity invariant: a model's class NAME crosses a project
boundary, never its raw dense id. These tests cover the three plan
cases directly (exact match, case-insensitive match, no match) plus the
class-remap-source ordering (promote.json.classes wins over labels.txt).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

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


def _registry(*entries: tuple[int, str]) -> ClassRegistryFile:
    return ClassRegistryFile(
        classes=[RegistryClassEntry(class_id=cid, class_name=name) for cid, name in entries]
    )


def test_model_classes_prefers_promote_json_over_labels_txt(tmp_path: Path, monkeypatch) -> None:
    """§5.5 #1/#2: promote.json.classes (model order) wins outright."""
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    model_dir = tmp_path / 'cars__det_v3'
    model_dir.mkdir()
    (model_dir / 'promote.json').write_text(
        json.dumps(
            {
                'project': 'cars',
                'shared': True,
                'classes': [{'model_id': 0, 'name': 'truck'}, {'model_id': 1, 'name': 'car'}],
            }
        )
    )
    (model_dir / 'labels.txt').write_text('wrong\nnames\n')

    assert model_classes('cars__det_v3') == [
        ModelClass(0, 'truck'),
        ModelClass(1, 'car'),
    ]
    assert is_model_shared('cars__det_v3') is True
    assert model_owner_project('cars__det_v3') == 'cars'


def test_model_classes_falls_back_to_labels_txt_for_a_base_model(
    tmp_path: Path, monkeypatch
) -> None:
    """A base/core-pipeline model has no promote.json at all."""
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    model_dir = tmp_path / 'yolov11_small_trt_end2end'
    model_dir.mkdir()
    (model_dir / 'labels.txt').write_text('person\nbicycle\ncar\n')

    from src.utils.class_names import clear_class_name_cache

    clear_class_name_cache()
    assert model_classes('yolov11_small_trt_end2end') == [
        ModelClass(0, 'person'),
        ModelClass(1, 'bicycle'),
        ModelClass(2, 'car'),
    ]
    assert is_model_shared('yolov11_small_trt_end2end') is False
    assert model_owner_project('yolov11_small_trt_end2end') is None


def test_model_class_mapping_exact_case_insensitive_and_none(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    model_dir = tmp_path / 'cars__det_v3'
    model_dir.mkdir()
    (model_dir / 'promote.json').write_text(
        json.dumps(
            {
                'project': 'cars',
                'shared': True,
                'classes': [
                    {'model_id': 0, 'name': 'car'},
                    {'model_id': 1, 'name': 'Truck'},
                    {'model_id': 2, 'name': 'van'},
                ],
            }
        )
    )
    registry = _registry((2, 'car'), (7, 'truck'), (9, 'bus'))

    mapping = model_class_mapping('cars__det_v3', registry, project='wheels')

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


def test_a_renamed_class_is_unmapped_even_in_its_own_project(tmp_path: Path, monkeypatch) -> None:
    """§5.5 #4: this runs for every model, own-project included -- a
    class renamed since training shows up as unmapped there too."""
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    model_dir = tmp_path / 'cars__det_v3'
    model_dir.mkdir()
    (model_dir / 'promote.json').write_text(
        json.dumps(
            {
                'project': 'cars',
                'shared': False,
                'classes': [{'model_id': 0, 'name': 'sedan'}],
            }
        )
    )
    registry = _registry((0, 'sedan_renamed'))

    mapping = model_class_mapping('cars__det_v3', registry, project='cars')
    assert mapping.unmapped == ('sedan',)
    assert mapping.not_covered == ('sedan_renamed',)
