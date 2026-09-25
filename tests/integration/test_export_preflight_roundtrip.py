"""Export -> preflight round trip against a REAL produced export directory.

The single highest-value test in this suite: ``src/services/curation/export.py`` used to never
write ``class_registry.json`` at all, which meant every subset-training
``include_classes`` filter both readers apply
(``src/services/training/preflight_scan.py`` and
``src/routers/curation_train.py::_unresolvable_include_classes``) was
silently checking against nothing. Existing unit tests hid this because
they hand-wrote the fixture the product itself never produced.

This test runs the REAL :meth:`GenericYoloExportService.export_dataset`,
then feeds the directory it actually wrote to the REAL
``scan_export_labels`` and ``_unresolvable_include_classes``, with
``include_classes`` set to a real subset. It must fail against the
pre-fix code (no ``class_registry.json`` on disk -> every requested class
unresolvable / the scan's filter is a no-op) and pass after the fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.clients.curation_opensearch import ClassRegistry
from src.config import CurationConfig
from src.routers.curation_train import _unresolvable_include_classes
from src.services.curation.export import GenericYoloExportService
from src.services.training import preflight_scan
from src.services.training.preflight_scan import scan_export_labels


pytestmark = pytest.mark.integration


class _FakeOpenSearch:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs

    async def search(self, index: str, body: dict[str, Any], scroll: str | None = None) -> dict:  # noqa: ARG002
        hits = [{'_id': d['crop_id'], '_source': d} for d in self._docs]
        return {'_scroll_id': 'scroll-1', 'hits': {'hits': hits}}

    async def scroll(self, scroll_id: str, scroll: str) -> dict:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, scroll_id: str) -> None:  # noqa: ARG002
        return None


@pytest.mark.asyncio
async def test_include_classes_filter_actually_applies_to_a_real_export(tmp_path):
    reg = ClassRegistry(path=tmp_path / 'registry' / 'class_registry.json')
    car_id = reg.add_class('car')
    truck_id = reg.add_class('truck')
    bus_id = reg.add_class('bus')

    # 10 cars, 10 trucks, 10 buses, one item per image so an empty-after-
    # filter image is easy to reason about.
    docs = [
        {
            'crop_id': f'{name}-{i}',
            'image_id': f'{name}-img-{i}',
            'image_path': f'{name}-{i}.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_id': cls_id,
            'class_name': name,
        }
        for cls_id, name in ((car_id, 'car'), (truck_id, 'truck'), (bus_id, 'bus'))
        for i in range(10)
    ]

    cfg = CurationConfig(export_root=tmp_path / 'exports')
    service = GenericYoloExportService(_FakeOpenSearch(docs), config=cfg, registry=reg)
    result = await service.export_dataset(seed=42)
    export_dir = Path(result.export_dir)

    # class_registry.json must actually exist -- this is the bug itself.
    registry_artifact = export_dir / 'class_registry.json'
    assert registry_artifact.is_file(), (
        'export_dataset never wrote class_registry.json -- the bug this test exists to catch'
    )

    # --- preflight_scan.scan_export_labels: filtering to just "car" -------
    # scan_export_labels's in-process cache is keyed on (export_dir,
    # manifest fingerprint) only -- not on include_classes -- so each
    # distinct include_classes value needs its own cache-clear here to
    # avoid one call's filtered result masking the next's. Pre-existing
    # cache-key design, not part of this wave's fix.
    preflight_scan._scan_cache.clear()
    scan_all = scan_export_labels(export_dir)
    assert scan_all.status == 'ok'
    assert scan_all.total_images == 30

    preflight_scan._scan_cache.clear()
    scan_car_only = scan_export_labels(export_dir, include_classes=[car_id])
    assert scan_car_only.status == 'ok'
    # Every truck/bus image has zero rows once filtered to car-only --
    # that's 20 "empty" images. If the filter were a no-op (the bug),
    # empty_label_images would be 0, same as scan_all.
    assert scan_car_only.empty_label_images == 20
    assert scan_car_only.empty_label_images != scan_all.empty_label_images

    # --- curation_train._unresolvable_include_classes ----------------------
    unresolvable = _unresolvable_include_classes(str(export_dir), [car_id, truck_id])
    assert unresolvable == [], (
        'a real, valid subset of exported classes was reported unresolvable -- '
        'the export produced no usable class_registry.json/export_id_map'
    )

    # A registry id that was never exported (or never registered) IS
    # reported unresolvable -- the check still does real work.
    bogus_id = max(car_id, truck_id, bus_id) + 100
    assert _unresolvable_include_classes(str(export_dir), [bogus_id]) == [bogus_id]
