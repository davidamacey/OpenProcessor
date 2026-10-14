"""Object rows for the multi-class export: index hits to rows, dense class ids.

Turns the validated-items cohort into :class:`_ExportRow` objects (skipping
hits that cannot be exported and counting why), and renumbers registry
class ids to the dense ids written into the label files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.curation.export_images import normalized_box
from src.services.curation.export_readiness import NothingToExportError
from src.services.curation.export_support import (
    _build_export_id_map,
    _ExportRow,
    _remap_rows_to_export_ids,
)


if TYPE_CHECKING:
    from src.clients.curation_opensearch import ClassRegistryFile


def hits_to_rows(hits: list[dict[str, Any]]) -> tuple[list[_ExportRow], dict[str, int]]:
    """Object rows for every hit with an image, a usable box and a class.

    Returns ``(rows, skipped)``; ``skipped`` counts the hits left out
    by reason. A skipped hit that sits on an exported image is still
    counted there as an unlabeled object.
    """
    rows: list[_ExportRow] = []
    skipped = {'no_image_id': 0, 'no_usable_box_or_class': 0}
    for hit in hits:
        src = hit.get('_source') or {}
        item_id = src.get('crop_id') or hit.get('_id')
        box = normalized_box(src.get('bbox_norm'))
        class_id = src.get('class_id')
        if not item_id or box is None or class_id is None:
            skipped['no_usable_box_or_class'] += 1
            continue
        if not src.get('image_id'):
            # No image key to group on or to name files by.
            skipped['no_image_id'] += 1
            continue
        rows.append(
            _ExportRow(
                item_id=str(item_id),
                image_id=str(src['image_id']),
                image_path=str(src.get('image_path') or ''),
                bbox_norm=list(box),
                class_id=int(class_id),
                class_name=str(src.get('class_name') or class_id),
                has_test_crop=bool(src.get('test_holdout')),
                dataset_split=src.get('dataset_split') or None,
                import_source_stem=src.get('import_source_stem') or None,
            )
        )
    return rows, skipped


def to_dense_ids(
    rows: list[_ExportRow], registry_file: ClassRegistryFile
) -> tuple[list[_ExportRow], dict[int, int], dict[str, int]]:
    """Renumber rows to dense export ids; refuses when no row has a registered class.

    Returns ``(rows, id_map, dropped_unregistered)`` where ``dropped_unregistered``
    counts the rows left out by registry class id.
    """
    id_map = _build_export_id_map(registry_file.classes)
    dropped_unregistered: dict[str, int] = {}
    for row in rows:
        if row.class_id not in id_map:
            key = str(row.class_id)
            dropped_unregistered[key] = dropped_unregistered.get(key, 0) + 1
    rows = _remap_rows_to_export_ids(rows, id_map)
    if not rows:
        raise NothingToExportError(
            'every validated item has a class id that is not in the class registry '
            f'(or is deprecated): {dict(sorted(dropped_unregistered.items()))}'
        )
    return rows, id_map, dropped_unregistered


def dense_class_names(registry_file: ClassRegistryFile, id_map: dict[int, int]) -> list[str]:
    name_by_registry_id = {c.class_id: c.class_name for c in registry_file.classes}
    names: list[str] = [''] * len(id_map)
    for registry_id, dense_id in id_map.items():
        names[dense_id] = name_by_registry_id.get(registry_id, str(registry_id))
    return names
