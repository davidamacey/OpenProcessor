"""Dataset import job (W10.6) — REDUCED SCOPE this pass. See the wave
report for the full list of what any_domain_plan.md's W10.6/W10.11
describe that this module does NOT implement yet:

* ``options.processing`` is always treated as ``"none"``: no detector
  cascade integration, so ``"propose"`` and ``parents: "detect"`` raise
  :class:`NotImplementedError` rather than silently doing nothing.
* No persistent ledger file, no resume, no backpressure, no chunking —
  one synchronous call imports the whole scan and returns a
  :class:`DatasetImportReport`.
* No undo, no negatives/holdout stamping beyond ``dataset_split``, no
  reconciliation against a PRIOR import of overlapping images (every
  entry is treated as a fresh write; a re-import onto items another
  import already wrote is not idempotent yet).
* Region attachment only supports ``parents: "labels"`` (the region
  target's parents are this same entry's just-imported item boxes) —
  ``parents: "detect"``/``"auto"`` needs the detector cascade above.

What IS real: every write goes through the same primitives a human or
the ingest pipeline uses (``class_label`` / ``item_doc`` / W8's
``region_boxes`` / OCC), so the lock rule and the class-identity
invariant hold end to end — this is the part the class-identity E2E
test (``tests/integration/test_class_identity_e2e.py``) exercises.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from PIL import Image

from src.clients.occ import occ_update_one, occ_upsert_bulk
from src.config.region_fields import get_region_fields
from src.config.region_source import CANDIDATE_IMPORT
from src.config.region_state import RegionStatus
from src.services.curation.class_label import ItemLabel
from src.services.curation.dataset_import.mapping import RegistryClassView, ResolvedMapping
from src.services.curation.dataset_import.regions import ParentCandidate, attach_region_boxes
from src.services.curation.item_doc import DetectedItem, build_image_doc, build_item_doc
from src.services.curation.region_boxes import RegionBox, boxes_write_fields, derive_status
from src.services.detection.geometry import crop_id as _crop_id


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.clients.curation_opensearch import ClassRegistry
    from src.services.curation.dataset_import.scan import DatasetScan


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def registry_class_views(registry: ClassRegistry) -> list[RegistryClassView]:
    """Adapt a :class:`ClassRegistry` snapshot into the decoupled view
    ``mapping.py`` consumes."""
    reg = registry.load()
    return [
        RegistryClassView(
            class_id=c.class_id,
            class_name=c.class_name,
            deprecated=c.deprecated,
            merged_into=c.merged_into,
        )
        for c in reg.classes
    ]


def materialize_created_classes(resolved: ResolvedMapping, registry: ClassRegistry) -> None:
    """Call ``ClassRegistry.add_class`` for every ``create`` target,
    BEFORE any item write (W10.5), patching the assigned id into
    ``resolved.targets``."""
    for dataset_class in list(resolved.created_classes):
        target = resolved.targets[dataset_class]
        if target.class_id is not None:
            continue
        new_id = registry.add_class(target.class_name or dataset_class)
        resolved.targets[dataset_class] = dataclasses.replace(target, class_id=new_id)
        resolved.created_classes[dataset_class] = new_id


def _image_id_for(abs_path: Any) -> str:
    """Simplified image identity for this pass (see module docstring):
    sha256 of the resolved path. Deliberately NOT ingest.py's
    imohash-content-based scheme — a dataset import does not dedupe
    against an image ingest.py already wrote for the same bytes under a
    different path this pass. See the wave report."""
    return hashlib.sha256(str(abs_path).encode()).hexdigest()[:32]


@dataclass
class DatasetImportReport:
    images_created: int = 0
    items_created: int = 0
    items_updated: int = 0
    labels_written: int = 0
    boxes_written: int = 0
    standalone_regions: int = 0
    unlabeled: int = 0
    negatives: int = 0
    errors: list[str] = field(default_factory=list)


async def import_dataset(
    opensearch: AsyncOpenSearch,
    scan: DatasetScan,
    resolved: ResolvedMapping,
    *,
    import_id: str,
    images_index: str,
    items_index: str,
    label_trust: Literal['validated', 'suggestion'] = 'validated',
    region_containment: float = 0.9,
) -> DatasetImportReport:
    """Import every :class:`~...scan.ScanEntry` in ``scan`` under the
    resolved class mapping. See the module docstring for what's
    deliberately not implemented this pass."""
    report = DatasetImportReport()
    now = _now_iso()
    region_fields = get_region_fields()

    for entry in scan.entries:
        image_id = _image_id_for(entry.abs_image_path)
        width = height = 0
        try:
            with Image.open(entry.abs_image_path) as img:
                width, height = img.size
        except Exception as exc:  # pragma: no cover - defensive; fixtures always decode
            report.errors.append(f'{entry.rel_path}: image open failed: {exc}')

        item_boxes: list[tuple[Any, Any]] = []
        region_boxes_pending = []
        for box in entry.boxes:
            target = resolved.targets.get(box.dataset_class)
            if target is None or target.kind == 'skip':
                continue
            if target.kind == 'item':
                item_boxes.append((box, target))
            elif target.kind == 'region':
                region_boxes_pending.append(box)

        if not item_boxes and not region_boxes_pending:
            report.unlabeled += 1
            continue
        if entry.label_state == 'negative' and not item_boxes and not region_boxes_pending:
            report.negatives += 1
            continue

        image_doc = build_image_doc(
            image_id=image_id,
            image_path=str(entry.abs_image_path),
            source=f'import:{import_id}',
            width=width,
            height=height,
            imohash=image_id,
            now=now,
        )
        await opensearch.bulk(
            body=[{'index': {'_index': images_index, '_id': image_id}}, image_doc], refresh=False
        )
        report.images_created += 1

        item_docs: list[dict[str, Any]] = []
        parent_candidates: list[ParentCandidate] = []
        for box, item_target in item_boxes:
            cid = _crop_id(image_id, list(box.bbox_norm))
            label = ItemLabel.imported(
                import_id=import_id,
                class_id=item_target.class_id,
                class_name=item_target.class_name,
                trust=label_trust,
                now=now,
            )
            item = DetectedItem(
                bbox_pixel=(0.0, 0.0, 0.0, 0.0),
                score=1.0,
                class_id=item_target.class_id,
                class_name=item_target.class_name,
                label=label,
            )
            area = max(0.0, box.bbox_norm[2] - box.bbox_norm[0]) * max(
                0.0, box.bbox_norm[3] - box.bbox_norm[1]
            )
            doc = build_item_doc(
                crop_id=cid,
                image_id=image_id,
                image_path=str(entry.abs_image_path),
                source=f'import:{import_id}',
                request_id=import_id,
                bbox_norm=list(box.bbox_norm),
                item=item,
                now=now,
                crop_area_norm=area,
                crop_rank_in_image=1,
                blur_full_var=None,
                blur_lap_var=None,
                blur_lap_ratio=None,
            )
            doc['dataset_split'] = entry.split
            doc['import_ids'] = [import_id]
            doc['imported_at'] = now
            doc['import_dataset_name'] = import_id
            item_docs.append(doc)
            parent_candidates.append(
                ParentCandidate(key=cid, bbox_norm=box.bbox_norm, class_name=item_target.class_name)
            )

        if item_docs:
            upsert = await occ_upsert_bulk(
                opensearch,
                item_docs,
                index=items_index,
                human_field_guards=['label_source', 'class_source'],
                writer_id=f'import:{import_id}',
            )
            report.items_created += upsert['created']
            report.items_updated += upsert['updated']
            report.labels_written += len(item_docs)

        if region_boxes_pending:
            attach_result = attach_region_boxes(
                parent_candidates, region_boxes_pending, containment_threshold=region_containment
            )
            by_parent: dict[str, list] = {}
            for attachment in attach_result.attachments:
                by_parent.setdefault(attachment.parent_key, []).append(attachment.box)
            for parent_key, label_boxes in by_parent.items():
                region_box_list = [
                    RegionBox(
                        box_id=f'b{i + 1}',
                        bbox_norm=lb.bbox_norm,
                        state='accepted',
                        score=1.0,
                        detector='import',
                        detector_version=import_id,
                        source=CANDIDATE_IMPORT,
                        detected_at=now,
                    )
                    for i, lb in enumerate(label_boxes)
                ]
                fields = boxes_write_fields(region_box_list, set_complete=True, F=region_fields)
                fields[region_fields.status] = derive_status(
                    region_box_list, empty_status=RegionStatus.NO_REGION_VISIBLE
                ).value
                fields[region_fields.validated] = True
                fields[region_fields.label_source] = 'import'
                fields[region_fields.verifier] = 'import'
                fields[region_fields.verifier_version] = import_id

                def _region_merger(
                    _current: dict[str, Any], _fields: dict[str, Any] = fields
                ) -> dict[str, Any]:
                    return _fields

                try:
                    await occ_update_one(
                        opensearch,
                        doc_id=parent_key,
                        merger=_region_merger,
                        index=items_index,
                        writer_id=f'import:{import_id}',
                    )
                    report.boxes_written += len(label_boxes)
                except Exception as exc:
                    report.errors.append(f'{parent_key}: region write failed: {exc}')
            report.standalone_regions += len(attach_result.standalone)

    return report


__all__ = [
    'DatasetImportReport',
    'import_dataset',
    'materialize_created_classes',
    'registry_class_views',
]
