"""Source docs -> target docs (projects plan section 6, "What is copied").

Pure transforms plus the file link. The class identity invariant lives here:
every field that carries a SOURCE registry id or a source clustering result is
dropped or rewritten, so no source number reaches the target.
"""

from __future__ import annotations

import contextlib
import copy
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.config.curation import BACKBONE_EMBEDDING_FIELD
from src.services.detection.geometry import crop_id as make_crop_id


if TYPE_CHECKING:
    from src.config.region_fields import RegionFields

# Fields that hold a source registry id, a restorable snapshot keyed by source
# ids, or a source clustering/UMAP result: none of them mean anything in the
# target. ``class_id`` is rewritten, not listed.
_SOURCE_NUMBERED_FIELDS = (
    'class_id_history',
    'edit_history',
    'excluded_prior_cluster_id',
    'excluded_prior_cluster_subid',
    'vlm_dismissed_class_id',
    'probe_pred_class_id',
    'cluster_id',
    'cluster_subid',
    'cluster_distance',
    'cluster_distance_cluster_id',
    'cluster_nearest_id',
    'cluster_auto_suggest',
    'vlm_label_cluster_id',
    'vlm_label_cluster_name',
    'vlm_label_cluster_distance',
    'import_ids',
    'imported_at',
    'import_dataset_name',
    'import_dataset_sha',
    'proposed_by_import',
)
_BOX_CLUSTER_KEYS = ('cluster_id', 'cluster_subid', 'cluster_distance')

LinkResult = Literal['linked', 'copied', 'exists']


def target_image_path(
    source_path: str, *, source_upload_root: Path, target_upload_root: Path
) -> tuple[str, Path | None]:
    """The target's path for a source image, and the file to link into it.

    An upload-root file moves to the same relative place under the target's
    upload root (so deleting the source never breaks the target); any other
    path (an archive root) is referenced as it is. The path is resolved
    first, so a ``..`` that climbs out of the upload root is not an upload-root
    file, and one that stays inside is judged by where it really lands.
    """
    path = Path(source_path).resolve()
    with contextlib.suppress(ValueError):
        rel = path.relative_to(source_upload_root.resolve())
        return str(target_upload_root / rel), path
    return source_path, None


def link_or_copy(source: Path, dest: Path) -> LinkResult:
    """Hard-link ``source`` to ``dest`` (a copy across filesystems). Never
    touches the source."""
    if dest.exists():
        return 'exists'
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)
        return 'copied'
    return 'linked'


def keep_vector(value: Any, dim: int) -> bool:
    return isinstance(value, list) and len(value) == dim


def _clean_boxes(boxes: Any) -> Any:
    if not isinstance(boxes, list):
        return boxes
    return [{k: v for k, v in b.items() if k not in _BOX_CLUSTER_KEYS} for b in boxes]


def transform_item(
    item: dict[str, Any],
    *,
    target_image_id: str,
    target_image_path_: str,
    target_class: tuple[int, str] | None,
    job_id: str,
    origin_project: str,
    now: str,
    embedding_dim: int,
    fields: RegionFields,
) -> tuple[dict[str, Any], bool]:
    """A target item doc and whether an embedding had to be dropped (a
    dimension the target cannot use). ``target_class`` is the target
    ``(class_id, class_name)``; ``None`` keeps the item unclassed."""
    doc = copy.deepcopy(item)
    for name in (*_SOURCE_NUMBERED_FIELDS, fields.class_id, fields.cluster_id):
        doc.pop(name, None)
    for name in (fields.cluster_subid, fields.cluster_distance):
        doc.pop(name, None)
    if fields.boxes in doc:
        doc[fields.boxes] = _clean_boxes(doc[fields.boxes])
    dropped = False
    for vector in ('pe_embedding', BACKBONE_EMBEDDING_FIELD):
        if vector in doc and not keep_vector(doc[vector], embedding_dim):
            if vector == 'pe_embedding':
                dropped = True
            del doc[vector]
    if target_class is None:
        doc.pop('class_id', None)
        doc.pop('class_name', None)
    else:
        doc['class_id'], doc['class_name'] = target_class
    doc['crop_id'] = make_crop_id(target_image_id, list(doc['bbox_norm']))
    doc.update(
        image_id=target_image_id,
        image_path=target_image_path_,
        import_ids=[job_id],
        origin_project=origin_project,
        origin_item_id=str(item.get('crop_id')),
        origin_image_id=str(item.get('image_id')),
        imported_at=now,
        updated_at=now,
    )
    if item.get('dataset_split'):
        doc['origin_split'] = item['dataset_split']
    return doc, dropped


def transform_image(
    image_doc: dict[str, Any],
    *,
    target_image_id: str,
    target_image_path_: str,
    job_id: str,
    origin_project: str,
    now: str,
    embedding_dim: int,
    negative_for: list[str] | None,
) -> dict[str, Any]:
    """A target images doc: the source's, re-keyed, with its provenance.
    ``negative_for`` is the reviewed-negative class list already mapped to
    target class names (``None`` when the source image is not a negative)."""
    doc = copy.deepcopy(image_doc)
    for name in ('import_ids', 'negative_for'):
        doc.pop(name, None)
    if 'pe_embedding' in doc and not keep_vector(doc['pe_embedding'], embedding_dim):
        del doc['pe_embedding']
    doc.update(
        image_id=target_image_id,
        image_path=target_image_path_,
        import_ids=[job_id],
        origin_project=origin_project,
        origin_image_id=str(image_doc.get('image_id')),
        indexed_at=now,
    )
    if negative_for is not None:
        doc['negative_for'] = negative_for
    if image_doc.get('dataset_split'):
        doc['origin_split'] = image_doc['dataset_split']
    return doc


__all__ = [
    'keep_vector',
    'link_or_copy',
    'target_image_path',
    'transform_image',
    'transform_item',
]
