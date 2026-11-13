"""The materialized export datasets under the project's export root: the one scan the export
list, the last-export status and the dashboard's export step all read."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.curation.export import resolve_current_export_dir


if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

# Dataset kinds on the wire — the same ids the /methods `export` axis
# advertises for POST /export/{kind}.
MULTI_CLASS_DATASET_KIND = 'yolo'
SINGLE_CLASS_DATASET_KIND = 'single_class'


def read_manifest(d: Path) -> dict[str, Any] | None:
    from src.services.curation.export_models import ARTIFACT_FILENAMES

    manifest = d / ARTIFACT_FILENAMES['manifest']
    if not manifest.is_file():
        return None
    try:
        meta = json.loads(manifest.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        logger.warning('export_manifest_skip', dir=str(d), error=str(exc))
        return None
    return meta if isinstance(meta, dict) else None


def dataset_row(
    d: Path, meta: dict[str, Any], *, kind: str, profile_name: str | None, current: str | None
) -> dict[str, Any]:
    return {
        'kind': kind,
        'profile_name': profile_name,
        'export_dir': str(d),
        'version_tag': meta.get('version_tag') or '',
        'image_count': meta.get('image_count'),
        'object_count': meta.get('object_count'),
        'split_counts': meta.get('split_counts'),
        'dataset_sha': meta.get('dataset_sha'),
        'exported_at': meta.get('exported_at') or meta.get('started_at'),
        # class_count is the registry size written into data.yaml
        # (nc); classes_with_objects is how many of those actually have a
        # labeled object in this export -- older manifests written before
        # this field existed serve null, never a fabricated 0.
        'class_count': meta.get('class_count'),
        'classes_with_objects': meta.get('classes_with_objects'),
        'is_current': current is not None and str(d) == current,
    }


def scan_export_datasets() -> list[dict[str, Any]]:
    """Every materialized dataset version under the export root.

    Multi-class versions live directly under the root
    (``<root>/<version>/manifest.json``). Single-class exports live one
    level down under their profile's own root
    (``<root>/<profile_name>/<version>/manifest.json``), each with its own
    ``current`` symlink, so a root child without a manifest is scanned as
    a profile root.
    """
    from src.services.curation.export_models import SingleClassExportProfile
    from src.services.curation.export_single_class import resolve_current_single_class_dir

    cfg = get_curation_config()
    root = cfg.export_root
    if not root.is_dir():
        return []
    try:
        current: str | None = str(resolve_current_export_dir(cfg))
    except FileNotFoundError:
        current = None
    rows: list[dict[str, Any]] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name == 'current':
            continue
        meta = read_manifest(d)
        if meta is not None:
            rows.append(
                dataset_row(
                    d, meta, kind=MULTI_CLASS_DATASET_KIND, profile_name=None, current=current
                )
            )
            continue
        profile = SingleClassExportProfile(name=d.name)
        try:
            profile_current: str | None = str(resolve_current_single_class_dir(profile, cfg))
        except FileNotFoundError:
            profile_current = None
        for v in sorted(d.iterdir()):
            if not v.is_dir() or v.name == profile.current_link_name:
                continue
            vmeta = read_manifest(v)
            if vmeta is None:
                continue
            rows.append(
                dataset_row(
                    v,
                    vmeta,
                    kind=SINGLE_CLASS_DATASET_KIND,
                    profile_name=d.name,
                    current=profile_current,
                )
            )
    return rows
