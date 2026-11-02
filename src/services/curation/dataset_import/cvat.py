"""CVAT "for images 1.1" XML dataset reading.

Layout::

    root/annotations.xml          (<annotations><image name=... width=... height=...>)
    root/images/<name>            (else the root)

Classes are the ``label`` attribute texts; CVAT has no class index, so
``DatasetScan.class_ids`` stays empty. Only axis-aligned ``<box>`` shapes are
boxes: a rotated box and every other shape (polygon, polyline, points,
ellipse, mask, cuboid, skeleton) is skipped with ``shape_not_box_skipped``
and still counts as present, so the image is ``unlabeled``, not ``negative``.
An image with no shape at all is ``negative``. ``<image name>`` resolves
through :func:`~src.services.curation.dataset_import.paths.resolve_ref`.
The XML is untrusted: size-capped, every DTD refused
(:mod:`~src.services.curation.dataset_import.safe_xml`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.services.curation.dataset_import.capped_io import read_bytes_capped
from src.services.curation.dataset_import.issues import IssueCollector
from src.services.curation.dataset_import.limits import MAX_CVAT_XML_BYTES, preview_max_files
from src.services.curation.dataset_import.native_common import (
    SizeBudget,
    box_from_coords,
    declared_size,
    entry_state,
    find_image,
    finish_scan,
    read_head,
    rel_path,
    resolve_size,
    scan_too_large,
    split_of_dirname,
)
from src.services.curation.dataset_import.paths import DatasetPathNotAllowedError, PathGuard
from src.services.curation.dataset_import.safe_xml import parse_xml_bytes
from src.services.curation.dataset_import.scan import DatasetScan, LabelBox, ScanEntry


if TYPE_CHECKING:
    import xml.etree.ElementTree as ET  # nosec B405  typing only; parsing goes through safe_xml
    from pathlib import Path

_FILE = 'annotations.xml'
_OTHER_SHAPES = frozenset(
    {'polygon', 'polyline', 'points', 'ellipse', 'mask', 'cuboid', 'skeleton'}
)


def is_cvat(root: Path) -> bool:
    """``annotations.xml`` whose head opens an ``<annotations>`` element."""
    path = root / _FILE
    return root.is_dir() and path.is_file() and '<annotations' in read_head(path, 4096)


@dataclass
class _Ctx:
    root: Path
    images_dirs: list[Path]
    issues: IssueCollector
    path_guard: PathGuard | None
    budget: SizeBudget


def _float(el: ET.Element, key: str) -> float:
    value = float(el.get(key, ''))
    if not math.isfinite(value):
        raise ValueError(key)
    return value


def _cvat_boxes(
    image_el: ET.Element, size: tuple[int, int], rel_file: str, ctx: _Ctx
) -> tuple[list[LabelBox], bool]:
    boxes: list[LabelBox] = []
    present = False
    for index, shape in enumerate(image_el):
        if shape.tag in _OTHER_SHAPES:
            present = True
            ctx.issues.add(
                'shape_not_box_skipped',
                file=rel_file,
                detail={'object': index, 'shape': shape.tag},
            )
            continue
        if shape.tag != 'box':
            continue  # <tag> is image-level classification, not geometry
        present = True
        try:
            rotation = float(shape.get('rotation') or 0)
            if abs(rotation) > 1e-6:
                ctx.issues.add(
                    'shape_not_box_skipped',
                    file=rel_file,
                    detail={'object': index, 'shape': 'rotated box'},
                )
                continue
            coords = (
                _float(shape, 'xtl'),
                _float(shape, 'ytl'),
                _float(shape, 'xbr'),
                _float(shape, 'ybr'),
            )
        except ValueError:
            ctx.issues.add(
                'label_row_malformed',
                file=rel_file,
                detail={'object': index, 'reason': 'bad coordinates'},
            )
            continue
        box = box_from_coords(
            (shape.get('label') or '').strip(),
            coords,
            size=size,
            rel_file=rel_file,
            index=index,
            issues=ctx.issues,
        )
        if box is not None:
            boxes.append(box)
    return boxes, present


def _read_image(image_el: ET.Element, ctx: _Ctx) -> ScanEntry | None:
    name = (image_el.get('name') or '').strip()
    rel_file = f'{_FILE}#{name}'[:200]
    image = find_image(
        ctx.images_dirs,
        [name] if name else [],
        rel_file=rel_file,
        issues=ctx.issues,
        path_guard=ctx.path_guard,
    )
    if image is None:
        return None
    sized = resolve_size(
        declared_size(image_el.get('width'), image_el.get('height')),
        image,
        rel_file,
        ctx.issues,
        ctx.budget,
    )
    if sized is None:
        return None
    size, declared = sized
    boxes, present = _cvat_boxes(image_el, size, rel_file, ctx)
    return ScanEntry(
        rel_path=rel_path(image, ctx.root),
        source_stem=image.stem,
        abs_image_path=image,
        split=split_of_dirname((image_el.get('subset') or '').strip()),
        label_state=entry_state(present, boxes),
        boxes=boxes,
        declared_size=declared,
    )


def _parse(path: Path, issues: IssueCollector) -> ET.Element | None:
    raw = read_bytes_capped(path, MAX_CVAT_XML_BYTES)
    if raw is None:
        if path.is_file() and path.stat().st_size > MAX_CVAT_XML_BYTES:
            issues.add('dataset_file_too_large', file=path.name)
        else:
            issues.add('cvat_xml_invalid', file=path.name, detail={'reason': 'unreadable'})
        return None
    try:
        root_el = parse_xml_bytes(raw)
    except ValueError as exc:
        issues.add('cvat_xml_invalid', file=path.name, detail={'reason': str(exc)[:200]})
        return None
    if root_el.tag != 'annotations':
        issues.add(
            'cvat_xml_invalid',
            file=path.name,
            detail={'reason': 'root element is not <annotations>'},
        )
        return None
    return root_el


def scan_cvat(root: Path, *, path_guard: PathGuard | None = None) -> DatasetScan:
    if path_guard is not None and not path_guard(root):
        raise DatasetPathNotAllowedError(str(root))
    issues = IssueCollector()
    path = root / _FILE
    if not path.is_file():
        issues.add('cvat_xml_invalid', file=_FILE, detail={'reason': 'annotations.xml not found'})
        return finish_scan('cvat', root, [], issues)
    root_el = _parse(path, issues)
    if root_el is None:
        return finish_scan('cvat', root, [], issues)
    image_els = root_el.findall('image')
    if len(image_els) > preview_max_files():
        return scan_too_large('cvat', root, issues)
    images_dirs = [d for d in (root / 'images', root) if d.is_dir()]
    if path_guard is not None and not all(path_guard(d) for d in images_dirs):
        raise DatasetPathNotAllowedError(str(root / 'images'))
    ctx = _Ctx(root, images_dirs, issues, path_guard, SizeBudget())
    entries = [e for e in (_read_image(el, ctx) for el in image_els) if e is not None]
    return finish_scan('cvat', root, entries, issues)


__all__ = ['is_cvat', 'scan_cvat']
