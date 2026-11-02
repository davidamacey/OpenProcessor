"""Pascal VOC dataset reading.

Layout (the root is the ``VOC2012``-style directory, not ``VOCdevkit``)::

    root/Annotations/*.xml            (or annotations/)
    root/JPEGImages/*.jpg             (else images/, else the root)
    root/ImageSets/Main/{train,val,test}.txt   (optional, for splits)

The classes are the ``<name>`` texts. VOC has no class index, so
``DatasetScan.class_ids`` stays empty. The XML is untrusted: it is size-capped
before it is read and parsed with every DTD refused
(:mod:`~src.services.curation.dataset_import.safe_xml`); ``<filename>``
resolves through :func:`~src.services.curation.dataset_import.paths.resolve_ref`
and ``<folder>`` / ``<path>`` (the annotator's machine) are never used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.services.curation.dataset_import.box_norm import CLAMP_TOLERANCE_PX, normalize_box
from src.services.curation.dataset_import.capped_io import read_bytes_capped
from src.services.curation.dataset_import.coco import PREVIEW_SIZE_CHECKS
from src.services.curation.dataset_import.issues import IssueCollector
from src.services.curation.dataset_import.limits import MAX_VOC_XML_BYTES, preview_max_files
from src.services.curation.dataset_import.native_common import (
    entry_state,
    read_split_lists,
    rel_path,
)
from src.services.curation.dataset_import.paths import (
    DatasetPathNotAllowedError,
    PathGuard,
    resolve_ref,
)
from src.services.curation.dataset_import.safe_xml import parse_xml_bytes
from src.services.curation.dataset_import.scan import (
    DatasetScan,
    FormatUndetectedError,
    LabelBox,
    ScanEntry,
    class_box_counts,
    exif_transposed_size,
)
from src.services.curation.dataset_import.yolo import IMAGE_EXTENSIONS


if TYPE_CHECKING:
    import xml.etree.ElementTree as ET  # nosec B405  typing only; parsing goes through safe_xml
    from pathlib import Path

_ANNOTATION_DIRS = ('Annotations', 'annotations')


def _annotation_dir(root: Path) -> Path | None:
    for name in _ANNOTATION_DIRS:
        if (root / name).is_dir():
            return root / name
    return None


def is_voc(root: Path) -> bool:
    """``Annotations/`` (or ``annotations/``) holds XML and no JSON."""
    if not root.is_dir():
        return False
    ann = _annotation_dir(root)
    if ann is None:
        return False
    return any(ann.glob('*.xml')) and not any(ann.glob('*.json'))


@dataclass
class _Ctx:
    root: Path
    images_dir: Path
    issues: IssueCollector
    path_guard: PathGuard | None
    splits: dict[str, str]
    size_checks_left: int


def _images_dir(root: Path) -> Path:
    for name in ('JPEGImages', 'images'):
        if (root / name).is_dir():
            return root / name
    return root


def _voc_image(el: ET.Element, xml_path: Path, ctx: _Ctx) -> Path | None:
    """The existing image file the annotation describes, or ``None``."""
    filename = (el.findtext('filename') or '').strip()
    found: Path | None = None
    if filename:
        candidate = resolve_ref(ctx.images_dir, filename)
        if candidate is not None and candidate.is_file():
            found = candidate
    if found is None:
        for ext in sorted(IMAGE_EXTENSIONS):
            candidate = resolve_ref(ctx.images_dir, f'{xml_path.stem}{ext}')
            if candidate is not None and candidate.is_file():
                found = candidate
                break
    if found is None:
        ctx.issues.add(
            'image_file_missing', file=rel_path(xml_path, ctx.root), detail={'filename': filename}
        )
        return None
    if ctx.path_guard is not None and not ctx.path_guard(found):
        ctx.issues.add('image_path_not_servable', file=rel_path(xml_path, ctx.root))
        return None
    return found


def _declared_size(el: ET.Element) -> tuple[int, int] | None:
    try:
        w = int(float(el.findtext('size/width') or ''))
        h = int(float(el.findtext('size/height') or ''))
    except (ValueError, OverflowError):
        return None
    return (w, h) if w > 0 and h > 0 else None


def _voc_size(
    el: ET.Element, image: Path, rel_xml: str, ctx: _Ctx
) -> tuple[tuple[int, int], tuple[int, int] | None] | None:
    """``(size used for normalizing, declared size to carry)``, or ``None``
    when the image must be skipped."""
    declared = _declared_size(el)
    if declared is not None:
        if ctx.size_checks_left > 0:
            ctx.size_checks_left -= 1
            actual = exif_transposed_size(image)
            if actual is not None and actual != declared:
                ctx.issues.add(
                    'image_size_mismatch',
                    file=rel_xml,
                    detail={'declared': list(declared), 'actual': list(actual)},
                )
                return None
        return declared, declared
    ctx.issues.add('voc_size_missing', file=rel_xml)
    actual = exif_transposed_size(image)
    if actual is None:
        ctx.issues.add('image_unreadable', file=rel_xml)
        return None
    return actual, None


def _voc_box(
    obj: ET.Element, index: int, size: tuple[int, int], rel_xml: str, ctx: _Ctx
) -> LabelBox | None:
    """The :class:`LabelBox` of one non-difficult object, or ``None`` after
    recording why not."""
    name = (obj.findtext('name') or '').strip()
    if not name:
        ctx.issues.add(
            'label_row_malformed', file=rel_xml, detail={'object': index, 'reason': 'no name'}
        )
        return None
    try:
        xmin, ymin, xmax, ymax = (
            float((obj.findtext(f'bndbox/{key}') or '').strip())
            for key in ('xmin', 'ymin', 'xmax', 'ymax')
        )
    except ValueError:
        ctx.issues.add(
            'label_row_malformed', file=rel_xml, detail={'object': index, 'reason': 'bad bndbox'}
        )
        return None
    # VOC pixels are 1-based and inclusive; the 1 px tolerance absorbs 0-based writers.
    norm = normalize_box(
        xmin - 1,
        ymin - 1,
        xmax,
        ymax,
        width=size[0],
        height=size[1],
        tolerance=CLAMP_TOLERANCE_PX,
    )
    if norm.box is None:
        code = (
            'label_coords_out_of_range'
            if norm.problem == 'out_of_range'
            else 'label_box_degenerate'
        )
        ctx.issues.add(code, file=rel_xml, detail={'object': index})
        return None
    return LabelBox(dataset_class=name, bbox_norm=norm.box)


def _voc_objects(
    root_el: ET.Element, size: tuple[int, int], rel_xml: str, ctx: _Ctx
) -> tuple[list[LabelBox], bool]:
    boxes: list[LabelBox] = []
    objects = root_el.findall('object')
    for i, obj in enumerate(objects):
        if (obj.findtext('difficult') or '0').strip() == '1':
            ctx.issues.add('voc_difficult_skipped', file=rel_xml, detail={'object': i})
            continue
        box = _voc_box(obj, i, size, rel_xml, ctx)
        if box is not None:
            boxes.append(box)
    return boxes, bool(objects)


def _parse_file(xml_path: Path, rel_xml: str, issues: IssueCollector) -> ET.Element | None:
    raw = read_bytes_capped(xml_path, MAX_VOC_XML_BYTES)
    if raw is None:
        if xml_path.is_file() and xml_path.stat().st_size > MAX_VOC_XML_BYTES:
            issues.add('dataset_file_too_large', file=rel_xml)
        else:
            issues.add('voc_xml_invalid', file=rel_xml, detail={'error': 'unreadable'})
        return None
    try:
        root_el = parse_xml_bytes(raw)
    except ValueError as exc:
        issues.add('voc_xml_invalid', file=rel_xml, detail={'error': str(exc)[:200]})
        return None
    if root_el.tag != 'annotation':
        issues.add(
            'voc_xml_invalid',
            file=rel_xml,
            detail={'reason': 'root element is not <annotation>'},
        )
        return None
    return root_el


def _read_voc_file(xml_path: Path, ctx: _Ctx) -> ScanEntry | None:
    rel_xml = rel_path(xml_path, ctx.root)
    root_el = _parse_file(xml_path, rel_xml, ctx.issues)
    if root_el is None:
        return None
    image = _voc_image(root_el, xml_path, ctx)
    if image is None:
        return None
    sized = _voc_size(root_el, image, rel_xml, ctx)
    if sized is None:
        return None
    size, declared = sized
    boxes, has_objects = _voc_objects(root_el, size, rel_xml, ctx)
    return ScanEntry(
        rel_path=rel_path(image, ctx.root),
        source_stem=image.stem,
        abs_image_path=image,
        split=ctx.splits.get(xml_path.stem),
        label_state=entry_state(has_objects, boxes),
        boxes=boxes,
        declared_size=declared,
    )


def scan_voc(root: Path, *, path_guard: PathGuard | None = None) -> DatasetScan:
    if path_guard is not None and not path_guard(root):
        raise DatasetPathNotAllowedError(str(root))
    ann_dir = _annotation_dir(root)
    if ann_dir is None:
        raise FormatUndetectedError(str(root))
    issues = IssueCollector()
    xml_files = sorted(ann_dir.glob('*.xml'))
    if len(xml_files) > preview_max_files():
        issues.add('dataset_too_large', file=root.name)
        return DatasetScan(format='voc', root=root, entries=[], issues=issues)
    images_dir = _images_dir(root)
    if path_guard is not None and not path_guard(images_dir):
        raise DatasetPathNotAllowedError(str(images_dir))
    ctx = _Ctx(
        root=root,
        images_dir=images_dir,
        issues=issues,
        path_guard=path_guard,
        splits=read_split_lists(root / 'ImageSets' / 'Main', issues),
        size_checks_left=PREVIEW_SIZE_CHECKS,
    )
    entries = [e for e in (_read_voc_file(p, ctx) for p in xml_files) if e is not None]
    return DatasetScan(
        format='voc',
        root=root,
        entries=entries,
        issues=issues,
        class_ids={},
        class_box_counts=class_box_counts(entries),
    )


__all__ = ['is_voc', 'scan_voc']
