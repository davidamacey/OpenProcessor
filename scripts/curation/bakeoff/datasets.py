"""Convert labelled datasets into the YOLO test layout the harness reads.

The harness scores any ``images/<split>`` + ``labels/<split>`` tree, so a
cross-dataset matrix only needs per-format converters. Converters live in
a registry (:func:`register_converter`); each one reads a source tree and
hands boxes to a :class:`YoloWriter`, which owns the output layout (images
symlinked, no copy; ``<stem>.txt`` labels in normalized ``cls cx cy w h``)
and a ``data.yaml`` whose class names come from the active
:class:`~scripts.curation.bakeoff.profile.BakeoffProfile`.

Built-in, domain-neutral converters:

* ``yolo`` --- already-YOLO datasets: symlink images, remap/filter labels.
* ``voc``  --- Pascal VOC XML; ``<object><name>`` maps to a class id.

Domain-specific converters register themselves from a profile's
``converter_modules`` (an example profile under ``examples/<name>/`` can ship
the converters for its domain's public benchmark formats).

Class-id policy (both built-ins): when the profile's label space has a
single class, every source box collapses onto ``target_class_id`` (a
single-class benchmark of a multi-class source); otherwise source ids /
names are kept and anything outside the label space is dropped.

CLI:
    python -m scripts.curation.bakeoff.datasets \
        --profile my_profile.json --format voc --src /data/raw/set --out /data/bench/set
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from .profile import BakeoffProfile, load_converter_plugins, resolve_profile


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


IMG_EXTS = ('.jpg', '.jpeg', '.png')

AbsBox = tuple[float, float, float, float]
YoloBox = tuple[int, float, float, float, float]


def abs_xyxy_to_yolo(box: AbsBox, w: int, h: int) -> tuple[float, float, float, float] | None:
    """Absolute ``(x1,y1,x2,y2)`` px -> normalized ``(cx,cy,w,h)`` or None."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1 or w <= 0 or h <= 0:
        return None
    return (((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h, (x2 - x1) / w, (y2 - y1) / h)


class YoloWriter:
    """Writes one YOLO split plus ``data.yaml`` for a given label space.

    Args:
        out_root: Output dataset root.
        class_names: Label space, index = class id (``profile.class_names``).
        split: Split directory name.
        target_class_id: Class id used for single-class sources and as the
            collapse target when ``class_names`` has one entry.
    """

    def __init__(
        self,
        out_root: Path,
        class_names: Sequence[str],
        *,
        split: str = 'test',
        target_class_id: int = 0,
    ) -> None:
        if not class_names:
            raise ValueError('YoloWriter needs at least one class name')
        if not 0 <= target_class_id < len(class_names):
            raise ValueError(f'target_class_id {target_class_id} outside nc={len(class_names)}')
        self.out_root = out_root
        self.class_names = tuple(class_names)
        self.split = split
        self.target_class_id = target_class_id
        self._by_name = {n.strip().lower(): i for i, n in enumerate(self.class_names)}

    @classmethod
    def for_profile(cls, out_root: Path, profile: BakeoffProfile, *, split: str) -> YoloWriter:
        return cls(out_root, profile.class_names or ('object',), split=split)

    @property
    def nc(self) -> int:
        return len(self.class_names)

    @property
    def single_class(self) -> bool:
        return self.nc == 1

    def map_class_id(self, source_id: int) -> int | None:
        """Source class id -> output id (collapse if single-class, else keep/drop)."""
        if self.single_class:
            return self.target_class_id
        return source_id if 0 <= source_id < self.nc else None

    def map_class_name(self, source_name: str | None) -> int | None:
        """Source class name -> output id (collapse if single-class, else lookup/drop)."""
        if self.single_class:
            return self.target_class_id
        if source_name is None:
            return None
        return self._by_name.get(source_name.strip().lower())

    def write(self, stem: str, src_img: Path, boxes: Sequence[YoloBox]) -> None:
        """Symlink ``src_img`` and write its label file (empty => background)."""
        for cls_id, *_ in boxes:
            if not 0 <= cls_id < self.nc:
                raise ValueError(f'class id {cls_id} outside nc={self.nc} for {stem}')
        img_dir = self.out_root / 'images' / self.split
        lbl_dir = self.out_root / 'labels' / self.split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        link = img_dir / f'{stem}{src_img.suffix}'
        if not link.exists():
            link.symlink_to(src_img.resolve())
        lines = [f'{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}' for c, cx, cy, bw, bh in boxes]
        (lbl_dir / f'{stem}.txt').write_text(('\n'.join(lines) + '\n') if lines else '')

    def write_abs(
        self,
        stem: str,
        src_img: Path,
        abs_boxes: Sequence[AbsBox],
        w: int,
        h: int,
        *,
        class_id: int | None = None,
    ) -> None:
        """Convenience for single-class formats: abs xyxy boxes -> one class id."""
        cid = self.target_class_id if class_id is None else class_id
        yolo = [(cid, *b) for b in (abs_xyxy_to_yolo(bx, w, h) for bx in abs_boxes) if b]
        self.write(stem, src_img, yolo)

    def write_data_yaml(self) -> Path:
        names = ''.join(f'  {i}: {n}\n' for i, n in enumerate(self.class_names))
        path = self.out_root / 'data.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f'path: {self.out_root}\ntrain: images/{self.split}\nval: images/{self.split}\n'
            f'test: images/{self.split}\nnc: {self.nc}\nnames:\n{names}'
        )
        return path


@dataclass(frozen=True)
class DatasetConverter:
    """A registered source format. ``fn(src, writer) -> n_images_written``."""

    name: str
    fn: Callable[[Path, YoloWriter], int]
    description: str = ''
    example_for: str | None = None


_CONVERTERS: dict[str, DatasetConverter] = {}


def register_converter(
    name: str,
    fn: Callable[[Path, YoloWriter], int],
    *,
    description: str = '',
    example_for: str | None = None,
    replace: bool = False,
) -> DatasetConverter:
    """Register a converter under ``name`` (error on a clash unless ``replace``)."""
    existing = _CONVERTERS.get(name)
    if existing is not None and not replace and existing.fn is not fn:
        raise ValueError(f'dataset converter {name!r} already registered')
    conv = DatasetConverter(name, fn, description, example_for)
    _CONVERTERS[name] = conv
    return conv


def available_converters() -> dict[str, DatasetConverter]:
    return dict(_CONVERTERS)


def get_converter(name: str) -> DatasetConverter:
    try:
        return _CONVERTERS[name]
    except KeyError:
        raise ValueError(
            f'unknown dataset format {name!r}; registered: {", ".join(sorted(_CONVERTERS))} '
            "(domain formats come from a profile's converter_modules)"
        ) from None


def iter_images(src: Path) -> list[Path]:
    out: list[Path] = []
    for ext in IMG_EXTS:
        out.extend(src.rglob(f'*{ext}'))
    return sorted(out)


def image_size(img: Path) -> tuple[int, int] | None:
    """``(width, height)`` of an image, or None if unreadable."""
    arr = cv2.imread(str(img))
    if arr is None:
        return None
    h, w = arr.shape[:2]
    return w, h


# --- built-in: Pascal VOC --------------------------------------------------


def parse_voc_xml(text: str) -> tuple[tuple[int, int] | None, list[tuple[str | None, AbsBox]]]:
    """Parse a Pascal VOC annotation.

    Returns ``((width, height) | None, [(object_name, (x1,y1,x2,y2) abs)])``.
    Size may be None if the XML omits it (then the caller reads the image).
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(text)  # nosec B314 - parsing local dataset files
    except ET.ParseError:
        return None, []
    size_el = root.find('size')
    size: tuple[int, int] | None = None
    if size_el is not None:
        w_el, h_el = size_el.find('width'), size_el.find('height')
        if w_el is not None and h_el is not None and w_el.text and h_el.text:
            size = (int(float(w_el.text)), int(float(h_el.text)))
    boxes: list[tuple[str | None, AbsBox]] = []
    for obj in root.findall('object'):
        bb = obj.find('bndbox')
        if bb is None:
            continue
        try:
            x1 = float(bb.findtext('xmin'))  # type: ignore[arg-type]
            y1 = float(bb.findtext('ymin'))  # type: ignore[arg-type]
            x2 = float(bb.findtext('xmax'))  # type: ignore[arg-type]
            y2 = float(bb.findtext('ymax'))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        boxes.append((obj.findtext('name'), (x1, y1, x2, y2)))
    return size, boxes


def convert_voc(src: Path, writer: YoloWriter) -> int:
    """Pascal VOC. XML co-located with the image or in a sibling ``annotations/``."""
    n = 0
    for img in iter_images(src):
        ann = img.with_suffix('.xml')
        if not ann.is_file():
            ann = img.parent.parent / 'annotations' / f'{img.stem}.xml'
        if not ann.is_file():
            continue
        size, named = parse_voc_xml(ann.read_text(encoding='utf-8', errors='ignore'))
        dims = size or image_size(img)
        if dims is None:
            continue
        w, h = dims
        boxes: list[YoloBox] = []
        for obj_name, box in named:
            cid = writer.map_class_name(obj_name)
            yolo = abs_xyxy_to_yolo(box, w, h)
            if cid is not None and yolo is not None:
                boxes.append((cid, *yolo))
        writer.write(img.stem, img, boxes)
        n += 1
    return n


# --- built-in: YOLO passthrough ---------------------------------------------


def convert_yolo_passthrough(src: Path, writer: YoloWriter) -> int:
    """Datasets already in YOLO format: symlink images, remap/filter class ids."""
    n = 0
    for img in iter_images(src):
        # YOLO layout keeps labels in a sibling ``labels/`` dir
        # (``.../images/x.jpg`` -> ``.../labels/x.txt``); fall back to a label
        # sitting next to the image for flat layouts.
        sibling = Path(str(img.parent).replace('/images', '/labels')) / f'{img.stem}.txt'
        lbl = sibling if sibling.is_file() else img.with_suffix('.txt')
        lines = lbl.read_text(encoding='utf-8').splitlines() if lbl.is_file() else []
        boxes: list[YoloBox] = []
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                src_id = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:])
            except ValueError:
                continue
            cid = writer.map_class_id(src_id)
            if cid is not None:
                boxes.append((cid, cx, cy, bw, bh))
        writer.write(img.stem, img, boxes)
        n += 1
    return n


register_converter('voc', convert_voc, description='Pascal VOC XML (object name -> class id)')
register_converter('yolo', convert_yolo_passthrough, description='YOLO txt passthrough')


def convert(
    fmt: str,
    src: Path,
    out_root: Path,
    profile: BakeoffProfile,
    *,
    split: str = 'test',
) -> int:
    """Run the ``fmt`` converter under ``profile``; write ``data.yaml``; return count."""
    load_converter_plugins(profile)
    converter = get_converter(fmt)
    writer = YoloWriter.for_profile(out_root, profile, split=split)
    n = converter.fn(src, writer)
    writer.write_data_yaml()
    return n


def main(argv: list[str] | None = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--profile', default=None)
    known, _ = pre.parse_known_args(argv)
    profile = resolve_profile(known.profile)
    load_converter_plugins(profile)

    p = argparse.ArgumentParser(
        description='Convert a labelled dataset to the bake-off YOLO test layout.'
    )
    p.add_argument(
        '--profile',
        default=None,
        help='Bake-off profile: registered/example name or profile .json (default: generic)',
    )
    p.add_argument('--format', required=True, choices=sorted(available_converters()))
    p.add_argument('--src', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--split', default='test')
    args = p.parse_args(argv)
    n = convert(args.format, args.src, args.out, profile, split=args.split)
    print(
        f'converted {n} images ({args.format}, profile={profile.name}) '
        f'-> {args.out}/images/{args.split}'
    )
    if n == 0:
        print('WARNING: 0 images converted --- check --src layout / format')
    return 0


if __name__ == '__main__':
    # Run through the canonically-imported module: plugin converters register
    # into ``scripts.curation.bakeoff.datasets``, not into this ``__main__`` copy.
    from scripts.curation.bakeoff import datasets as _canonical

    raise SystemExit(_canonical.main())
