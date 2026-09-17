"""Convert public LPR datasets into the YOLO test layout the harness reads.

The harness scores any ``images/<split>`` + ``labels/<split>`` tree, so a
cross-dataset matrix only needs per-format converters. Implemented:

* ``ccpd``   --- CCPD encodes the plate bbox in the filename.
* ``ufpr``   --- UFPR-ALPR ships a per-image annotation ``.txt`` with a
  ``position_plate: x y w h`` line.
* ``voc``    --- Pascal VOC XML (the andrewmvd Kaggle car-plate set).
* ``yolo``   --- already-YOLO datasets (e.g. Roboflow exports): passthrough.

Output is a single-class (``license_plate``) YOLO dir with images symlinked
(no copy) and ``<stem>.txt`` labels in normalized ``cx cy w h``.

CLI:
    python -m scripts.curation.bakeoff.datasets \
        --format ccpd --src /data/raw/ccpd --out /data/bench/ccpd
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


_IMG_EXTS = ('.jpg', '.jpeg', '.png')
LPR_CLASS_ID = 0


def _abs_xyxy_to_yolo(
    box: tuple[float, float, float, float], w: int, h: int
) -> tuple[float, float, float, float] | None:
    """Absolute ``(x1,y1,x2,y2)`` px -> normalized ``(cx,cy,w,h)`` or None."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1 or w <= 0 or h <= 0:
        return None
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return (cx, cy, bw, bh)


def parse_ccpd_filename(name: str) -> tuple[float, float, float, float] | None:
    """Extract the plate bbox from a CCPD filename.

    CCPD encodes fields separated by ``-``; the 3rd field is the bounding
    box as ``x1&y1_x2&y2`` in absolute pixels, e.g.
    ``...-154&383_386&473-...``.
    """
    parts = Path(name).stem.split('-')
    if len(parts) < 3:
        return None
    try:
        tl, br = parts[2].split('_')
        x1, y1 = (float(v) for v in tl.split('&'))
        x2, y2 = (float(v) for v in br.split('&'))
    except (ValueError, IndexError):
        return None
    return (x1, y1, x2, y2)


def parse_voc_xml(
    text: str,
) -> tuple[tuple[int, int] | None, list[tuple[float, float, float, float]]]:
    """Parse a Pascal VOC annotation (the andrewmvd Kaggle format).

    Returns ``((width, height) | None, [ (x1,y1,x2,y2) abs ])``. Size may be
    None if the XML omits it (then the caller reads the image instead).
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(text)  # nosec B314 - parsing our own dataset files
    except ET.ParseError:
        return None, []
    size_el = root.find('size')
    size: tuple[int, int] | None = None
    if size_el is not None:
        w_el, h_el = size_el.find('width'), size_el.find('height')
        if w_el is not None and h_el is not None and w_el.text and h_el.text:
            size = (int(float(w_el.text)), int(float(h_el.text)))
    boxes: list[tuple[float, float, float, float]] = []
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
        boxes.append((x1, y1, x2, y2))
    return size, boxes


def parse_openalpr_annotation(text: str) -> list[tuple[float, float, float, float]]:
    """OpenALPR endtoend benchmark: ``<name> x y w h <plate_text>`` per line.

    x,y are the plate top-left and w,h its size (absolute px). Returns
    absolute ``(x1,y1,x2,y2)`` boxes (one per line).
    """
    boxes: list[tuple[float, float, float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            x, y, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if w > 0 and h > 0:
            boxes.append((x, y, x + w, y + h))
    return boxes


def parse_ufpr_annotation(text: str) -> list[tuple[float, float, float, float]]:
    """Extract plate boxes from a UFPR-ALPR annotation file.

    Lines of the form ``position_plate: x y w h`` (absolute px, top-left +
    size). Returns absolute ``(x1,y1,x2,y2)`` boxes.
    """
    boxes: list[tuple[float, float, float, float]] = []
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith('position_plate:'):
            nums = line.split(':', 1)[1].split()
            if len(nums) >= 4:
                try:
                    x, y, w, h = (float(v) for v in nums[:4])
                except ValueError:
                    continue
                boxes.append((x, y, x + w, y + h))
    return boxes


def _write(out_root: Path, stem: str, src_img: Path, boxes_yolo: list, *, split: str) -> None:
    img_dir = out_root / 'images' / split
    lbl_dir = out_root / 'labels' / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    link = img_dir / f'{stem}{src_img.suffix}'
    if not link.exists():
        link.symlink_to(src_img.resolve())
    lines = [f'{LPR_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}' for cx, cy, bw, bh in boxes_yolo]
    (lbl_dir / f'{stem}.txt').write_text(('\n'.join(lines) + '\n') if lines else '')


def _iter_images(src: Path) -> list[Path]:
    out: list[Path] = []
    for ext in _IMG_EXTS:
        out.extend(src.rglob(f'*{ext}'))
    return sorted(out)


def convert_ccpd(src: Path, out_root: Path, *, split: str = 'test') -> int:
    n = 0
    for img in _iter_images(src):
        box = parse_ccpd_filename(img.name)
        if box is None:
            continue
        dims = cv2.imread(str(img))
        if dims is None:
            continue
        h, w = dims.shape[:2]
        yolo = _abs_xyxy_to_yolo(box, w, h)
        if yolo is None:
            continue
        _write(out_root, img.stem, img, [yolo], split=split)
        n += 1
    return n


def convert_ufpr(src: Path, out_root: Path, *, split: str = 'test') -> int:
    n = 0
    for img in _iter_images(src):
        ann = img.with_suffix('.txt')
        if not ann.is_file():
            continue
        abs_boxes = parse_ufpr_annotation(ann.read_text(encoding='utf-8', errors='ignore'))
        dims = cv2.imread(str(img))
        if dims is None:
            continue
        h, w = dims.shape[:2]
        yolo = [b for b in (_abs_xyxy_to_yolo(bx, w, h) for bx in abs_boxes) if b]
        _write(out_root, img.stem, img, yolo, split=split)
        n += 1
    return n


def convert_openalpr(src: Path, out_root: Path, *, split: str = 'test') -> int:
    """OpenALPR endtoend benchmark (image + co-located ``.txt``)."""
    n = 0
    for img in _iter_images(src):
        ann = img.with_suffix('.txt')
        if not ann.is_file():
            continue
        abs_boxes = parse_openalpr_annotation(ann.read_text(encoding='utf-8', errors='ignore'))
        dims = cv2.imread(str(img))
        if dims is None:
            continue
        h, w = dims.shape[:2]
        yolo = [b for b in (_abs_xyxy_to_yolo(bx, w, h) for bx in abs_boxes) if b]
        _write(out_root, img.stem, img, yolo, split=split)
        n += 1
    return n


def convert_voc(src: Path, out_root: Path, *, split: str = 'test') -> int:
    """Pascal VOC (andrewmvd Kaggle). XML co-located or in sibling annotations/."""
    n = 0
    for img in _iter_images(src):
        ann = img.with_suffix('.xml')
        if not ann.is_file():
            ann = img.parent.parent / 'annotations' / f'{img.stem}.xml'
        if not ann.is_file():
            continue
        size, abs_boxes = parse_voc_xml(ann.read_text(encoding='utf-8', errors='ignore'))
        if size is not None:
            w, h = size
        else:
            dims = cv2.imread(str(img))
            if dims is None:
                continue
            h, w = dims.shape[:2]
        yolo = [b for b in (_abs_xyxy_to_yolo(bx, w, h) for bx in abs_boxes) if b]
        _write(out_root, img.stem, img, yolo, split=split)
        n += 1
    return n


def convert_yolo_passthrough(src: Path, out_root: Path, *, split: str = 'test') -> int:
    """Datasets already in YOLO format: symlink images + copy labels as-is."""
    n = 0
    for img in _iter_images(src):
        # YOLO layout keeps labels in a sibling ``labels/`` dir
        # (``.../images/x.jpg`` -> ``.../labels/x.txt``); fall back to a label
        # sitting next to the image for flat layouts.
        sibling = Path(str(img.parent).replace('/images', '/labels')) / f'{img.stem}.txt'
        lbl = sibling if sibling.is_file() else img.with_suffix('.txt')
        boxes_lines = lbl.read_text(encoding='utf-8').splitlines() if lbl.is_file() else []
        img_dir = out_root / 'images' / split
        lbl_dir = out_root / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        link = img_dir / f'{img.stem}{img.suffix}'
        if not link.exists():
            link.symlink_to(img.resolve())
        # Force every class id to 0 (single-class license_plate benchmark).
        norm = [f'0 {" ".join(ln.split()[1:])}' for ln in boxes_lines if len(ln.split()) == 5]
        (lbl_dir / f'{img.stem}.txt').write_text(('\n'.join(norm) + '\n') if norm else '')
        n += 1
    return n


_CONVERTERS = {
    'ccpd': convert_ccpd,
    'openalpr': convert_openalpr,
    'ufpr': convert_ufpr,
    'voc': convert_voc,
    'yolo': convert_yolo_passthrough,
}


def _write_data_yaml(out_root: Path) -> None:
    (out_root / 'data.yaml').write_text(
        f'path: {out_root}\ntrain: images/test\nval: images/test\ntest: images/test\n'
        'nc: 1\nnames:\n  0: license_plate\n'
    )


def main() -> int:
    p = argparse.ArgumentParser(description='Convert a public LPR dataset to YOLO test layout.')
    p.add_argument('--format', required=True, choices=sorted(_CONVERTERS))
    p.add_argument('--src', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--split', default='test')
    args = p.parse_args()
    n = _CONVERTERS[args.format](args.src, args.out, split=args.split)
    _write_data_yaml(args.out)
    print(f'converted {n} images ({args.format}) -> {args.out}/images/{args.split}')
    if n == 0:
        print('WARNING: 0 images converted --- check --src layout / format')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
